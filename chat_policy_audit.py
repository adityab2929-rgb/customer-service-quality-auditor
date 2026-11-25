"""Audit customer-support chats against retrieved policy chunks.

Run from the project folder:
    uv run python chat_policy_audit.py

Or, without uv:
    .venv\\Scripts\\python.exe chat_policy_audit.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHAT_DIR = PROJECT_DIR / "customer-chats"
DEFAULT_VECTOR_DB_DIR = PROJECT_DIR / "vector-db"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "audit-results"
DEFAULT_POLICY_DIR = PROJECT_DIR / "policies"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ANSI terminal colours: cyan = starting, yellow = retrieving, green = finished.
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "retrieve_policy",
        "description": (
            "Retrieve the policy passages needed to assess this customer-support chat. "
            "First identify every customer issue, question, risk, and agent action that "
            "needs to be checked. Then write a focused retrieval query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "retrieval_query": {
                    "type": "string",
                    "description": (
                        "A precise semantic-search query containing the policy topics and "
                        "facts required to evaluate the chat."
                        "that captures every issue, question, risk, and relevant agent action in the chat." 
                    ),
                },
            },
            "required": ["retrieval_query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


ASSESSMENT_SCHEMA = {
    "name": "customer_chat_policy_assessment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "customer_issue": {
                "type": "string",
                "description": "Main customer issue; include multiple issues when needed.",
            },
            "customer_sentiment": {
                "type": "string",
                "description": "Best-supported customer sentiment.",
            },
            "sentiment_score": {
                "type": "number",
                "description": "Confidence from 0.0 to 1.0 in the sentiment label.",
            },
            "agent_score": {
                "type": "integer",
                "description": "Support quality score from 1 (poor) to 10 (excellent).",
            },
            "escalation_risk": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Risk that the case needs urgent escalation.",
            },
            "policy_status": {
                "type": "string",
                "enum": [
                    "compliant",
                    "possible_violation",
                    "violation",
                    "insufficient_policy",
                ],
                "description": "Whether the retrieved policy supports the agent's handling.",
            },
            "evidence_quote": {
                "type": "string",
                "description": "Short, exact quote from the chat supporting the assessment.",
            },
            "policy_source": {
                "type": "string",
                "description": "Retrieved policy filename/title and section when available.",
            },
            "coaching_suggestion": {
                "type": "string",
                "description": "Specific, practical next-step coaching for the agent.",
            },
        },
        "required": [
            "customer_issue",
            "customer_sentiment",
            "sentiment_score",
            "agent_score",
            "escalation_risk",
            "policy_status",
            "evidence_quote",
            "policy_source",
            "coaching_suggestion",
        ],
        "additionalProperties": False,
    },
}


SYSTEM_PROMPT = """You are a quality-assurance analyst for airline customer support.

You will receive one complete customer-support chat. You MUST call the
`retrieve_policy` tool exactly once before assessing it completely, to get the
relevant policy chunks needed to assess the chat. In the tool arguments,
write a semantic retrieval_query that captures every issue, question, risk, 
and relevant agent action in the chat. Do not include any other text in the query.

After the tool returns policy excerpts, evaluate the chat only against those
excerpts and the chat itself. Do not invent policy. Assess the customer issue,
sentiment, escalation risk, whether the agent followed policy, an exact chat
quote as evidence, the policy source, and useful coaching. The final answer must
be the requested JSON object and nothing else.
"""


def log(colour: str, message: str) -> None:
    print(f"{colour}{message}{RESET}")


def chat_to_transcript(chat: dict[str, Any]) -> str:
    """Turn the JSON chat format into explicit speaker-labelled text."""
    lines = []
    for message in chat.get("messages", []):
        speaker = str(message.get("speaker", "unknown")).strip().title()
        text = str(message.get("text", "")).strip()
        if text:
            lines.append(f"{speaker}: {text}")
    if not lines:
        raise ValueError("The chat has no usable messages.")
    return "\n".join(lines)


def create_vectorstore(vector_db_dir: Path, collection_name: str) -> Chroma:
    """Open the existing Chroma collection using its original embedding model."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(vector_db_dir),
        embedding_function=embeddings,
    )


def retrieve_policy(vectorstore: Chroma, query: str, k: int) -> list[dict[str, Any]]:
    """Return compact, source-labelled chunks for the model's required tool output."""
    matches = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", "Unknown policy source"),
            "relevance_score": round(float(score), 4),
        }
        for document, score in matches
    ]


def assess_chat(
    client: OpenAI,
    vectorstore: Chroma,
    chat: dict[str, Any],
    model: str,
    policy_chunks: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require a retrieval tool call, then return one schema-validated assessment."""
    transcript = chat_to_transcript(chat)
    user_message = (
        "Assess this customer-support chat. First call retrieve_policy.\n\n"
        f"CHAT JSON:\n{json.dumps(chat, ensure_ascii=False)}\n\n"
        f"CHAT TRANSCRIPT:\n{transcript}"
    )
    first_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tools=[TOOL_SPEC],
        tool_choice={"type": "function", "function": {"name": "retrieve_policy"}},
        parallel_tool_calls=False,
    )

    assistant_message = first_response.choices[0].message
    if not assistant_message.tool_calls:
        raise RuntimeError("The model did not call the required retrieve_policy tool.")
    tool_call = assistant_message.tool_calls[0]
    if tool_call.function.name != "retrieve_policy":
        raise RuntimeError("The model called an unexpected tool.")

    tool_arguments = json.loads(tool_call.function.arguments)
    query = tool_arguments["retrieval_query"]
    policy = retrieve_policy(vectorstore, query, policy_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        assistant_message,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(
                {"policy_chunks": policy}, ensure_ascii=False
            ),
        },
    ]
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": ASSESSMENT_SCHEMA},
    )
    content = final_response.choices[0].message.content
    if not content:
        raise RuntimeError("The model returned an empty final assessment.")
    return json.loads(content), policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve policy for each customer chat and produce a JSON assessment."
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--chat-dir", type=Path, default=DEFAULT_CHAT_DIR)
    parser.add_argument("--vector-db-dir", type=Path, default=DEFAULT_VECTOR_DB_DIR)
    parser.add_argument("--collection", default="langchain")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    parser.add_argument("--policy-chunks", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to the project's .env file.")
    if not args.chat_dir.is_dir():
        raise FileNotFoundError(f"Chat folder not found: {args.chat_dir}")
    if not args.vector_db_dir.is_dir():
        raise FileNotFoundError(f"Vector database folder not found: {args.vector_db_dir}")
    if args.policy_chunks < 1:
        raise ValueError("--policy-chunks must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.policy_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI()
    vectorstore = create_vectorstore(args.vector_db_dir, args.collection)
    chat_files = sorted(args.chat_dir.glob("*.json"))
    if not chat_files:
        raise FileNotFoundError(f"No JSON chat files found in {args.chat_dir}")

    for index, chat_path in enumerate(chat_files, start=1):
        log(CYAN, f"[{index}/{len(chat_files)}] Processing chat: {chat_path.name}")
        try:
            with chat_path.open(encoding="utf-8") as file:
                chat = json.load(file)

                assessment, relevant_policies = assess_chat(
                    client, vectorstore, chat, args.model, args.policy_chunks
                )

                output_policy_path = args.policy_dir / f"{chat_path.stem}_policies.json"
                with output_policy_path.open("w", encoding="utf-8") as file:
                    json.dump(relevant_policies, file, indent=2, ensure_ascii=False)
                log(YELLOW, f"RAG retrieved: {chat["agent_id"]}-{chat["chat_number"]}-{chat["issue"]} -> {output_policy_path.name}")
                output_path = args.output_dir / f"{chat_path.stem}_assessment.json"
                with output_path.open("w", encoding="utf-8") as file:
                    json.dump(assessment, file, indent=2, ensure_ascii=False)
                log(GREEN, f"Completed: {chat_path.name} -> {output_path.name}")
        except Exception as error:
            log(RED, f"Failed: {chat_path.name} | {error}")


if __name__ == "__main__":
    main()
