"""OpenAI tool-calling audit logic. Retrieved policies remain in memory only."""

import json
from typing import Any

from openai import OpenAI

from backend.config import settings
from chat_policy_audit import ASSESSMENT_SCHEMA, create_vectorstore, retrieve_policy


CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_policy",
        "description": "Retrieve the policy passages required to assess this customer-support chat.",
        "parameters": {
            "type": "object",
            "properties": {
                "retrieval_query": {
                    "type": "string",
                    "description": "A focused semantic retrieval query covering every policy issue in the chat.",
                }
            },
            "required": ["retrieval_query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

AUDIO_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_policy",
        "description": "Retrieve policy passages required to assess a transcribed customer-service audio recording.",
        "parameters": {
            "type": "object",
            "properties": {
                "retrieval_query": {
                    "type": "string",
                    "description": "A focused semantic retrieval query covering every policy issue in the transcription.",
                }
            },
            "required": ["retrieval_query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

SYSTEM_PROMPT = """You are an airline customer-service quality auditor.
You MUST call retrieve_policy exactly once before producing an assessment, it is used to
retrieve the relevant policy passages using RAG. Build a focused retrieval query from all 
material customer issues, agent actions, and risks, which will be used to retrieve policy passages.
After the tool provides policy passages, assess only the supplied case
and those passages. Do not invent policy. Return only the required JSON object."""


def transcribe_audio(file_path: str) -> str:
    """Send the actual audio file to OpenAI and return its transcription."""
    client = OpenAI()
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=settings.openai_transcription_model,
            file=audio_file,
        )
    return response.text


def assess_case(case_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Force an appropriate RAG tool call and return a structured assessment.

    Policy chunks are passed only to the final LLM call and are never persisted.
    """
    client = OpenAI()
    tool = AUDIO_TOOL if case_type == "audio" else CHAT_TOOL
    initial = client.chat.completions.create(
        model=settings.openai_audit_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "retrieve_policy"}},
        parallel_tool_calls=False,
    )
    assistant_message = initial.choices[0].message
    if not assistant_message.tool_calls:
        raise RuntimeError("The model did not call the required policy-retrieval tool.")
    tool_call = assistant_message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    query = arguments["retrieval_query"]
    vectorstore = create_vectorstore(settings.vector_db_dir, "langchain")
    policy_chunks = retrieve_policy(vectorstore, query, 4)

    final = client.chat.completions.create(
        model=settings.openai_audit_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"policy_chunks": policy_chunks}, ensure_ascii=False),
            },
        ],
        response_format={"type": "json_schema", "json_schema": ASSESSMENT_SCHEMA},
    )
    if not final.choices[0].message.content:
        raise RuntimeError("The model returned an empty assessment.")
    return json.loads(final.choices[0].message.content)
