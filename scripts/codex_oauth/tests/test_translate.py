"""Tests for codex_oauth.translate — Chat Completions <-> Responses translation
and the SSE terminal-event / truncation guards."""

from __future__ import annotations

import json

from codex_oauth import translate as x


# --- chat -> responses -------------------------------------------------------
def test_chat_to_responses_maps_system_to_instructions_and_messages_to_input():
    chat = {
        "model": "openai/gpt-6.1-sol",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "developer", "content": "no markdown"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        "max_tokens": 128,
        "stream": True,
    }
    out = x.chat_to_responses(chat)
    assert out["model"] == "openai/gpt-6.1-sol"
    assert out["instructions"] == "be terse\n\nno markdown"
    # user + assistant become input items (system/developer do NOT)
    roles = [(i["role"], i["content"][0]["type"]) for i in out["input"]]
    assert roles == [("user", "input_text"), ("assistant", "output_text")]
    assert out["max_output_tokens"] == 128
    assert out["stream"] is True


def test_chat_to_responses_flattens_list_content_parts():
    chat = {"model": "m", "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "input_text", "text": "b"}]}]}
    out = x.chat_to_responses(chat)
    assert out["input"][0]["content"][0]["text"] == "ab"


# --- responses -> chat (unary) ----------------------------------------------
def test_responses_to_chat_extracts_message_text_and_usage():
    resp = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "the answer"}]},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    }
    chat = x.responses_to_chat(resp, "openai/gpt-6.1-sol", created=1000)
    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"]["content"] == "the answer"
    assert chat["choices"][0]["finish_reason"] == "stop"
    assert chat["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_responses_to_chat_maps_function_call_to_tool_calls():
    resp = {"status": "completed", "output": [
        {"type": "function_call", "call_id": "c1", "name": "do", "arguments": "{}"}]}
    chat = x.responses_to_chat(resp, "m", 0)
    choice = chat["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "do"
    assert choice["message"]["content"] is None


# --- terminal-event / truncation byte-tail guard (CRITICAL pitfall) ----------
def test_tail_has_terminal_event_detects_json_forms():
    assert x.tail_has_terminal_event(b'data: {"type":"response.completed"}')
    assert x.tail_has_terminal_event(b'data: {"type": "response.incomplete"}')
    assert x.tail_has_terminal_event(b"event: response.failed\ndata: {}")


def test_tail_has_terminal_event_does_not_false_latch_on_event_word_inside_delta_text():
    # A model streaming the literal words "event: response.completed" inside an
    # output_text delta MUST NOT be read as a terminal event: the bare `event:`
    # form is only honored line-anchored, and the JSON form requires the real
    # unescaped "type":"..." key (a delta escapes its quotes to \").
    poisoned = (b'data: {"type":"response.output_text.delta",'
                b'"delta":"here is some text event: response.completed yes"}')
    assert x.tail_has_terminal_event(poisoned) is False


def test_iter_responses_sse_as_chat_happy_path():
    lines = [
        b'data: {"type":"response.output_text.delta","delta":"Hel"}',
        b'data: {"type":"response.output_text.delta","delta":"lo"}',
        b'data: {"type":"response.completed","response":{"usage":'
        b'{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}',
    ]
    chunks = list(x.iter_responses_sse_as_chat(lines, "m", 0))
    joined = b"".join(chunks)
    assert b'"role": "assistant"' in joined or b'"role":"assistant"' in joined
    assert b"Hel" in joined and b"lo" in joined
    assert b'"finish_reason": "stop"' in joined or b'"finish_reason":"stop"' in joined
    assert joined.rstrip().endswith(b"[DONE]")


def test_iter_responses_sse_as_chat_emits_error_on_truncation():
    # Stream ends WITHOUT a terminal event -> must emit an error chunk, never a
    # clean finish=stop, so the client raises instead of recording a short turn.
    lines = [b'data: {"type":"response.output_text.delta","delta":"partial"}']
    chunks = list(x.iter_responses_sse_as_chat(lines, "m", 0))
    joined = b"".join(chunks)
    assert b'"error"' in joined
    assert b'"finish_reason": "stop"' not in joined and b'"finish_reason":"stop"' not in joined


def test_responses_truncation_error_sse_is_well_formed_failed_event():
    raw = x.responses_truncation_error_sse()
    assert raw.startswith(b"event: response.failed")
    body = raw.split(b"data: ", 1)[1]
    obj = json.loads(body)
    assert obj["type"] == "response.failed"
    assert "error" in obj["response"]
