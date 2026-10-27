#!/usr/bin/env python3
"""Renders a set of conversations with transformers' apply_chat_template, for tests/test_text.cpp to check Liyab's
chat template interpreter against, byte for byte.

    tools/chat_template_reference.py --out tests/data/chat_templates.json \\
        --model qwen3.5=Qwen/Qwen3.5-0.8B --model llama3.2=meta-llama/Llama-3.2-1B-Instruct \\
        --model gemma3=google/gemma-3-270m-it:gemma3.jinja ...

Each --model is NAME=TOKENIZER[:TEMPLATE]: a Hugging Face model id or local directory whose tokenizer supplies the
special tokens (bos_token, eos_token...) and, unless a TEMPLATE file is given (e.g. one written by
tools/gguf_vocab.py --template, as the GGUF carries it), the chat template. Only tokenizer files are read.
An explicit TEMPLATE uses transformers' Jinja renderer directly, including when a tokenizer backend implements
its chat protocol outside Jinja and ignores apply_chat_template's template argument.

For every template the same conversations are rendered: system / user / assistant turns, a conversation without
a system message, replies carrying reasoning (inline <think> blocks and reasoning_content), tools with tool calls
and tool results, thinking on and off (enable_thinking, preserve_thinking), with and without a generation
prompt, non-ASCII and white space at the edges of messages. A template that refuses a conversation
(raise_exception, an error in the sandbox) is recorded as an error, which Liyab must also report. Templates that
read the clock (strftime_now) use 2024-07-26 12:00:00 as their UTC clock and a fixed date_string.
Their fixture stores clock_utc outside the request so the C++ test can supply the same clock.

The output is one JSON file: {name: {"template", "special", "cases": [{"request", "output" | "error"}]}}, where a
request is what Engine::render_chat takes.

Needs transformers (pip install transformers jinja2).
"""

import argparse
from datetime import datetime, timezone
import json

FIXED_CLOCK = datetime(2024, 7, 26, 12, 0, 0)
CLOCK_UTC = int(FIXED_CLOCK.replace(tzinfo=timezone.utc).timestamp())

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "add", "description": "Adds two numbers.",
                                      "parameters": {"type": "object", "properties": {"a": {"type": "number"},
                                                                                      "b": {"type": "number"}}}}},
]
SYSTEM = {"role": "system", "content": "You are a concise assistant."}
U1 = {"role": "user", "content": "What is the capital of Italy?"}
A1 = {"role": "assistant", "content": "Rome."}
U2 = {"role": "user", "content": "And of France? Answer in one word."}
A1_THINK = {"role": "assistant", "content": "<think>\nThe user asks about Italy.\n</think>\n\nRome."}
A1_REASONING = {"role": "assistant", "content": "Rome.", "reasoning_content": "Italy's capital is Rome."}
CALL = {"role": "assistant", "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]}
RESULT = {"role": "tool", "content": "{\"temp\": 21, \"sky\": \"clear\"}"}
WEATHER = {"role": "user", "content": "Weather in Rome?"}

CASES = [
    {"messages": [SYSTEM, U1], "add_generation_prompt": True},
    {"messages": [U1], "add_generation_prompt": True},
    {"messages": [SYSTEM, U1, A1, U2], "add_generation_prompt": True},
    {"messages": [SYSTEM, U1, A1, U2], "add_generation_prompt": False},
    {"messages": [SYSTEM, U1, A1]},
    {"messages": [SYSTEM, U1, A1_THINK, U2], "add_generation_prompt": True},
    {"messages": [SYSTEM, U1, A1_REASONING, U2], "add_generation_prompt": True},
    {"messages": [SYSTEM, U1], "add_generation_prompt": True, "enable_thinking": False},
    {"messages": [SYSTEM, U1], "add_generation_prompt": True, "enable_thinking": True},
    {"messages": [SYSTEM, U1, A1_THINK, U2], "add_generation_prompt": True, "preserve_thinking": True},
    {"messages": [SYSTEM, WEATHER], "tools": TOOLS, "add_generation_prompt": True},
    {"messages": [WEATHER, CALL, RESULT], "tools": TOOLS, "add_generation_prompt": True},
    {"messages": [{"role": "system", "content": "  Città: Zürich, 東京 🌍  \n"},
                  {"role": "user", "content": "\n  ¿Qué tal?  \t"}], "add_generation_prompt": True},
    {"messages": [{"role": "system", "content": ""}, U1], "add_generation_prompt": True},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="NAME=TOKENIZER[:TEMPLATE]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="run tokenizer code from trusted official model repositories")
    parser.add_argument("--slow-tokenizer", action="store_true", help="use the model's original slow tokenizer")
    args = parser.parse_args()

    from reference_tokenizer import load_tokenizer

    out = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        source, _, template_file = rest.partition(":")
        tok = load_tokenizer(source, trust_remote_code=args.trust_remote_code, slow=args.slow_tokenizer)
        template = open(template_file, encoding="utf-8").read() if template_file else tok.chat_template
        if isinstance(template, dict):
            template = template.get("default")
        if not isinstance(template, str):
            parser.error(f"{source} has no Jinja chat template; supply NAME=TOKENIZER:TEMPLATE")
        special = {k: getattr(tok, k) for k in ("bos_token", "eos_token", "unk_token", "pad_token")
                   if isinstance(getattr(tok, k, None), str)}
        cases = []
        for case in CASES:
            request = dict(case)
            if "strftime_now" in template:
                request["date_string"] = "26 Jul 2024"
            kwargs = {k: v for k, v in request.items() if k not in ("messages", "tools", "add_generation_prompt")}
            if "strftime_now" in template:
                kwargs["strftime_now"] = FIXED_CLOCK.strftime
            try:
                if template_file:
                    from transformers.utils.chat_template_utils import render_jinja_template
                    rendered, _ = render_jinja_template(
                        conversations=[request["messages"]], tools=request.get("tools"), chat_template=template,
                        add_generation_prompt=request.get("add_generation_prompt", False), **special, **kwargs)
                    text = rendered[0]
                else:
                    text = tok.apply_chat_template(request["messages"], tools=request.get("tools"), tokenize=False,
                                                   add_generation_prompt=request.get("add_generation_prompt", False),
                                                   chat_template=template, **kwargs)
                cases.append({"request": request, "output": text})
            except Exception as e:  # the template refuses the conversation: Liyab must refuse it too
                cases.append({"request": request, "error": f"{type(e).__name__}: {e}"})
        out[name] = {"template": template, "special": special, "cases": cases}
        if "strftime_now" in template:
            out[name]["clock_utc"] = CLOCK_UTC
        print(f"{name}: {sum('output' in c for c in cases)} rendered, {sum('error' in c for c in cases)} refused")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
