"""
The conversational agent layer.

Instead of driving the bot with buttons, the user talks to it in plain language
("find me some tech news and post the best one to Instagram"). The LLM (Gemini)
is given a set of TOOLS and decides, on its own, which ones to call and in what
order. This module implements that agent loop explicitly:

    user message
        -> LLM sees the conversation + the tool definitions
        -> LLM either replies, or emits a function_call
        -> we execute the tool and hand the result back to the LLM
        -> repeat until the LLM produces a final natural-language answer

The tools themselves live in bot.py; they are passed in as a `toolbox` dict so
this module has no circular dependency on the bot.
"""

import asyncio
import logging

from google.genai import types

logger = logging.getLogger(__name__)

import os

from dotenv import load_dotenv

load_dotenv()  # this module can be imported before the bot loads its config

# Free-tier quota is per project PER MODEL, so the model is configurable via .env.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_STEPS = 6          # safety guard against a runaway tool-calling loop
MAX_HISTORY = 20       # keep the last N turns of conversation

SYSTEM_PROMPT = """You are Agentic News, an autonomous news-publishing assistant.

You can search live news, check headline quality, write and publish articles, and
post them to a Telegram channel and Instagram. Use your tools to do what the user asks.

Important rules:
- You have a fine-tuned clickbait classifier. Headlines it flags as clickbait CANNOT be
  published. If the user picks one, explain that it was blocked and offer alternatives.
- Always call search_news before referring to headline numbers.
- Call prepare_story before publish_story - a story must be prepared first.
- Be brief and conversational. Do not use markdown formatting or asterisks.
- After publishing, tell the user exactly where it went and include the article link.
"""

# ---------------------------------------------------------------------------
# Tool definitions: what the LLM is allowed to call, and with what arguments.
# ---------------------------------------------------------------------------
FUNCTION_DECLARATIONS = [
    {
        "name": "search_news",
        "description": (
            "Fetch the latest live news headlines. Use 'category' for a broad section, or "
            "'query' to search a specific topic, person or keyword (e.g. 'trump', 'AI chips'). "
            "Provide exactly one of them. Each headline is checked by the clickbait classifier "
            "and returned with a quality verdict and a number the user can refer to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "A broad news section. Use this OR query, not both.",
                    "enum": ["technology", "business", "science", "entertainment", "sports", "general"],
                },
                "query": {
                    "type": "string",
                    "description": "Free-text topic/person/keyword to search for. Use this OR category.",
                },
            },
        },
    },
    {
        "name": "check_headline_quality",
        "description": (
            "Run any headline text through the fine-tuned DistilBERT clickbait classifier "
            "and return whether it is clickbait, with a confidence score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "The headline text to classify."}
            },
            "required": ["headline"],
        },
    },
    {
        "name": "prepare_story",
        "description": (
            "Prepare one of the headlines returned by search_news: write a curated headline, "
            "a short post and a full article, gather images, and publish the article webpage. "
            "Fails if the headline was flagged as clickbait. Must be called before publish_story."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "integer",
                    "description": "Which headline to prepare (1-5), as listed by search_news.",
                }
            },
            "required": ["number"],
        },
    },
    {
        "name": "publish_story",
        "description": (
            "Publish the story that was prepared by prepare_story to the chosen platforms."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "platforms": {
                    "type": "string",
                    "description": "Where to publish the prepared story.",
                    "enum": ["channel", "instagram", "both"],
                }
            },
            "required": ["platforms"],
        },
    },
]


def _config(declarations: list):
    return types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=declarations)],
        system_instruction=SYSTEM_PROMPT,
        # We run the tool loop ourselves so we can await async tools and log each step.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


async def run_agent(client, user_text: str, history: list, call_tool,
                    declarations: list | None = None) -> tuple[str, list]:
    """Run one conversational turn.

    `call_tool` is an async callable (name, args) -> dict. `declarations` are the tool
    schemas shown to the LLM; when they come from the MCP server the bot does not need
    to know its own tools in advance. Returns (reply_text, updated_history).
    """
    declarations = declarations or FUNCTION_DECLARATIONS
    history = (history or [])[-MAX_HISTORY:]
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    for step in range(MAX_STEPS):
        # The SDK call is blocking, so keep it off the bot's event loop.
        response = await asyncio.to_thread(
            client.models.generate_content, model=MODEL, contents=history,
            config=_config(declarations)
        )
        candidate = response.candidates[0]
        parts = candidate.content.parts or []

        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not calls:
            # No tool needed: the model has produced its final answer.
            reply = (response.text or "Sorry, I couldn't work out what to do there.").strip()
            history.append(candidate.content)
            return reply, history

        # Record the model's tool call, then execute each requested tool.
        history.append(candidate.content)
        for call in calls:
            name = call.name
            args = dict(call.args or {})
            logger.info(f"[agent] step {step + 1}: calling {name}({args})")
            try:
                result = await call_tool(name, args)
            except Exception as e:
                logger.error(f"[agent] tool {name} failed: {e}")
                result = {"error": str(e)}
            history.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=name, response={"result": result})],
            ))

    return ("That took too many steps, so I stopped. Try asking for one thing at a time.",
            history)

