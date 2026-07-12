"""
Agentic News - MCP server.

Exposes the news pipeline as standard Model Context Protocol tools, so that ANY
MCP-compatible client (this Telegram bot, Claude Desktop, another agent) can drive
the system through one common interface instead of a bespoke API.

Tools exposed:
    search_news(category)            - live headlines, each screened by the clickbait model
    check_headline_quality(headline) - the fine-tuned DistilBERT classifier
    prepare_story(number)            - write + publish the article webpage
    publish_story(platforms)         - post the prepared story to the channel / Instagram

Run standalone:      python mcp_server.py
Used by the bot via: mcp_client.py (stdio transport)
"""

import contextlib
import os
import sys
from types import SimpleNamespace

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from telegram import Bot

load_dotenv()

# stdout carries the MCP JSON-RPC protocol, so nothing else may write to it.
# Import the pipeline with stdout redirected in case anything prints on import.
with contextlib.redirect_stdout(sys.stderr):
    import bot as B  # reuse the pipeline implementation (no bot is started on import)

mcp = FastMCP("agentic-news")

# Load the classifier now, on the main thread, before the event loop starts. Loading it
# lazily inside a request worker thread proved unreliable under the server's event loop.
with contextlib.redirect_stdout(sys.stderr):
    from ml.classifier import check_headline as _warm_classifier
    _warm_classifier("warm up the model")

# One long-lived session: search_news stores the headlines that prepare_story then uses.
_tg = Bot(os.getenv("TELEGRAM_TOKEN"))
_tg_ready = False
_session = SimpleNamespace(user_data={}, bot=_tg)
_tools = B.build_toolbox(_session)


async def _ensure_telegram():
    global _tg_ready
    if not _tg_ready:
        await _tg.initialize()
        _tg_ready = True


@mcp.tool()
async def search_news(category: str) -> dict:
    """Fetch the latest live news headlines for a category (technology, business, science,
    entertainment, sports, general). Every headline is screened by the fine-tuned clickbait
    classifier and returned with a quality verdict, a confidence score and a number."""
    return await _tools["search_news"](category=category)


@mcp.tool()
async def check_headline_quality(headline: str) -> dict:
    """Classify any headline with the fine-tuned DistilBERT clickbait detector.
    Returns the label (genuine/clickbait) and a confidence score."""
    return await _tools["check_headline_quality"](headline=headline)


@mcp.tool()
async def prepare_story(number: int) -> dict:
    """Prepare one of the headlines returned by search_news: write a curated headline, a hook
    and a full article, gather images, and publish the article webpage. Refuses headlines that
    the clickbait classifier has flagged. Must be called before publish_story."""
    return await _tools["prepare_story"](number=number)


@mcp.tool()
async def publish_story(platforms: str) -> dict:
    """Publish the story prepared by prepare_story. platforms must be one of:
    'channel' (Telegram channel), 'instagram', or 'both'."""
    await _ensure_telegram()
    return await _tools["publish_story"](platforms=platforms)


if __name__ == "__main__":
    mcp.run()  # stdio transport
