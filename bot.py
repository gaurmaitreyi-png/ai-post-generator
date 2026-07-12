import os
import io
import sys
import re
import json
import time
import base64
import asyncio
import logging
import tempfile
from datetime import datetime, timezone
import requests
import tweepy
import praw
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest
from google import genai
from dotenv import load_dotenv

from ml.classifier import check_headline  # fine-tuned DistilBERT clickbait detector
from agent import run_agent               # conversational agent (LLM tool-calling loop)
from mcp_client import MCPToolbox         # MCP client: tools are discovered, not hard-coded

# Set at start-up once the MCP server is reachable; None means fall back to in-process tools.
mcp_toolbox: MCPToolbox | None = None

load_dotenv()

# Windows consoles default to cp1252, which crashes on emoji in print()/logging output.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CATEGORIES = ['Tech', 'Business', 'Science', 'Entertainment', 'Sports']
# Map our display names to NewsAPI's exact category slugs (NewsAPI uses "technology", not "tech").
NEWS_CATEGORY_MAP = {'tech': 'technology'}
# Stable fallback images so every post ALWAYS has a valid image (Instagram is strict about this).
DEFAULT_IMAGE_URL = "https://images.pexels.com/photos/60133/pexels-photo-60133.jpeg?auto=compress&cs=tinysrgb&w=1200&h=650"
DEFAULT_IG_SQUARE = "https://images.pexels.com/photos/60133/pexels-photo-60133.jpeg?auto=compress&cs=tinysrgb&fit=crop&w=1080&h=1080"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def get_twitter_client():
    return tweepy.Client(
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )


def get_twitter_v1_api():
    auth = tweepy.OAuth1UserHandler(
        os.getenv("TWITTER_API_KEY"),
        os.getenv("TWITTER_API_SECRET"),
        os.getenv("TWITTER_ACCESS_TOKEN"),
        os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
    )
    return tweepy.API(auth)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat.lower()}")] for cat in CATEGORIES]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "👋 Welcome to Agentic News!\n\n"
        "💬 Just talk to me, e.g.\n"
        "   \"find the latest tech news and post the best one to the channel\"\n"
        "   \"is this headline clickbait: ...\"\n\n"
        "🤖 Every headline is checked by my fine-tuned clickbait model before publishing.\n\n"
        "Or tap a category below:",
        reply_markup=reply_markup,
    )


async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    category = query.data.split('_')[1]
    news_category = NEWS_CATEGORY_MAP.get(category, category)
    await query.edit_message_text(text=f"🔍 Fetching latest live {category} news...")
    key = os.getenv('NEWS_API_KEY')
    try:
        res = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"category": news_category, "country": "us", "pageSize": 5, "apiKey": key},
            timeout=10,
        ).json()
        if res.get("status") == "error":
            logger.error(f"NewsAPI error: {res.get('message')}")
            await query.edit_message_text(f"❌ News service error: {res.get('message', 'please try again later')}")
            return
        articles = res.get('articles', [])
        if not articles:
            # Fallback: broader search via the /everything endpoint.
            res2 = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": news_category, "language": "en", "sortBy": "publishedAt", "pageSize": 5, "apiKey": key},
                timeout=10,
            ).json()
            articles = res2.get('articles', [])
        if not articles:
            await query.edit_message_text(f"❌ No news found for {category} right now. Try another category.")
            return
        context.user_data['articles'] = articles

        # Run every headline past the fine-tuned clickbait model (the editorial quality gate).
        news_list = f"📰 Top {category.capitalize()} News\n\n"
        flagged = 0
        for idx, art in enumerate(articles):
            title = art.get('title', 'No Title')
            verdict = await asyncio.to_thread(check_headline, title)
            if verdict['is_clickbait']:
                flagged += 1
                news_list += f"{idx+1}. ⚠️ {title}\n   (clickbait — {verdict['confidence']*100:.0f}% — blocked)\n\n"
            else:
                news_list += f"{idx+1}. ✅ {title}\n\n"

        news_list += "👉 Type the number (1-5) of the news you want to turn into a post!"
        if flagged:
            news_list += f"\n\n🤖 My clickbait model flagged {flagged} of these — they can't be published."
        await query.edit_message_text(text=news_list)
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        await query.edit_message_text("❌ Error fetching news.")


def download_image(url: str) -> io.BytesIO | None:
    try:
        res = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if res.status_code == 200 and "image" in res.headers.get("Content-Type", ""):
            buf = io.BytesIO(res.content)
            buf.name = "image.jpg"
            buf.seek(0)
            return buf
    except Exception as e:
        logger.error(f"Image download error: {e}")
    return None


def get_article_image(article: dict) -> io.BytesIO | None:
    img_url = article.get("urlToImage")
    if img_url and img_url.startswith("http"):
        buf = download_image(img_url)
        if buf:
            return buf
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if unsplash_key:
        try:
            res = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": article.get("title", "news"), "per_page": 1, "orientation": "landscape", "client_id": unsplash_key},
                timeout=10,
            ).json()
            results = res.get("results", [])
            if results:
                buf = download_image(results[0]["urls"]["regular"])
                if buf:
                    return buf
        except Exception as e:
            logger.error(f"Unsplash fallback failed: {e}")
    return None


def post_to_twitter(text: str, image_bytes: io.BytesIO | None) -> str | None:
    try:
        twitter_v2 = get_twitter_client()
        media_id = None
        if image_bytes:
            try:
                twitter_v1 = get_twitter_v1_api()
                image_bytes.seek(0)
                media = twitter_v1.media_upload(filename="news.jpg", file=image_bytes)
                media_id = media.media_id
            except Exception as media_err:
                logger.error(f"Media upload failed, posting without image: {media_err}")
        tweet_text = text[:277] + "..." if len(text) > 280 else text
        if media_id:
            response = twitter_v2.create_tweet(text=tweet_text, media_ids=[media_id])
        else:
            response = twitter_v2.create_tweet(text=tweet_text)
        tweet_id = response.data["id"]
        me = twitter_v2.get_me()
        username = me.data.username
        return f"https://twitter.com/{username}/status/{tweet_id}"
    except Exception as e:
        logger.error(f"Twitter post failed: {e}")
        return None


def get_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "AgenticNewsBot/1.0"),
    )


def post_to_reddit(title: str, text: str, image_bytes: io.BytesIO | None) -> str | None:
    try:
        reddit = get_reddit_client()
        username = os.getenv("REDDIT_USERNAME")
        # Posting to your own profile uses the special "u_<username>" subreddit.
        profile = reddit.subreddit(f"u_{username}")
        reddit_title = title[:297] + "..." if len(title) > 300 else title

        submission = None
        if image_bytes:
            tmp_path = None
            try:
                image_bytes.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_bytes.read())
                    tmp_path = tmp.name
                submission = profile.submit_image(title=reddit_title, image_path=tmp_path)
            except Exception as img_err:
                logger.error(f"Reddit image submit failed, falling back to text post: {img_err}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        if submission is None:
            # No image (or image upload failed): post the AI text as a self post.
            submission = profile.submit(title=reddit_title, selftext=text)
        elif text:
            # Image post carries only a title, so add the AI text as the first comment.
            try:
                submission.reply(text)
            except Exception as comment_err:
                logger.error(f"Reddit comment failed: {comment_err}")

        return f"https://www.reddit.com{submission.permalink}"
    except Exception as e:
        logger.error(f"Reddit post failed: {e}")
        return None


def post_to_instagram(caption: str, image_url: str | None) -> str | None:
    ig_user_id = os.getenv("IG_USER_ID")
    access_token = os.getenv("IG_ACCESS_TOKEN")
    if not ig_user_id or not access_token:
        logger.error("Instagram IG_USER_ID / IG_ACCESS_TOKEN not set.")
        return None
    if not image_url or not image_url.startswith("http"):
        logger.error("Instagram requires a public image URL; none available for this article.")
        return None
    base = "https://graph.facebook.com/v21.0"
    try:
        # 1. Create a media container (Instagram fetches the image from image_url).
        create = requests.post(
            f"{base}/{ig_user_id}/media",
            data={"image_url": image_url, "caption": caption[:2200], "access_token": access_token},
            timeout=30,
        ).json()
        creation_id = create.get("id")
        if not creation_id:
            logger.error(f"Instagram container creation failed: {create}")
            return None

        # 2. Wait until the container finishes processing before publishing.
        for _ in range(10):
            status = requests.get(
                f"{base}/{creation_id}",
                params={"fields": "status_code", "access_token": access_token},
                timeout=30,
            ).json()
            if status.get("status_code") == "FINISHED":
                break
            if status.get("status_code") == "ERROR":
                logger.error(f"Instagram container processing error: {status}")
                return None
            time.sleep(2)

        # 3. Publish the container.
        publish = requests.post(
            f"{base}/{ig_user_id}/media_publish",
            data={"creation_id": creation_id, "access_token": access_token},
            timeout=30,
        ).json()
        media_id = publish.get("id")
        if not media_id:
            logger.error(f"Instagram publish failed: {publish}")
            return None

        perma = requests.get(
            f"{base}/{media_id}",
            params={"fields": "permalink", "access_token": access_token},
            timeout=30,
        ).json()
        return perma.get("permalink", f"https://www.instagram.com (id {media_id})")
    except Exception as e:
        logger.error(f"Instagram post failed: {e}")
        return None


def post_to_facebook(caption: str, image_url: str | None) -> str | None:
    page_id = os.getenv("FB_PAGE_ID")
    token = os.getenv("FB_PAGE_ACCESS_TOKEN")
    if not page_id or not token:
        logger.error("Facebook FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN not set.")
        return None
    base = "https://graph.facebook.com/v21.0"
    try:
        if image_url and image_url.startswith("http"):
            # Photo post (Facebook accepts any aspect ratio and makes links in the caption clickable).
            res = requests.post(
                f"{base}/{page_id}/photos",
                data={"url": image_url, "caption": caption[:2000], "access_token": token},
                timeout=60,
            ).json()
            post_id = res.get("post_id") or res.get("id")
        else:
            res = requests.post(
                f"{base}/{page_id}/feed",
                data={"message": caption[:2000], "access_token": token},
                timeout=60,
            ).json()
            post_id = res.get("id")
        if post_id:
            return f"https://www.facebook.com/{post_id}"
        logger.error(f"Facebook post failed: {res}")
        return None
    except Exception as e:
        logger.error(f"Facebook post failed: {e}")
        return None


def post_to_linkedin(text: str, image_bytes: io.BytesIO | None) -> str | None:
    token = os.getenv("LINKEDIN_ACCESS_TOKEN")
    person_urn = os.getenv("LINKEDIN_PERSON_URN")  # e.g. urn:li:person:xxxx
    if not token or not person_urn:
        logger.error("LinkedIn LINKEDIN_ACCESS_TOKEN / LINKEDIN_PERSON_URN not set.")
        return None
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }
    try:
        asset_urn = None
        if image_bytes:
            # 1. Register an image upload to get an upload URL + asset URN.
            reg = requests.post(
                "https://api.linkedin.com/v2/assets?action=registerUpload",
                headers=headers,
                json={
                    "registerUploadRequest": {
                        "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
                        "owner": person_urn,
                        "serviceRelationships": [
                            {"relationshipType": "OWNER", "identifier": "urn:li:userGeneratedContent"}
                        ],
                    }
                },
                timeout=30,
            ).json()
            try:
                upload_url = reg["value"]["uploadMechanism"][
                    "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
                ]["uploadUrl"]
                asset_urn = reg["value"]["asset"]
                # 2. Upload the raw image bytes to that URL.
                image_bytes.seek(0)
                up = requests.put(
                    upload_url,
                    headers={"Authorization": f"Bearer {token}"},
                    data=image_bytes.read(),
                    timeout=60,
                )
                if up.status_code not in (200, 201):
                    logger.error(f"LinkedIn image upload failed: {up.status_code} {up.text}")
                    asset_urn = None
            except (KeyError, TypeError) as parse_err:
                logger.error(f"LinkedIn registerUpload parse failed: {reg} ({parse_err})")
                asset_urn = None

        share_content = {
            "shareCommentary": {"text": text[:3000]},
            "shareMediaCategory": "IMAGE" if asset_urn else "NONE",
        }
        if asset_urn:
            share_content["media"] = [{"status": "READY", "media": asset_urn}]

        body = {
            "author": person_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {"com.linkedin.ugc.ShareContent": share_content},
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }
        res = requests.post("https://api.linkedin.com/v2/ugcPosts", headers=headers, json=body, timeout=30)
        if res.status_code in (200, 201):
            post_id = res.headers.get("x-restli-id") or res.json().get("id")
            return f"https://www.linkedin.com/feed/update/{post_id}"
        logger.error(f"LinkedIn post failed: {res.status_code} {res.text}")
        return None
    except Exception as e:
        logger.error(f"LinkedIn post failed: {e}")
        return None


async def post_to_telegram_channel(bot, text: str, image_bytes: io.BytesIO | None, reply_markup=None) -> str | None:
    channel_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if not channel_id:
        logger.error("TELEGRAM_CHANNEL_ID not set in .env.")
        return None
    try:
        if image_bytes:
            image_bytes.seek(0)
            # Telegram photo captions max out at 1024 chars.
            msg = await bot.send_photo(chat_id=channel_id, photo=image_bytes, caption=text[:1024],
                                       reply_markup=reply_markup, write_timeout=60)
        else:
            msg = await bot.send_message(chat_id=channel_id, text=text[:4096], reply_markup=reply_markup)
        username = getattr(msg.chat, "username", None)
        if username:
            return f"https://t.me/{username}/{msg.message_id}"
        internal = str(msg.chat.id).replace("-100", "", 1)
        return f"https://t.me/c/{internal}/{msg.message_id}"
    except Exception as e:
        logger.error(f"Telegram channel post failed: {e}")
        return None


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "news").lower()).strip("-")
    return (slug[:60] or "news")


def generate_channel_article(title: str, desc: str, source: str) -> dict | None:
    """Ask Gemini for channel-specific headline, hook caption, and a full article body."""
    prompt = (
        "You are the editor of a news channel called 'Agentic News'. "
        "Using ONLY the information in the headline and description below, produce JSON with keys "
        "'headline', 'caption', and 'article'.\n"
        "- headline: an eye-catching but accurate, informative headline (max ~90 characters). No clickbait falsehoods.\n"
        "- caption: 1-2 punchy sentences to hook a reader on Telegram, followed by 2-3 relevant hashtags.\n"
        "- article: a neutral, well-structured news article of 4-6 paragraphs. Do NOT invent specific "
        "quotes, numbers, names, or events that are not implied by the given information.\n"
        "Return ONLY raw JSON, no markdown fences.\n\n"
        f"Headline: {title}\nDescription: {desc}\nSource: {source}"
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
            raw = (resp.text or "").strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(raw)
            if data.get("headline") and data.get("article"):
                return data
        except Exception as e:
            logger.error(f"Article generation attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None


def get_extra_image_urls(query: str, n: int = 2) -> list[str]:
    """Fetch a few extra related photos from Pexels for the article body."""
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    try:
        res = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": key},
            params={"query": query, "per_page": n, "orientation": "landscape"},
            timeout=15,
        ).json()
        return [p["src"]["large"] for p in res.get("photos", []) if p.get("src", {}).get("large")]
    except Exception as e:
        logger.error(f"Pexels fetch failed: {e}")
        return []


def to_instagram_square(url: str) -> str | None:
    """Turn a Pexels image URL into a 1080x1080 square crop — always a valid Instagram aspect ratio."""
    if url and "images.pexels.com" in url:
        return url.split("?")[0] + "?auto=compress&cs=tinysrgb&fit=crop&w=1080&h=1080"
    return None


def get_instagram_image_url(image_urls: list[str], query: str) -> str:
    """Always return an Instagram-safe image URL (valid aspect ratio), never None."""
    # 1. Square-crop a Pexels image we already gathered for this story.
    for u in image_urls:
        sq = to_instagram_square(u)
        if sq:
            return sq
    # 2. Fetch a fresh Pexels image, then square-crop it.
    for q in (query, "breaking news"):
        for u in get_extra_image_urls(q, 1):
            sq = to_instagram_square(u)
            if sq:
                return sq
    # 3. Guaranteed fallback so Instagram never fails for lack of a valid image.
    return DEFAULT_IG_SQUARE


def build_article_html(headline: str, caption: str, article: str, image_urls: list[str],
                       source: str, source_url: str | None) -> str:
    """Render a clean, news-style standalone HTML page for GitHub Pages."""
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", article) if p.strip()]
    date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # Interleave the extra images between paragraphs.
    body_parts = []
    extra_imgs = image_urls[1:] if len(image_urls) > 1 else []
    for i, para in enumerate(paragraphs):
        body_parts.append(f"<p>{esc(para)}</p>")
        if extra_imgs and i == max(1, len(paragraphs) // 2):
            body_parts.append(f'<figure><img src="{esc(extra_imgs[0])}" alt=""></figure>')
        if len(extra_imgs) > 1 and i == len(paragraphs) - 1:
            body_parts.append(f'<figure><img src="{esc(extra_imgs[1])}" alt=""></figure>')
    body_html = "\n".join(body_parts)

    hero = f'<figure class="hero"><img src="{esc(image_urls[0])}" alt=""></figure>' if image_urls else ""
    source_line = f'<a href="{esc(source_url)}" target="_blank" rel="noopener">{esc(source)}</a>' if source_url else esc(source)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(headline)} — Agentic News</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: Georgia, 'Times New Roman', serif; line-height:1.7;
         color:#1a1a1a; background:#fafafa; }}
  .bar {{ background:#0b3d91; color:#fff; padding:14px 20px; font-family:Arial,Helvetica,sans-serif;
          font-weight:700; letter-spacing:.5px; font-size:20px; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:24px 20px 60px; }}
  h1 {{ font-size:34px; line-height:1.25; margin:18px 0 8px; }}
  .meta {{ font-family:Arial,Helvetica,sans-serif; color:#666; font-size:14px; margin-bottom:20px; }}
  .lead {{ font-size:20px; color:#333; font-style:italic; margin:0 0 24px; }}
  figure {{ margin:24px 0; }}
  figure img {{ width:100%; height:auto; border-radius:8px; display:block; }}
  figure.hero img {{ border-radius:10px; }}
  p {{ font-size:19px; margin:0 0 20px; }}
  .foot {{ font-family:Arial,Helvetica,sans-serif; font-size:13px; color:#888; border-top:1px solid #ddd;
           margin-top:40px; padding-top:16px; }}
  a {{ color:#0b3d91; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#121212; color:#e6e6e6; }}
    .lead {{ color:#c8c8c8; }} .meta,.foot {{ color:#9a9a9a; }} a {{ color:#7aa7ff; }}
  }}
</style>
</head>
<body>
  <div class="bar">📰 AGENTIC NEWS</div>
  <div class="wrap">
    <h1>{esc(headline)}</h1>
    <div class="meta">{date_str} · Source: {source_line}</div>
    <p class="lead">{esc(caption)}</p>
    {hero}
    {body_html}
    <div class="foot">
      This article was auto-generated by Agentic News from public reporting. For the original report, see {source_line}.
    </div>
  </div>
</body>
</html>"""


def publish_article_to_github(slug: str, html: str) -> str | None:
    """Publish the HTML article to the GitHub Pages repo via the Contents API."""
    token = os.getenv("GITHUB_TOKEN")
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_PAGES_REPO")
    if not (token and owner and repo):
        logger.error("GITHUB_TOKEN / GITHUB_OWNER / GITHUB_PAGES_REPO not set in .env.")
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = f"articles/{stamp}-{slug}.html"
    try:
        res = requests.put(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={
                "message": f"Publish article: {slug}",
                "content": base64.b64encode(html.encode("utf-8")).decode("ascii"),
                "branch": "main",
            },
            timeout=30,
        )
        if res.status_code in (200, 201):
            return f"https://{owner}.github.io/{repo}/{path}"
        logger.error(f"GitHub publish failed: {res.status_code} {res.text}")
        return None
    except Exception as e:
        logger.error(f"GitHub publish failed: {e}")
        return None


async def wait_for_pages_live(url: str, timeout: int = 120, interval: int = 6) -> bool:
    """Poll a GitHub Pages URL until it returns 200 (Pages rebuilds take ~1-2 min)."""
    elapsed = 0
    while elapsed < timeout:
        try:
            if requests.head(url, timeout=10, allow_redirects=True).status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)
        elapsed += interval
    return False


def build_toolbox(context: ContextTypes.DEFAULT_TYPE) -> dict:
    """The tools the LLM agent is allowed to call. Each returns a JSON-serialisable dict."""

    async def search_news(category: str) -> dict:
        news_category = NEWS_CATEGORY_MAP.get(category, category)
        key = os.getenv('NEWS_API_KEY')
        res = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"category": news_category, "country": "us", "pageSize": 5, "apiKey": key},
            timeout=10,
        ).json()
        articles = res.get('articles', [])
        if not articles:
            res2 = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": news_category, "language": "en", "sortBy": "publishedAt",
                        "pageSize": 5, "apiKey": key},
                timeout=10,
            ).json()
            articles = res2.get('articles', [])
        if not articles:
            return {"error": f"no news found for {category}"}

        context.user_data['articles'] = articles
        listed = []
        for i, art in enumerate(articles, 1):
            title = art.get('title', '')
            verdict = await asyncio.to_thread(check_headline, title)
            listed.append({
                "number": i,
                "headline": title,
                "quality": verdict['label'],
                "confidence": round(verdict['confidence'], 3),
                "publishable": not verdict['is_clickbait'],
            })
        return {"headlines": listed}

    async def check_headline_quality(headline: str) -> dict:
        verdict = await asyncio.to_thread(check_headline, headline)
        return {"label": verdict['label'],
                "confidence": round(verdict['confidence'], 3),
                "is_clickbait": verdict['is_clickbait']}

    async def prepare_story(number: int) -> dict:
        articles = context.user_data.get('articles', [])
        idx = int(number) - 1
        if not articles or idx < 0 or idx >= len(articles):
            return {"error": "invalid number - call search_news first"}

        art = articles[idx]
        title = art.get('title', '')
        verdict = await asyncio.to_thread(check_headline, title)
        if verdict['is_clickbait']:
            return {"blocked": True, "headline": title,
                    "reason": "the clickbait classifier flagged this headline",
                    "confidence": round(verdict['confidence'], 3)}

        context.user_data['pending_title'] = title
        context.user_data['pending_desc'] = art.get('description', '')
        context.user_data['pending_source'] = art.get('source', {}).get('name', 'News')
        context.user_data['pending_source_url'] = art.get('url')
        context.user_data['pending_image_url'] = art.get('urlToImage')
        context.user_data['pending_article'] = None
        context.user_data['pending_article_url'] = None
        context.user_data['pending_image_urls'] = None

        data = await publish_curated_article(context)
        if not data:
            return {"error": "could not generate or publish the article"}
        return {"prepared": True, "headline": data['headline'], "article_url": data['article_url']}

    async def publish_story(platforms: str) -> dict:
        if not context.user_data.get('pending_title'):
            return {"error": "no story prepared yet - call prepare_story first"}
        data = await publish_curated_article(context)
        if not data:
            return {"error": "the prepared story is no longer available"}

        results = {"article_url": data['article_url']}
        if platforms in ("channel", "both"):
            results["telegram_channel"] = await _post_curated_to_channel(context, data) or "failed"
        if platforms in ("instagram", "both"):
            results["instagram"] = await asyncio.to_thread(_post_curated_to_instagram, data) or "failed"
        return results

    return {
        "search_news": search_news,
        "check_headline_quality": check_headline_quality,
        "prepare_story": prepare_story,
        "publish_story": publish_story,
    }


async def handle_agent_chat(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Free-text message: hand it to the LLM agent, which decides which tools to call.

    Tools are served over MCP when the server is up; otherwise we fall back to calling
    the same functions in-process so the bot keeps working either way.
    """
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    history = context.user_data.get('chat_history', [])

    if mcp_toolbox is not None and mcp_toolbox.session is not None:
        call_tool = mcp_toolbox.call_tool
        declarations = mcp_toolbox.declarations
    else:
        toolbox = build_toolbox(context)

        async def call_tool(name: str, args: dict) -> dict:
            fn = toolbox.get(name)
            if fn is None:
                return {"error": f"unknown tool: {name}"}
            return await fn(**args)

        declarations = None  # agent falls back to its built-in declarations

    try:
        reply, history = await run_agent(client, text, history, call_tool, declarations)
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        await update.message.reply_text("Sorry, something went wrong on my side. Try again.")
        return
    context.user_data['chat_history'] = history
    await update.message.reply_text(reply)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return
    # Anything that isn't a plain story number is a conversation with the agent.
    if not text.isdigit():
        await handle_agent_chat(update, context, text)
        return
    idx = int(text) - 1
    articles = context.user_data.get('articles', [])
    if not articles or idx < 0 or idx >= len(articles):
        await update.message.reply_text("❌ Invalid number. Please pick 1 to 5.")
        return

    article = articles[idx]
    title = article.get('title', 'Latest News')
    desc = article.get('description', '')
    source = article.get('source', {}).get('name', 'News')

    # --- Editorial quality gate: the fine-tuned DistilBERT classifier decides ---
    verdict = await asyncio.to_thread(check_headline, title)
    if verdict['is_clickbait']:
        await update.message.reply_text(
            f"🚫 Blocked by the quality model.\n\n"
            f"\"{title}\"\n\n"
            f"Classified as CLICKBAIT ({verdict['confidence']*100:.1f}% confidence).\n"
            f"Agentic News doesn't publish clickbait — please pick another story."
        )
        return

    await update.message.reply_text(
        f"✅ Quality check passed (genuine, {verdict['confidence']*100:.1f}% confidence).\n"
        f"🤖 Gemini is crafting your post..."
    )

    # --- Gemini: generate post (with retry) ---
    post_prompt = (
        f"Write ONE short news post of 4-5 lines as a single paragraph that clearly explains this story "
        f"in simple, engaging words. Output ONLY the post text — no preamble like 'Here are the posts', "
        f"no platform labels (LinkedIn/Twitter), no headings, no bullet points, no separators or asterisks. "
        f"End with 2-3 relevant hashtags.\n\nHeadline: {title}\nDescription: {desc}"
    )
    ai_text = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model='gemini-2.5-flash', contents=post_prompt)
            ai_text = response.text
            break
        except Exception as e:
            logger.error(f"Gemini attempt {attempt+1} failed: {e}")
            await asyncio.sleep(3)

    if not ai_text:
        await update.message.reply_text("❌ Gemini is overloaded right now. Please try again in 30 seconds.")
        return

    # --- Get image ---
    image_bytes = get_article_image(article)

    # --- Send image to Telegram ---
    if image_bytes:
        try:
            image_bytes.seek(0)
            await update.message.reply_photo(photo=image_bytes, caption=f"📸 via {source}", write_timeout=60)
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")

    # --- Show generated post ---
    await update.message.reply_text(f"📝 Generated Post:\n\n{ai_text}")

    # --- Save to user_data for the confirm step ---
    context.user_data['pending_post'] = ai_text
    context.user_data['pending_title'] = title
    context.user_data['pending_desc'] = desc
    context.user_data['pending_source'] = source
    context.user_data['pending_source_url'] = article.get("url")
    context.user_data['pending_image_url'] = article.get("urlToImage")
    # Reset per-story caches so a new pick doesn't reuse the previous article/page.
    context.user_data['pending_article'] = None
    context.user_data['pending_article_url'] = None
    context.user_data['pending_image_urls'] = None

    # --- Build the full article page now, so "Read full article" works + posting is instant later ---
    prep_msg = await update.message.reply_text("🌐 Building the full article page...")
    article_data = await publish_curated_article(context, wait=False)
    read_more_row = []
    if article_data:
        read_more_row = [[InlineKeyboardButton("📖 Read full article", url=article_data["article_url"])]]
        try:
            await prep_msg.delete()
        except Exception:
            pass
    else:
        try:
            await prep_msg.edit_text("⚠️ Couldn't build the article page now — you can still post below.")
        except Exception:
            pass

    # --- Ask for confirmation ---
    keyboard = read_more_row + [
        [InlineKeyboardButton("🚀 Post to All (Channel, Instagram, Facebook)", callback_data="both_yes")],
        [InlineKeyboardButton("📢 Post to Agentic News channel", callback_data="channel_yes")],
        [
            InlineKeyboardButton("📸 Post to Instagram", callback_data="instagram_yes"),
            InlineKeyboardButton("📘 Post to Facebook", callback_data="facebook_yes"),
        ],
        [InlineKeyboardButton("❌ Skip", callback_data="skip")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📢 Where should I post this?",
        reply_markup=reply_markup
    )


def get_curated_article(context: ContextTypes.DEFAULT_TYPE) -> dict | None:
    """Generate the curated headline/hook/article once, then cache it for reuse across platforms."""
    art = context.user_data.get('pending_article')
    if art:
        return art
    art = generate_channel_article(
        context.user_data.get('pending_title', 'Latest News'),
        context.user_data.get('pending_desc', ''),
        context.user_data.get('pending_source', 'News'),
    )
    if art:
        context.user_data['pending_article'] = art
    return art


def update_index_page(headline: str, article_url: str) -> None:
    """Maintain a homepage (index.html) listing recent articles — used as the Instagram bio link."""
    token = os.getenv("GITHUB_TOKEN"); owner = os.getenv("GITHUB_OWNER"); repo = os.getenv("GITHUB_PAGES_REPO")
    if not (token and owner and repo):
        return
    def esc(s): return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/index.html"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        sha = None; items = ""
        r = requests.get(api, headers=headers, timeout=30)
        if r.status_code == 200:
            j = r.json(); sha = j["sha"]
            existing = base64.b64decode(j["content"]).decode("utf-8")
            m = re.search(r"<!--LIST-->(.*?)<!--/LIST-->", existing, re.S)
            items = m.group(1) if m else ""
        date = datetime.now(timezone.utc).strftime("%b %d, %Y")
        items = f'<li><a href="{esc(article_url)}">{esc(headline)}</a><span> · {date}</span></li>\n' + items
        html = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Agentic News</title><style>"
            "body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#fafafa;color:#1a1a1a}"
            ".bar{background:#0b3d91;color:#fff;padding:16px 20px;font-weight:700;font-size:22px}"
            ".wrap{max-width:720px;margin:0 auto;padding:24px 20px 60px}"
            "ul{list-style:none;padding:0}li{padding:16px 0;border-bottom:1px solid #ddd;font-size:18px}"
            "a{color:#0b3d91;text-decoration:none;font-weight:600}span{color:#888;font-size:13px}"
            "@media(prefers-color-scheme:dark){body{background:#121212;color:#e6e6e6}a{color:#7aa7ff}}"
            "</style></head><body><div class=\"bar\">📰 AGENTIC NEWS</div><div class=\"wrap\">"
            "<h2>Latest stories</h2><ul><!--LIST-->" + items + "<!--/LIST--></ul></div></body></html>"
        )
        body = {"message": "Update index", "content": base64.b64encode(html.encode("utf-8")).decode("ascii"), "branch": "main"}
        if sha:
            body["sha"] = sha
        requests.put(api, headers=headers, json=body, timeout=30)
    except Exception as e:
        logger.error(f"Index update failed: {e}")


async def publish_curated_article(context: ContextTypes.DEFAULT_TYPE, wait: bool = True) -> dict | None:
    """Curate the story, publish its article page once (cached), update the index, return details.

    Returns {headline, caption, article_url, image_urls} or None on failure. No UI side effects."""
    art = get_curated_article(context)
    if not art:
        return None
    headline = art["headline"].strip()
    caption = art.get("caption", "").strip()

    # Already published this story's page (e.g. second platform, or done at preview time)? Reuse it.
    cached_url = context.user_data.get('pending_article_url')
    if cached_url:
        if wait:
            await wait_for_pages_live(cached_url)
        return {"headline": headline, "caption": caption, "article_url": cached_url,
                "image_urls": context.user_data.get('pending_image_urls') or []}

    # Gather images: the article's own photo + a couple related ones from Pexels.
    image_urls = []
    main_image_url = context.user_data.get('pending_image_url')
    if main_image_url and main_image_url.startswith("http"):
        image_urls.append(main_image_url)
    image_urls += get_extra_image_urls(headline, n=2)
    if not image_urls:  # never leave a story without an image
        image_urls = get_extra_image_urls("breaking news", n=1) or [DEFAULT_IMAGE_URL]
    context.user_data['pending_image_urls'] = image_urls

    html = build_article_html(headline, caption, art["article"].strip(), image_urls,
                              context.user_data.get('pending_source', 'News'),
                              context.user_data.get('pending_source_url'))
    article_url = publish_article_to_github(slugify(headline), html)
    if not article_url:
        return None
    context.user_data['pending_article_url'] = article_url
    update_index_page(headline, article_url)  # keep the bio-link homepage current
    if wait:
        await wait_for_pages_live(article_url)
    return {"headline": headline, "caption": caption, "article_url": article_url, "image_urls": image_urls}


async def _post_curated_to_channel(context, data) -> str | None:
    image_bytes = download_image(data["image_urls"][0]) if data["image_urls"] else None
    cap = f"📰 {data['headline']}\n\n{data['caption']}" if data["caption"] else f"📰 {data['headline']}"
    read_more = InlineKeyboardMarkup([[InlineKeyboardButton("📖 Read the full story", url=data["article_url"])]])
    return await post_to_telegram_channel(context.bot, cap, image_bytes, reply_markup=read_more)


def _post_curated_to_instagram(data) -> str | None:
    # Always use an Instagram-safe (valid aspect ratio) image so it never gets rejected.
    img = get_instagram_image_url(data["image_urls"], data["headline"])
    # Instagram captions can't have tappable links, so we point readers to the bio link (the index page).
    caption = f"📰 {data['headline']}\n\n{data['caption']}\n\n📖 Read the full story — link in bio 🔗"
    return post_to_instagram(caption, img)


def _post_curated_to_facebook(data) -> str | None:
    img = data["image_urls"][0] if data["image_urls"] else None
    # Facebook makes links in the caption clickable, so include the article link directly.
    caption = f"📰 {data['headline']}\n\n{data['caption']}\n\n📖 Read the full story: {data['article_url']}"
    return post_to_facebook(caption, img)


async def handle_channel_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("📢 Preparing and posting to Agentic News...")
    data = await publish_curated_article(context)
    if not data:
        await query.edit_message_text("❌ Couldn't prepare the article (Gemini or GitHub issue). Try again.")
        return
    post_url = await _post_curated_to_channel(context, data)
    if post_url:
        await query.edit_message_text(f"✅ Posted to Agentic News!\n🔗 {post_url}\n📄 Article: {data['article_url']}")
    else:
        await query.edit_message_text(
            "❌ Channel post failed. The article page was published at:\n" + data["article_url"] +
            "\nMake sure the bot is an admin of the channel with 'Post Messages' enabled."
        )


async def handle_both_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("📝 Preparing the article...")
    data = await publish_curated_article(context)
    if not data:
        await query.edit_message_text("❌ Couldn't prepare the article (Gemini or GitHub issue). Try again.")
        return

    await query.edit_message_text("📢 Posting to your Telegram channel...")
    channel_ok = await _post_curated_to_channel(context, data)

    await query.edit_message_text("📸 Posting to Instagram...")
    insta_ok = _post_curated_to_instagram(data)

    await query.edit_message_text("📘 Posting to Facebook...")
    fb_ok = _post_curated_to_facebook(data)

    lines = [
        ("✅" if channel_ok else "❌") + " Telegram channel",
        ("✅" if insta_ok else "❌") + " Instagram",
        ("✅" if fb_ok else "❌") + " Facebook",
        f"\n📄 Article: {data['article_url']}",
    ]
    await query.edit_message_text("Done posting:\n" + "\n".join(lines))


async def handle_facebook_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("📝 Preparing the Facebook post...")
    data = await publish_curated_article(context)
    if not data:
        await query.edit_message_text("❌ Couldn't prepare the article (Gemini or GitHub issue). Try again.")
        return
    await query.edit_message_text("⏳ Posting to Facebook...")
    post_url = _post_curated_to_facebook(data)
    if post_url:
        await query.edit_message_text(f"✅ Posted to Facebook!\n🔗 {post_url}\n📄 Article: {data['article_url']}")
    else:
        await query.edit_message_text(
            "❌ Facebook post failed. Check FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN in .env and that the "
            "Page token is valid."
        )


async def handle_twitter_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "twitter_no":
        await query.edit_message_text("❌ Skipped. Post was NOT sent to Twitter.")
        return

    await query.edit_message_text("⏳ Posting to Twitter...")

    ai_text = context.user_data.get('pending_post')
    image_url = context.user_data.get('pending_image_url')

    # Re-download image for Twitter
    image_bytes = None
    if image_url:
        image_bytes = download_image(image_url)

    tweet_url = post_to_twitter(ai_text, image_bytes)

    if tweet_url:
        await query.edit_message_text(f"✅ Posted to Twitter!\n🔗 {tweet_url}")
    else:
        await query.edit_message_text(
            "❌ Twitter post failed. Make sure your API keys have Read+Write permission.\n"
            "Go to: https://developer.twitter.com → your app → Settings → User authentication settings"
        )


async def handle_reddit_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("⏳ Posting to Reddit...")

    ai_text = context.user_data.get('pending_post')
    title = context.user_data.get('pending_title', 'Latest News')
    image_url = context.user_data.get('pending_image_url')

    # Re-download image for Reddit
    image_bytes = None
    if image_url:
        image_bytes = download_image(image_url)

    post_url = post_to_reddit(title, ai_text, image_bytes)

    if post_url:
        await query.edit_message_text(f"✅ Posted to Reddit!\n🔗 {post_url}")
    else:
        await query.edit_message_text(
            "❌ Reddit post failed. Check that REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, "
            "REDDIT_USERNAME and REDDIT_PASSWORD are set in your .env, and that the app "
            "type on https://www.reddit.com/prefs/apps is 'script'."
        )


async def handle_instagram_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("📝 Preparing the Instagram post...")
    data = await publish_curated_article(context)
    if not data:
        await query.edit_message_text("❌ Couldn't prepare the article (Gemini or GitHub issue). Try again.")
        return

    await query.edit_message_text("⏳ Posting to Instagram...")
    post_url = _post_curated_to_instagram(data)

    if post_url:
        await query.edit_message_text(
            f"✅ Posted to Instagram!\n🔗 {post_url}\n📄 Article: {data['article_url']}"
        )
    else:
        await query.edit_message_text(
            "❌ Instagram post failed. Check that IG_USER_ID and IG_ACCESS_TOKEN are set in .env, "
            "your account is a Business account linked to a Facebook Page, and the token is valid."
        )


async def handle_linkedin_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    await query.edit_message_text("⏳ Posting to LinkedIn...")

    ai_text = context.user_data.get('pending_post')
    image_url = context.user_data.get('pending_image_url')

    # Re-download image for LinkedIn (it needs the raw bytes, not a URL).
    image_bytes = None
    if image_url:
        image_bytes = download_image(image_url)

    post_url = post_to_linkedin(ai_text, image_bytes)

    if post_url:
        await query.edit_message_text(f"✅ Posted to LinkedIn!\n🔗 {post_url}")
    else:
        await query.edit_message_text(
            "❌ LinkedIn post failed. Check that LINKEDIN_ACCESS_TOKEN and LINKEDIN_PERSON_URN are set in "
            ".env, the token has the w_member_social scope, and it hasn't expired."
        )


async def handle_skip(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("❌ Skipped — nothing was posted.")


def build_app() -> Application:
    token = os.getenv("TELEGRAM_TOKEN")
    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")

    request_kwargs = dict(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0,
        http_version="1.1",
    )
    if proxy_url:
        logger.info(f"Using proxy: {proxy_url}")
        request_kwargs["proxy_url"] = proxy_url

    custom_request = HTTPXRequest(**request_kwargs)

    app = (
        Application.builder()
        .token(token)
        .request(custom_request)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_category, pattern="^cat_"))
    app.add_handler(CallbackQueryHandler(handle_both_confirm, pattern="^both_"))
    app.add_handler(CallbackQueryHandler(handle_channel_confirm, pattern="^channel_"))
    app.add_handler(CallbackQueryHandler(handle_instagram_confirm, pattern="^instagram_"))
    app.add_handler(CallbackQueryHandler(handle_facebook_confirm, pattern="^facebook_"))
    app.add_handler(CallbackQueryHandler(handle_skip, pattern="^skip$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return app


async def run_with_retry(max_attempts: int = 10, base_delay: float = 5.0):
    attempt = 0
    delay = base_delay
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"Boot attempt {attempt}/{max_attempts} ...")
        app = build_app()
        try:
            await app.initialize()
            logger.info("✅ Connected to Telegram successfully.")

            # Bring up the MCP server and discover its tools. If it fails, the agent
            # still works using the same functions called in-process.
            global mcp_toolbox
            try:
                mcp_toolbox = await MCPToolbox("mcp_server.py").connect()
                logger.info("✅ MCP server connected; agent tools discovered over MCP.")
            except Exception as mcp_err:
                mcp_toolbox = None
                logger.error(f"⚠️ MCP server unavailable ({mcp_err}); using in-process tools.")

            await app.start()
            await app.updater.start_polling(timeout=60, drop_pending_updates=True)
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown requested.")
            break
        except Exception as exc:
            logger.error(f"Attempt {attempt} failed: {exc}")
            if attempt >= max_attempts:
                logger.critical("Max retry attempts reached. Giving up.")
                raise
            logger.info(f"Retrying in {delay:.0f}s ...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 120.0)
        finally:
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                pass


def main():
    print("🚀 Bot starting… Press Ctrl+C to stop.")
    asyncio.run(run_with_retry(max_attempts=10, base_delay=5.0))


if __name__ == '__main__':
    main()