import os
import io
import time
import asyncio
import logging
import tempfile
import requests
import tweepy
import praw
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest
from google import genai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
CATEGORIES = ['Tech', 'Business', 'Science', 'Entertainment', 'Sports']

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
    await update.message.reply_text("👋 Welcome! Choose a category to see live breaking headlines:", reply_markup=reply_markup)


async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    category = query.data.split('_')[1]
    await query.edit_message_text(text=f"🔍 Fetching latest live {category} news...")
    news_url = f"https://newsapi.org/v2/top-headlines?category={category}&language=en&pageSize=5&apiKey={os.getenv('NEWS_API_KEY')}"
    try:
        res = requests.get(news_url, timeout=10).json()
        articles = res.get('articles', [])
        if not articles:
            await query.edit_message_text(f"❌ No news found for {category}.")
            return
        context.user_data['articles'] = articles
        news_list = f"📰 Top {category.capitalize()} News\n\n"
        for idx, art in enumerate(articles):
            news_list += f"{idx+1}. {art.get('title', 'No Title')}\n\n"
        news_list += "👉 Type the number (1-5) of the news you want to turn into a post!"
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


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text.isdigit():
        return
    idx = int(text) - 1
    articles = context.user_data.get('articles', [])
    if not articles or idx < 0 or idx >= len(articles):
        await update.message.reply_text("❌ Invalid number. Please pick 1 to 5.")
        return

    article = articles[idx]
    await update.message.reply_text("🤖 Gemini is crafting your post...")
    title = article.get('title', 'Latest News')
    desc = article.get('description', '')
    source = article.get('source', {}).get('name', 'News')

    # --- Gemini: generate post (with retry) ---
    post_prompt = (
        f"Write an engaging social media post for LinkedIn and Twitter based on this "
        f"headline: {title}. Description: {desc}. "
        f"Include relevant hashtags and use simple formatting. Keep it under 270 characters for Twitter."
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
    context.user_data['pending_image_url'] = article.get("urlToImage")

    # --- Ask for confirmation ---
    keyboard = [
        [
            InlineKeyboardButton("🐦 Post to Twitter", callback_data="twitter_yes"),
            InlineKeyboardButton("🤖 Post to Reddit", callback_data="reddit_yes"),
        ],
        [
            InlineKeyboardButton("📸 Post to Instagram", callback_data="instagram_yes"),
            InlineKeyboardButton("💼 Post to LinkedIn", callback_data="linkedin_yes"),
        ],
        [InlineKeyboardButton("❌ Skip", callback_data="twitter_no")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "📢 Where should I post this?",
        reply_markup=reply_markup
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

    await query.edit_message_text("⏳ Posting to Instagram...")

    ai_text = context.user_data.get('pending_post')
    image_url = context.user_data.get('pending_image_url')

    if not image_url:
        await query.edit_message_text("❌ Instagram needs an image, but this article has none. Skipped.")
        return

    post_url = post_to_instagram(ai_text, image_url)

    if post_url:
        await query.edit_message_text(f"✅ Posted to Instagram!\n🔗 {post_url}")
    else:
        await query.edit_message_text(
            "❌ Instagram post failed. Check that IG_USER_ID and IG_ACCESS_TOKEN are set in .env, "
            "your account is a Business/Creator account linked to a Facebook Page, and the token is valid."
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
    app.add_handler(CallbackQueryHandler(handle_twitter_confirm, pattern="^twitter_"))
    app.add_handler(CallbackQueryHandler(handle_reddit_confirm, pattern="^reddit_"))
    app.add_handler(CallbackQueryHandler(handle_instagram_confirm, pattern="^instagram_"))
    app.add_handler(CallbackQueryHandler(handle_linkedin_confirm, pattern="^linkedin_"))
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