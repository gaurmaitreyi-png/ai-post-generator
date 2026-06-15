import os
import logging
import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
CATEGORIES = ['Tech', 'Business', 'Science', 'Entertainment', 'Sports']

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(cat, callback_data=f"cat_{cat.lower()}")] for cat in CATEGORIES]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("👋 Welcome to AI Post Generator! Choose a category to fetch top news:", reply_markup=reply_markup)

async def handle_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    category = query.data.split('_')[1]
    
    await query.edit_message_text(text=f"🔍 Fetching top {category} news...")
    
    # Fetch News
    news_url = f"https://newsapi.org/v2/top-headlines?category={category}&language=en&pageSize=5&apiKey={os.getenv('NEWS_API_KEY')}"
    try:
        res = requests.get(news_url).json()
        articles = res.get('articles', [])
        if not articles:
            await query.edit_message_text(f"❌ No news found for {category} right now.")
            return
        
        context.user_data['articles'] = articles
        keyboard = []
        for idx, art in enumerate(articles):
            keyboard.append([InlineKeyboardButton(f"{idx+1}. {art['title'][:40]}...", callback_data=f"art_{idx}")])
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=f"📰 Top {category} News. Select one to generate a post:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(e)
        await query.edit_message_text("❌ Error fetching news.")

async def handle_article(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    idx = int(query.data.split('_')[1])
    articles = context.user_data.get('articles', [])
    
    if not articles or idx >= len(articles):
        await query.edit_message_text("❌ Session expired. Please restart using /start")
        return
        
    article = articles[idx]
    await query.edit_message_text("🤖 Gemini is crafting your social media post...")
    
    # GenAI Crafting
    prompt = f"Write an engaging, high-quality social media post for LinkedIn and Twitter based on this article title: {article['title']} and description: {article['description']}. Include trending relevant hashtags and clean formatting."
    try:
        response = model.generate_content(prompt)
        ai_text = response.text
        
        # Pexels Image Match
        search_query = article['title'].split()[0] if article['title'] else 'news'
        pexels_url = f"https://api.pexels.com/v1/search?query={search_query}&per_page=1"
        headers = {"Authorization": os.getenv("PEXELS_API_KEY")}
        img_res = requests.get(pexels_url, headers=headers).json()
        
        photos = img_res.get('photos', [])
        if photos:
            img_url = photos[0]['src']['large']
            await query.message.reply_photo(photo=img_url, caption=f"📸 Suggested Visual Asset\n\n📝 **Generated Content:**\n\n{ai_text}")
        else:
            await query.message.reply_text(f"📝 **Generated Content:**\n\n{ai_text}")
            
    except Exception as e:
        logger.error(e)
        await query.message.reply_text("❌ Encountered an issue generating your post asset.")

def main():
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("Error: TELEGRAM_TOKEN not found in environment!")
        return
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(handle_category, pattern="^cat_"))
    app.add_handler(CallbackQueryHandler(handle_article, pattern="^art_"))
    print("🚀 Bot is running... Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == '__main__':
    main()
