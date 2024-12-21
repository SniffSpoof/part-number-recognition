import argparse
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

def parse_args():
    parser = argparse.ArgumentParser(description="Telegram Bot to log chat IDs")
    parser.add_argument('--telegram-token', type=str, required=True, help="Your Telegram API token")
    return parser.parse_args()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print(f"Chat ID: {chat_id}")
    await update.message.reply_text(f"Your chat ID is: {chat_id}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    print(f"Chat ID: {chat_id}")
    await update.message.reply_text(f"Your chat ID is: {chat_id}")

async def main():
    args = parse_args()

    application = ApplicationBuilder().token(args.telegram_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot started. Send a message to get the chat ID.")
    await application.initialize()
    print("Bot started. Use it via Telegram.")
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    await asyncio.Future()  # Keep the program running

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

