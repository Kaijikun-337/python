import asyncio
import logging
from aiogram import Bot, Dispatcher, F
from config import TOKEN
from app.user import user as user_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    """
    Main function to initialize and run the bot.
    """
    logging.info("Starting bot...")
    bot = Bot(token=TOKEN)
    dp = Dispatcher()

    # Register the user router with all its handlers
    dp.include_router(user_router)

    # Start the bot and skip any updates that occurred while the bot was offline
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
