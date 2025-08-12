import logging
from aiogram import Router, Bot
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command
from app.generators import handle_generate_image, handle_generate_speech, gemini, handle_analyze_image, VOICES
from aiogram.enums import ChatAction

# All handlers should be registered in the router
user = Router()

# Configure logging to a file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

@user.message(Command("start"))
async def start_handler(msg: Message):
    """
    Handler for the /start command.
    """
    await msg.answer(
        "Hello! I am a bot that can generate images, speech, and analyze images using Google's AI models.\n"
        "Here are my commands:\n\n"
        "• /generate_image <prompt>\n"
        "• /generate_speech <voice_name> <text>\n"
        "• /help\n\n"
        "You can also send me a photo with a caption to analyze it. Just send me a photo and type your question in the caption."
    )

@user.message(Command("help"))
async def help_handler(msg: Message):
    """
    Handler for the /help command.
    """
    help_text = (
        "Here are my commands:\n\n"
        "• /generate_image <prompt>: Generates an image based on your prompt.\n"
        "   Example: `/generate_image a hyperrealistic photo of a cat wearing a tiny hat`\n\n"
        "• /generate_speech <voice_name> <text>: Converts text to speech using a specified voice.\n"
        "   You can choose a voice from this list:\n"
        "   " + ", ".join(VOICES) + "\n"
        "   Example: `/generate_speech Kore The quick brown fox jumps over the lazy dog.`\n\n"
        "• /help: Displays this help message.\n\n"
        "• **Image Analysis**: Send a photo with a text caption to get an analysis of the image."
    )
    await msg.answer(help_text)

@user.message(Command("generate_image"))
async def generate_image_handler(msg: Message, bot: Bot):
    """
    Handler for the /generate_image command.
    """
    prompt = msg.text.removeprefix("/generate_image ").strip()
    if not prompt:
        await msg.answer("Please provide a prompt after the command. Example: `/generate_image a dog in a spacesuit`")
        return

    await bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.UPLOAD_PHOTO)
    
    result = await handle_generate_image(prompt, bot)

    if isinstance(result, BufferedInputFile):
        await msg.answer_photo(photo=result)
    else:
        await msg.answer(result)

@user.message(Command("generate_speech"))
async def generate_speech_handler(msg: Message, bot: Bot):
    """
    Handler for the /generate_speech command.
    """
    log.info(f"Received command: '{msg.text}'")
    command_parts = msg.text.removeprefix("/generate_speech ").strip().split(maxsplit=1)
    log.info(f"Parsed command parts: {command_parts}")
    
    # Check if a voice name and text were provided
    if len(command_parts) < 2:
        await msg.answer(
            "Please provide both a voice name and text after the command.\n"
            "Example: `/generate_speech Kore Hello, world!`\n"
            "Use `/help` to see a list of available voices."
        )
        return

    voice_name, text = command_parts
    
    # Check if the provided voice name is valid
    if voice_name not in VOICES:
        await msg.answer(
            f"The voice name '{voice_name}' is not valid.\n"
            "Please choose a voice from this list:\n"
            "   " + ", ".join(VOICES) + "\n"
            "Example: `/generate_speech Kore Hello, world!`"
        )
        return

    await bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.UPLOAD_VOICE)

    audio_file = await handle_generate_speech(text, voice_name, bot)
    
    if audio_file:
        await msg.answer_voice(voice=audio_file)
    else:
        await msg.answer("Sorry, I couldn't generate the speech. Check the console or `bot.log` for details.")

@user.message(lambda msg: msg.text and not msg.photo)
async def text_handler(msg: Message, bot: Bot):
    """
    Handler for all other text messages.
    """
    await bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.TYPING)
    
    response = await gemini(msg.text)
    await msg.answer(response)

@user.message(lambda msg: msg.photo)
async def image_handler(msg: Message, bot: Bot):
    """
    Handler for messages with photos.
    """
    if msg.caption:
        await bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.TYPING)
        
        photo = msg.photo[-1] # Get the highest resolution photo
        analysis = await handle_analyze_image(photo, bot)

        if analysis:
            await msg.answer(analysis)
        else:
            await msg.answer("Sorry, I couldn't analyze that image.")
    else:
        await msg.answer("Please send an image with a text caption so I know what to analyze.")
