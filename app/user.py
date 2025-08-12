import asyncio
from aiogram import Router, types, F, Bot
from aiogram.types import BufferedInputFile
from app.generators import handle_generate_image, handle_generate_speech, handle_analyze_image, gemini, VOICES

# Initialize the router
user = Router()

# --- Command Handlers ---

@user.message(F.text == '/start')
async def send_welcome(message: types.Message):
    """
    Sends a welcome message and lists available commands.
    """
    welcome_text = (
        "Hello! I am a bot powered by Gemini. You can use the following commands:\n\n"
        "/generate_image <prompt> - Generate an image from a text description.\n"
        "/generate_speech <voice_name> <text> - Convert text to speech. Voice names are:\n"
        f"    {', '.join(VOICES)}\n"
        "/analyze_image - Reply to a photo with this command to get a description.\n"
        "Just send me a message for a regular chat!"
    )
    await message.reply(welcome_text)

@user.message(F.text.startswith('/generate_image'))
async def handle_image_request(message: types.Message):
    """
    Handles the /generate_image command.
    """
    await message.bot.send_chat_action(chat_id=message.chat.id, action="upload_photo")
    prompt = message.text[len('/generate_image '):]

    # The handle_generate_image function now returns either a BufferedInputFile or a string
    response_content = await handle_generate_image(prompt, message.bot)

    if isinstance(response_content, BufferedInputFile):
        # If it's an image file, send it
        await message.reply_photo(photo=response_content)
    elif isinstance(response_content, str):
        # If it's a string, it's an error message, so send it as text
        await message.reply(text=response_content)
    else:
        await message.reply("An unknown error occurred with image generation.")

@user.message(F.text.startswith('/generate_speech'))
async def handle_speech_request(message: types.Message):
    """
    Handles the /generate_speech command.
    """
    await message.bot.send_chat_action(chat_id=message.chat.id, action="upload_voice")
    parts = message.text.split(' ', 2)
    if len(parts) < 3:
        await message.reply("Usage: /generate_speech <voice_name> <text>")
        return

    voice_name = parts[1]
    text = parts[2]

    if voice_name not in VOICES:
        await message.reply(f"Invalid voice name. Available voices are:\n{', '.join(VOICES)}")
        return

    audio_file = await handle_generate_speech(text, voice_name, message.bot)
    if audio_file:
        await message.reply_voice(voice=audio_file)
    else:
        await message.reply("Sorry, I couldn't generate the speech.")

@user.message(F.text == '/analyze_image')
async def handle_analyze_image_command(message: types.Message):
    if message.reply_to_message and message.reply_to_message.photo:
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
        photo = message.reply_to_message.photo[-1]
        analysis = await handle_analyze_image(photo, message.bot)
        if analysis:
            await message.reply(analysis)
        else:
            await message.reply("Sorry, I couldn't analyze that image.")
    else:
        await message.reply("Please reply to a photo with this command.")

@user.message()
async def handle_text_message(message: types.Message):
    await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    user_message = message.text
    response = await gemini(user_message)
    await message.reply(response)
