import asyncio
import logging
import base64
from io import BytesIO
import wave
import aiohttp
import json

from aiogram.types import BufferedInputFile
from aiogram import Bot
from config import AITOKEN

# --- API Constants ---
GEMINI_FLASH_VISION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={AITOKEN}"
IMAGEN_3_URL = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={AITOKEN}"
TTS_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={AITOKEN}"

# All of the voice names available for TTS
VOICES = [
    'Zephyr', 'Puck', 'Charon', 'Kore', 'Fenrir', 'Leda', 'Orus', 'Aoede', 'Callirrhoe',
    'Autonoe', 'Enceladus', 'Iapetus', 'Umbriel', 'Algieba', 'Despina', 'Erinome',
    'Algenib', 'Rasalgethi', 'Laomedeia', 'Achernar', 'Alnilam', 'Schedar', 'Gacrux',
    'Pulcherrima', 'Achird', 'Zubenelgenubi', 'Vindemiatrix', 'Sadachbia', 'Sadaltager', 'Sulafat'
]

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Helper function for exponential backoff ---
async def with_exponential_backoff(api_call, max_retries=10, delay=2.0):
    """
    Handles API calls with exponential backoff to manage rate limiting.
    Increased max_retries and initial delay to handle persistent rate-limiting.
    """
    for i in range(max_retries):
        try:
            response = await api_call()
            if response.status == 429:
                logging.warning(f"API call failed with status 429. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                return response
        except Exception as e:
            logging.error(f"API call failed with error: {e}")
            await asyncio.sleep(delay)
            delay *= 2
    raise Exception(f"API call failed after {max_retries} retries.")

# --- API Call Functions ---

async def handle_generate_image(prompt: str, bot: Bot):
    """
    Generates an image from a text prompt using the Imagen 3 API.
    Returns a BufferedInputFile object with the image data or an error string.
    """
    if not prompt:
        prompt = "A serene landscape with a small wooden cabin."

    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1}
    }

    async def api_call(session):
        return await session.post(IMAGEN_3_URL, json=payload, timeout=60)

    try:
        async with aiohttp.ClientSession() as session:
            response = await with_exponential_backoff(lambda: api_call(session))

            if response and response.status == 200:
                result = await response.json()
                if 'predictions' in result and len(result['predictions']) > 0 and result['predictions'][0].get('bytesBase64Encoded'):
                    base64_data = result['predictions'][0]['bytesBase64Encoded']
                    image_data = base64.b64decode(base64_data)
                    return BufferedInputFile(image_data, filename="generated_image.png")
                else:
                    error_message = result.get('predictions', [{}])[0].get('error', {}).get('message', 'Unknown error')
                    logging.error(f"Image generation failed for prompt: '{prompt}'. API returned: {error_message}")
                    return f"Sorry, the image model rejected that prompt: {error_message}"
            else:
                error_details = await response.text()
                logging.error(f"Image generation API returned an error: {response.status}. Details: {error_details}")
                return "Sorry, I couldn't connect to the image generation service."
    except Exception as e:
        logging.error(f"Failed to generate image due to an unexpected error: {e}")
        return "An unexpected error occurred while trying to generate the image."

async def handle_generate_speech(text: str, voice_name: str, bot: Bot):
    """
    Converts text to speech and returns a WAV audio file buffer.
    Returns a BufferedInputFile object with the audio data.
    """
    if not text:
        logging.error("TTS request received with empty text.")
        return None

    if voice_name not in VOICES:
        logging.error(f"Invalid voice_name provided: {voice_name}")
        voice_name = "Kore"

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_name}}
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    async def api_call(session):
        return await session.post(TTS_URL, json=payload, timeout=60)

    try:
        async with aiohttp.ClientSession() as session:
            response = await with_exponential_backoff(lambda: api_call(session))

            if response and response.status == 200:
                result = await response.json()
                part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
                audio_data_b64 = part.get('inlineData', {}).get('data')
                mime_type = part.get('inlineData', {}).get('mimeType')

                if audio_data_b64 and mime_type:
                    pcm_data = base64.b64decode(audio_data_b64)
                    
                    sample_rate = 24000
                    try:
                        rate_string = mime_type.split(';rate=')[-1]
                        sample_rate = int(rate_string)
                    except (ValueError, IndexError):
                        logging.warning(f"Could not parse sample rate from mimeType: {mime_type}. Using default {sample_rate} Hz.")

                    buffer = BytesIO()
                    with wave.open(buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(pcm_data)
                    
                    buffer.seek(0)
                    return BufferedInputFile(buffer.read(), filename="speech.wav")
                else:
                    logging.error("TTS API response missing audio data.")
                    return None
            else:
                error_details = await response.text()
                logging.error(f"TTS API returned a non-200 status code: {response.status}. Details: {error_details}")
                return None
    except Exception as e:
        logging.error(f"Failed to generate TTS after multiple retries: {e}")
        return None
