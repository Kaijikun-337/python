import base64
import requests
import io
import wave
from PIL import Image
import mimetypes

from config import AITOKEN

def _pcm_to_wav(pcm_data, sample_rate):
    wav_file = io.BytesIO()
    with wave.open(wav_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    wav_file.seek(0)
    return wav_file

def generate_image(prompt: str):
    # The API endpoint for the image generation model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key={AITOKEN}"
    
    # The payload MUST have "instances" as a list of dictionaries.
    # Each dictionary should contain the "prompt".
    payload = {
        "instances": [
            {
                "prompt": prompt
            }
        ],
        "parameters": {
            "sampleCount": 1
        }
    }
    
    try:
        # Check if the prompt is empty before making the API call
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        
        result = response.json()
        base64_data = result['predictions'][0]['bytesBase64Encoded']
        
        # Decode the base64 data to get the image bytes
        image_bytes = base64.b64decode(base64_data)
        
        # Open the image from bytes and save it
        image = Image.open(io.BytesIO(image_bytes))
        file_path = "generated_image.png"
        image.save(file_path)
        
        return file_path
        
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def analyze_image(image_path, question):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={AITOKEN}"
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type: mime_type = "image/jpeg"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": question },
                    { "inlineData": { "mimeType": mime_type, "data": encoded_string } }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        analysis_text = result['candidates'][0]['content']['parts'][0]['text']
        return analysis_text
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def generate_speech(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={AITOKEN}"
    payload = {
        "contents": [{ "parts": [{ "text": text }] }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": { "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": "Puck" } } }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        audio_data_b64 = result['candidates'][0]['content']['parts'][0]['inlineData']['data']
        audio_mime_type = result['candidates'][0]['content']['parts'][0]['inlineData']['mimeType']
        sample_rate = 16000
        if 'rate=' in audio_mime_type: sample_rate = int(audio_mime_type.split('rate=')[1])
        pcm_data = base64.b64decode(audio_data_b64)
        wav_file = _pcm_to_wav(pcm_data, sample_rate)
        file_path = "generated_speech.wav"
        with open(file_path, "wb") as f:
            f.write(wav_file.read())
        return file_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None