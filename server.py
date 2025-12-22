import os
import io
import logging
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from openai import OpenAI
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure max upload size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Initialize Groq client
try:
    api_key = os.getenv('GROQ_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY/OPENAI_API_KEY not found in environment variables")
        client = None
    else:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    client = None

# Global state for ESP32 communication
esp32_data = {
    'status': 'ready',
    'audio_data': None,
    'has_audio': False,
    'text': '',
    'response_text': ''
}

# ====================== HTML PAGE ======================

HTML_PAGE = """... نفس الـ HTML اللي عندك بدون تغيير ..."""

# ====================== ROUTES ======================

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/upload', methods=['POST'])
def upload_audio():
    try:
        if client is None:
            return jsonify({'status': 'error', 'error': 'Groq API key not configured'}), 500
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'error': 'لم يتم إرسال ملف صوتي'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'error': 'اسم الملف فارغ'}), 400

        esp32_data['status'] = 'processing'

        # 1. Transcribe (Whisper)
        audio_file.seek(0)
        audio_bytes = audio_file.read()
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(audio_file.filename, audio_bytes, audio_file.mimetype),
            language="ar"
        )
        user_text = transcript.text
        esp32_data['text'] = user_text

        # 2. AI Response (Llama)
        chat_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "أنت مساعد صوتي ذكي تتحدث العربية والانجليزيه فقط. أجب باختصار شديد."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        response_text = chat_response.choices[0].message.content
        esp32_data['response_text'] = response_text

        # 3. TTS & Resampling
        logger.info("Converting to speech (gTTS) & Resampling to 16kHz...")

        tts = gTTS(text=response_text, lang='ar')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        data, original_rate = sf.read(mp3_fp, dtype='float32')

        if len(data.shape) > 1:
            data = data.mean(axis=1)

        target_rate = 16000
        if original_rate != target_rate:
            number_of_samples = round(len(data) * float(target_rate) / original_rate)
            data = np.interp(
                np.linspace(0.0, 1.0, number_of_samples, endpoint=False),
                np.linspace(0.0, 1.0, len(data), endpoint=False),
                data
            )

        data_int16 = (data * 32767).astype(np.int16)
        raw_audio_bytes = data_int16.tobytes()

        esp32_data['audio_data'] = raw_audio_bytes
        esp32_data['has_audio'] = True
        esp32_data['status'] = 'sending_to_esp32'
        logger.info(f"Audio ready. Sample Rate: {target_rate}, Size: {len(raw_audio_bytes)}")

        return jsonify({
            'status': 'ok',
            'text': user_text,
            'response': response_text,
            'audio_url': '/get-audio-stream'
        })

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        esp32_data['status'] = 'ready'
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/get-audio-stream', methods=['GET'])
def get_audio_stream():
    try:
        if not esp32_data['has_audio'] or esp32_data['audio_data'] is None:
            return jsonify({'error': 'No audio available'}), 404

        return send_file(
            io.BytesIO(esp32_data['audio_data']),
            mimetype='application/octet-stream',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'server': 'online', 'esp32_status': esp32_data['status']})

@app.route('/clear', methods=['POST'])
def clear_audio():
    esp32_data['audio_data'] = None
    esp32_data['has_audio'] = False
    return jsonify({'status': 'cleared'})

# ====== روت اختبار اتصال Groq ======
@app.route('/test-groq')
def test_groq():
    try:
        if client is None:
            return "client is None (no API key)", 500

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
