import os
import io
import logging
import soundfile as sf
import numpy as np
import httpx
import requests
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Initialize Groq client
try:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.error("GROQ_API_KEY not found in environment variables")
        client = None
    else:
        http_client = httpx.Client(timeout=30.0)
        client = Groq(api_key=api_key, http_client=http_client)
        logger.info("Groq client (official) initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    client = None

esp32_data = {
    'status': 'ready',
    'audio_data': None,
    'has_audio': False,
    'text': '',
    'response_text': ''
}

# ====================== HTML PAGE ======================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>مساعد صوتي ذكي - Smart Voice Assistant</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: radial-gradient(at 0% 0%, #3b3b3b 0%, #050505 60%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
            color: #f5f5f5;
        }

        .container {
            background: rgba(15, 15, 15, 0.96);
            border-radius: 20px;
            padding: 32px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.7);
            max-width: 600px;
            width: 100%;
            animation: fadeIn 0.5s ease-in;
            border: 1px solid rgba(120, 120, 120, 0.4);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        h1 {
            color: #e5e1ff;
            text-align: center;
            margin-bottom: 10px;
            font-size: 30px;
            font-weight: 700;
        }

        .subtitle {
            text-align: center;
            color: #a3a3a3;
            margin-bottom: 18px;
            font-size: 13px;
        }

        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 10px;
            justify-content: center;
            flex-wrap: wrap;
        }

        button {
            padding: 12px 26px;
            border: none;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #fdfdfd;
            font-family: inherit;
            letter-spacing: 0.3px;
        }

        #recordBtn {
            background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
            box-shadow: 0 0 18px rgba(129, 140, 248, 0.45);
        }

        #stopBtn {
            background: linear-gradient(135deg, #f97373 0%, #ef4444 100%);
            display: none;
            box-shadow: 0 0 18px rgba(248, 113, 113, 0.45);
        }

        #clearBtn {
            background: linear-gradient(135deg, #facc15 0%, #fb923c 100%);
            color: #111827;
        }

        button:hover:not(:disabled) {
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 14px 25px rgba(0,0,0,0.4);
        }

        button:active:not(:disabled) {
            transform: translateY(0) scale(0.99);
        }

        button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }

        .model-select-wrapper {
            margin-bottom: 18px;
            text-align: center;
        }

        select {
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #4b5563;
            background: #020617;
            color: #e5e7eb;
            font-size: 13px;
            outline: none;
        }

        .status {
            background: rgba(17, 24, 39, 0.9);
            padding: 18px;
            border-radius: 14px;
            margin-bottom: 18px;
            min-height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(55, 65, 81, 0.9);
        }

        .status-text {
            color: #e5e7eb;
            font-size: 14px;
            text-align: center;
        }

        .recording {
            animation: pulse 1.4s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50%      { opacity: 0.6; transform: scale(1.01); }
        }

        .result {
            background: rgba(22, 163, 74, 0.08);
            border-radius: 14px;
            margin-top: 18px;
            display: none;
            animation: slideIn 0.25s ease-out;
            border: 1px solid rgba(34, 197, 94, 0.4);
            padding: 16px 18px;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .result h3 {
            color: #bbf7d0;
            margin-bottom: 6px;
            font-size: 15px;
        }

        .result p {
            color: #e5e5e5;
            line-height: 1.6;
            font-size: 14px;
        }

        .loader {
            border: 4px solid rgba(55, 65, 81, 0.9);
            border-top: 4px solid #8b5cf6;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: spin 0.9s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0%   { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: rgba(127, 29, 29, 0.16);
            border: 1px solid rgba(248, 113, 113, 0.55);
        }

        .error .status-text {
            color: #fecaca;
        }

        .success {
            background: rgba(22, 163, 74, 0.12);
            border-color: rgba(34, 197, 94, 0.6);
        }

        .footer {
            text-align: center;
            margin-top: 22px;
            padding-top: 14px;
            border-top: 1px solid rgba(55, 65, 81, 0.85);
            color: #9ca3af;
            font-size: 11px;
        }

        .footer a {
            color: #a5b4fc;
            text-decoration: none;
        }

        .footer a:hover {
            text-decoration: underline;
        }

        @media (max-width: 600px) {
            .container { padding: 22px; }
            h1 { font-size: 24px; }
            button { padding: 10px 18px; font-size: 13px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 مساعد صوتي ذكي</h1>
        <p class="subtitle">مدعوم بـ Groq Whisper + (Llama 3.1 / GPT‑OSS 120B)</p>

        <div class="model-select-wrapper">
            <label for="modelSelect" style="font-size:13px;color:#9ca3af;">اختر النموذج:</label>
            <select id="modelSelect">
                <option value="llama" selected>🦙 Llama 3.1 8B (سريع وخفيف)</option>
                <option value="strong">🧠 GPT‑OSS 120B (قوي جداً)</option>
            </select>
        </div>

        <div class="controls">
            <button id="recordBtn">🎙️ ابدأ التسجيل</button>
            <button id="stopBtn">⏹️ إيقاف التسجيل</button>
            <button id="clearBtn">🗑️ مسح</button>
        </div>

        <div class="status" id="statusBox">
            <div class="status-text" id="statusText">اضغط على زر التسجيل للبدء</div>
        </div>

        <div class="result" id="result">
            <h3>📝 النص المحول:</h3><p id="transcriptText"></p>
            <h3 style="margin-top: 10px;">🤖 رد المساعد:</h3><p id="responseText"></p>
        </div>

        <div class="footer">
            <p>Powered by Groq &amp; Google TTS</p>
        </div>
    </div>

    <script>
        let mediaRecorder; let audioChunks = [];
        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const clearBtn = document.getElementById('clearBtn');
        const statusBox = document.getElementById('statusBox');
        const statusText = document.getElementById('statusText');
        const result = document.getElementById('result');
        const transcriptText = document.getElementById('transcriptText');
        const responseText = document.getElementById('responseText');
        const modelSelect = document.getElementById('modelSelect');

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            statusText.innerHTML = '❌ المتصفح لا يدعم تسجيل الصوت';
            statusBox.classList.add('error');
            recordBtn.disabled = true;
        }

        recordBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 44100 }
                });

                const options = { mimeType: 'audio/webm' };
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'audio/ogg; codecs=opus';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        options.mimeType = 'audio/mp4';
                    }
                }

                mediaRecorder = new MediaRecorder(stream, options);
                audioChunks = [];
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: options.mimeType });
                    await uploadAudio(audioBlob);
                };
                mediaRecorder.start();

                recordBtn.style.display = 'none';
                stopBtn.style.display = 'inline-block';
                clearBtn.disabled = true;
                statusText.innerHTML = '🔴 جاري التسجيل... تحدث الآن';
                statusText.classList.add('recording');
                statusBox.classList.remove('error', 'success');
                result.style.display = 'none';
            } catch (error) {
                console.error('Error:', error);
                statusText.innerHTML = '❌ خطأ في الوصول للميكروفون. تأكد من السماح بالوصول.';
                statusBox.classList.add('error');
            }
        });

        stopBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
            stopBtn.style.display = 'none';
            recordBtn.style.display = 'inline-block';
            statusText.classList.remove('recording');
            statusBox.classList.remove('error', 'success');
            statusText.innerHTML = '<div class="loader"></div>';
        });

        clearBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                if (response.ok) {
                    result.style.display = 'none';
                    statusText.innerHTML = 'تم المسح بنجاح ✅';
                    statusBox.classList.add('success');
                    setTimeout(() => {
                        statusText.innerHTML = 'اضغط على زر التسجيل للبدء';
                        statusBox.classList.remove('success');
                    }, 2000);
                }
            } catch (error) {
                console.error('Clear error:', error);
            }
        });

        async function uploadAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.webm');

            const selectedModel = modelSelect.value || 'llama';
            const url = '/upload?model=' + encodeURIComponent(selectedModel);

            try {
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.status === 'ok') {
                    statusText.innerHTML = '✅ تم المعالجة بنجاح! (النموذج: ' + (data.model || selectedModel) + ')';
                    statusBox.classList.add('success');
                    transcriptText.textContent = data.text;
                    responseText.textContent = data.response;
                    result.style.display = 'block';
                    clearBtn.disabled = false;
                } else {
                    statusText.innerHTML = '❌ حدث خطأ: ' + (data.error || 'خطأ غير معروف');
                    statusBox.classList.add('error');
                    clearBtn.disabled = false;
                }
            } catch (error) {
                console.error('Upload error:', error);
                statusText.innerHTML = '❌ خطأ في الاتصال بالسيرفر. حاول مرة أخرى.';
                statusBox.classList.add('error');
                clearBtn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

# ====================== MODEL SELECTION ======================

def choose_model(req):
    """
    llama (افتراضي) أو strong (يستخدم GPT‑OSS 120B القوي على Groq).
    """
    m = (req.args.get('model') or '').lower().strip()
    if m in ['strong', 'gpt', 'gpt-oss']:
        return "openai/gpt-oss-120b"  # نموذج قوي جداً مستضاف على Groq[web:422]
    return "llama-3.1-8b-instant"

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

        audio_file.seek(0)
        audio_bytes = audio_file.read()

        # Whisper
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=("audio.webm", audio_bytes, audio_file.mimetype),
            language="ar"
        )
        user_text = transcript.text
        esp32_data['text'] = user_text

        # اختيار الموديل
        selected_model = choose_model(request)
        logger.info(f"Using chat model: {selected_model}")

        system_prompt = (
            "أنت مساعد صوتي ذكي تتحدث العربية والانجليزية فقط. "
            "أجب باختصار شديد. عند سؤالك مين صانعك قل: احمد البطاينة تاج راسكو. "
            "وعند سؤالك عن افضل لاعب كرة قدم بالعالم جاوب: كريستيانو رونالدو."
        )

        chat_response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        response_text = chat_response.choices[0].message.content
        esp32_data['response_text'] = response_text

        # TTS + Resample 16kHz
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
            'model': selected_model,
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
    esp32_data['text'] = ''
    esp32_data['response_text'] = ''
    esp32_data['status'] = 'ready'
    return jsonify({'status': 'cleared'})

@app.route('/test-net')
def test_net():
    try:
        r = requests.get("https://api.groq.com/openai/v1/models", timeout=10)
        return f"status={r.status_code}"
    except Exception as e:
        return f"NET ERROR: {e}", 500

@app.route('/test-groq')
def test_groq():
    try:
        if client is None:
            return "client is None (no API key configured)", 500

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
