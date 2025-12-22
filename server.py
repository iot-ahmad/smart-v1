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

HTML_PAGE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8" />
    <title>مساعد صوتي - ESP32 + Groq</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            margin: 0;
            padding: 0;
            direction: rtl;
        }
        .container {
            max-width: 800px;
            margin: 40px auto;
            background: #111827;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        h1 {
            text-align: center;
            margin-bottom: 8px;
            color: #f9fafb;
        }
        .subtitle {
            text-align: center;
            color: #9ca3af;
            margin-bottom: 24px;
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            margin-bottom: 16px;
            font-size: 14px;
        }
        .status-pill {
            padding: 6px 10px;
            border-radius: 999px;
            background: #1f2937;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
        }
        .dot.online { background: #22c55e; }
        .dot.offline { background: #ef4444; }
        .dot.processing { background: #eab308; }
        .card {
            background: #020617;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            border: 1px solid #1f2937;
        }
        .card h2 {
            margin-top: 0;
            font-size: 18px;
            margin-bottom: 8px;
        }
        .text-box {
            background: #020617;
            border-radius: 8px;
            padding: 12px;
            min-height: 60px;
            border: 1px solid #1f2937;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 14px;
        }
        .btn {
            background: linear-gradient(to right, #3b82f6, #6366f1);
            color: white;
            border: none;
            border-radius: 999px;
            padding: 10px 20px;
            font-size: 15px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: transform 0.1s ease, box-shadow 0.1s ease, opacity 0.2s;
            box-shadow: 0 10px 25px rgba(59,130,246,0.4);
        }
        .btn:active {
            transform: translateY(1px);
            box-shadow: 0 5px 15px rgba(59,130,246,0.3);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            box-shadow: none;
        }
        .btn-secondary {
            background: #1f2937;
            box-shadow: none;
        }
        .btn-secondary:hover {
            background: #374151;
        }
        .btn + .btn {
            margin-right: 8px;
        }
        .record-icon {
            width: 10px;
            height: 10px;
            border-radius: 999px;
            background: #ef4444;
            box-shadow: 0 0 0 0 rgba(239,68,68,0.6);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(239,68,68,0.7);
            }
            70% {
                transform: scale(1.4);
                box-shadow: 0 0 0 10px rgba(239,68,68,0);
            }
            100% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(239,68,68,0);
            }
        }
        .wave {
            display: inline-flex;
            align-items: flex-end;
            gap: 3px;
            margin-right: 8px;
        }
        .wave span {
            display: block;
            width: 3px;
            height: 6px;
            background: #60a5fa;
            border-radius: 999px;
            animation: wave 1s infinite ease-in-out;
        }
        .wave span:nth-child(2) { animation-delay: 0.1s; }
        .wave span:nth-child(3) { animation-delay: 0.2s; }
        .wave span:nth-child(4) { animation-delay: 0.3s; }
        .wave span:nth-child(5) { animation-delay: 0.4s; }
        @keyframes wave {
            0%, 100% { height: 6px; }
            50% { height: 16px; }
        }
        .log {
            font-family: monospace;
            font-size: 12px;
            color: #9ca3af;
            max-height: 120px;
            overflow-y: auto;
            background: #020617;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #1f2937;
        }
        .log-line {
            margin-bottom: 2px;
        }
        .footer {
            text-align: center;
            margin-top: 16px;
            font-size: 12px;
            color: #4b5563;
        }
        .error {
            color: #f87171;
        }
        .success {
            color: #4ade80;
        }
        .spinner {
            border: 3px solid #1f2937;
            border-top: 3px solid #60a5fa;
            border-radius: 50%;
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
            display: inline-block;
            margin-left: 6px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
<div class="container">
    <h1>المساعد الصوتي - ESP32 + Groq</h1>
    <div class="subtitle">تسجيل من ESP32 → خادم Flask → Groq (Whisper + Llama) → صوت gTTS → رجوع لـ ESP32</div>

    <div class="status-bar">
        <div class="status-pill" id="server-status-pill">
            <div class="dot offline" id="server-dot"></div>
            <span id="server-status-text">جاري فحص الخادم...</span>
        </div>
        <div class="status-pill" id="esp32-status-pill">
            <div class="dot offline" id="esp32-dot"></div>
            <span id="esp32-status-text">حالة ESP32 غير معروفة</span>
        </div>
    </div>

    <div class="card">
        <h2>تسجيل من ESP32</h2>
        <p style="margin-bottom: 8px; color: #9ca3af;">
            استخدم ESP32 لإرسال ملف الصوت (WAV/RAW) لهذا الخادم عبر HTTP POST إلى <code>/upload</code>.
        </p>
        <button class="btn" id="btn-test-record">
            <div class="record-icon" id="record-icon"></div>
            <span id="record-btn-text">اختبار إرسال ملف صوتي يدويًا</span>
        </button>
        <input type="file" id="audio-file-input" accept="audio/*" style="display:none" />
        <button class="btn btn-secondary" id="btn-clear">
            مسح آخر استجابة
        </button>
    </div>

    <div class="card">
        <h2>النص المستلم من ESP32</h2>
        <div class="text-box" id="user-text-box">لم يصل نص بعد.</div>
    </div>

    <div class="card">
        <h2>رد المساعد (سيتم تحويله لصوت)</h2>
        <div class="text-box" id="assistant-text-box">لا يوجد رد بعد.</div>
        <div style="margin-top: 10px;">
            <button class="btn btn-secondary" id="btn-play-audio" disabled>
                تشغيل الصوت الناتج في المتصفح
            </button>
        </div>
        <audio id="assistant-audio" style="display:none"></audio>
    </div>

    <div class="card">
        <h2>السجل (Logs)</h2>
        <div class="log" id="log-box"></div>
    </div>

    <div class="footer">
        تأكد أن ESP32 يقرأ البيانات من <code>/get-audio-stream</code> ويرسلها إلى DAC/I2S بنفس إعدادات 16kHz, 16-bit, mono.
    </div>
</div>

<script>
    const serverStatusText = document.getElementById('server-status-text');
    const serverDot = document.getElementById('server-dot');
    const esp32StatusText = document.getElementById('esp32-status-text');
    const esp32Dot = document.getElementById('esp32-dot');
    const logBox = document.getElementById('log-box');
    const userTextBox = document.getElementById('user-text-box');
    const assistantTextBox = document.getElementById('assistant-text-box');
    const btnTestRecord = document.getElementById('btn-test-record');
    const btnClear = document.getElementById('btn-clear');
    const recordIcon = document.getElementById('record-icon');
    const recordBtnText = document.getElementById('record-btn-text');
    const audioFileInput = document.getElementById('audio-file-input');
    const btnPlayAudio = document.getElementById('btn-play-audio');
    const assistantAudio = document.getElementById('assistant-audio');

    function log(message, type = '') {
        const line = document.createElement('div');
        line.className = 'log-line';
        if (type === 'error') line.classList.add('error');
        if (type === 'success') line.classList.add('success');
        const timestamp = new Date().toLocaleTimeString();
        line.textContent = `[${timestamp}] ${message}`;
        logBox.appendChild(line);
        logBox.scrollTop = logBox.scrollHeight;
    }

    async function checkStatus() {
        try {
            const res = await fetch('/status');
            if (!res.ok) throw new Error('Status not OK');
            const data = await res.json();
            serverStatusText.textContent = 'الخادم متصل';
            serverDot.classList.remove('offline');
            serverDot.classList.add('online');
            esp32StatusText.textContent = 'حالة ESP32: ' + (data.esp32_status || 'غير معروفة');
            if (data.esp32_status === 'ready') {
                esp32Dot.classList.remove('offline', 'processing');
                esp32Dot.classList.add('online');
            } else if (data.esp32_status === 'processing' || data.esp32_status === 'sending_to_esp32') {
                esp32Dot.classList.remove('offline', 'online');
                esp32Dot.classList.add('processing');
            } else {
                esp32Dot.classList.remove('online', 'processing');
                esp32Dot.classList.add('offline');
            }
        } catch (err) {
            serverStatusText.textContent = 'الخادم غير متصل';
            serverDot.classList.remove('online', 'processing');
            serverDot.classList.add('offline');
            esp32StatusText.textContent = 'لا يمكن جلب حالة ESP32';
            esp32Dot.classList.remove('online', 'processing');
            esp32Dot.classList.add('offline');
            log('تعذر الاتصال بمسار /status', 'error');
        }
    }

    setInterval(checkStatus, 5000);
    checkStatus();

    btnTestRecord.addEventListener('click', () => {
        audioFileInput.click();
    });

    audioFileInput.addEventListener('change', async () => {
        const file = audioFileInput.files[0];
        if (!file) return;

        log('جاري رفع الملف الصوتي للاختبار...', '');
        btnTestRecord.disabled = true;
        btnTestRecord.classList.add('btn-secondary');
        recordBtnText.textContent = 'جارٍ الرفع...';
        recordIcon.style.animationPlayState = 'running';

        const formData = new FormData();
        formData.append('audio', file);

        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();

            if (!res.ok || data.status === 'error') {
                log('خطأ في /upload: ' + (data.error || 'مجهول'), 'error');
                alert('حدث خطأ أثناء معالجة الصوت: ' + (data.error || 'مجهول'));
            } else {
                log('تم رفع الصوت بنجاح. النص: ' + data.text, 'success');
                userTextBox.textContent = data.text || 'لم يتم استخراج نص.';
                assistantTextBox.textContent = data.response || 'لا يوجد رد.';
                btnPlayAudio.disabled = false;
            }
        } catch (err) {
            log('فشل الاتصال بمسار /upload: ' + err.message, 'error');
            alert('تعذر الاتصال بالخادم.');
        } finally {
            btnTestRecord.disabled = false;
            btnTestRecord.classList.remove('btn-secondary');
            recordBtnText.textContent = 'اختبار إرسال ملف صوتي يدويًا';
            recordIcon.style.animationPlayState = 'paused';
            audioFileInput.value = '';
        }
    });

    btnClear.addEventListener('click', async () => {
        try {
            const res = await fetch('/clear', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'cleared') {
                userTextBox.textContent = 'لم يصل نص بعد.';
                assistantTextBox.textContent = 'لا يوجد رد بعد.';
                btnPlayAudio.disabled = true;
                assistantAudio.src = '';
                log('تم مسح آخر استجابة.', 'success');
            }
        } catch (err) {
            log('تعذر الاتصال بمسار /clear', 'error');
        }
    });

    btnPlayAudio.addEventListener('click', async () => {
        try {
            log('جلب الصوت من /get-audio-stream...', '');
            const res = await fetch('/get-audio-stream');
            if (!res.ok) {
                log('لا يوجد صوت متاح حالياً.', 'error');
                alert('لا يوجد صوت متاح الآن.');
                return;
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            assistantAudio.src = url;
            assistantAudio.play().then(() => {
                log('تم تشغيل الصوت في المتصفح.', 'success');
            }).catch(err => {
                log('تعذر تشغيل الصوت في المتصفح: ' + err.message, 'error');
            });
        } catch (err) {
            log('تعذر الاتصال بمسار /get-audio-stream', 'error');
        }
    });
</script>
</body>
</html>
"""

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

        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(audio_file.filename, audio_bytes, audio_file.mimetype),
            language="ar"
        )
        user_text = transcript.text
        esp32_data['text'] = user_text

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
    esp32_data['text'] = ''
    esp32_data['response_text'] = ''
    esp32_data['status'] = 'ready'
    return jsonify({'status': 'cleared'})

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
