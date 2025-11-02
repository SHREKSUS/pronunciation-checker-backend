from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import io

app = Flask(__name__)

# Настройка CORS для продакшена
CORS(app, origins=[
    "http://localhost:3000",
    "https://zphc-ai-teacheren.netlify.app",
    "https://pronunciation-checker-backend.onrender.com/"  # замените на ваш GitHub Pages
])

# Безопасное получение API ключа
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is required")
client = Groq(api_key=api_key)

# Конфигурация для аудио файлов
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'webm'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/api/check_pronunciation', methods=['POST'])
def check_pronunciation():
    try:
        # Получаем данные из запроса
        audio_file = request.files.get('audio')
        reference_text = request.form.get('reference_text', '').strip()
        
        if not audio_file or not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid audio file'}), 400
        
        if not reference_text:
            return jsonify({'error': 'Reference text is required'}), 400

        # Используем BytesIO для работы с файлом в памяти
        audio_data = audio_file.read()
        audio_stream = io.BytesIO(audio_data)
        audio_stream.name = "audio.wav"

        # Транскрибация аудио через Groq Whisper
        transcription = client.audio.transcriptions.create(
            file=audio_stream,
            model="whisper-large-v3",
            language="en",
            response_format="text"
        )
        
        # Анализ текста через Groq Chat
        analysis_prompt = f"""
        Compare the reference English phrase with the user's spoken version and provide detailed feedback:

        REFERENCE PHRASE: "{reference_text}"
        USER'S VERSION: "{transcription}"

        Analyze:
        1. Pronunciation accuracy
        2. Grammar correctness
        3. Overall comprehension
        4. Specific mistakes if any

        Provide response in Russian with this structure:
        - Точность произношения: [оценка/10 и комментарий]
        - Грамматика: [оценка/10 и комментарий]
        - Общая понятность: [оценка/10 и комментарий]
        - Советы по улучшению: [конкретные рекомендации]
        """

        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional English teacher. Provide constructive feedback on English pronunciation and grammar."
                },
                {
                    "role": "user",
                    "content": analysis_prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        analysis = chat_completion.choices[0].message.content

        return jsonify({
            'transcription': transcription,
            'analysis': analysis,
            'reference_text': reference_text
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Обработка для React Router в продакшене
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    return jsonify({
        'message': 'Backend API is running. Use frontend to access the application.',
        'endpoints': {
            'health_check': '/api/health',
            'pronunciation_check': '/api/check_pronunciation'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
