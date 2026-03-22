"""
VidSummarize - Flask Backend Application
UNICODE FIXED: Proper handling for multilingual content (Hindi, Tamil, etc.)
ANTIGRAVITY: Optimized processing for long videos
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import sys
import uuid
import json
import threading
import re
from datetime import datetime
from dotenv import load_dotenv # Added import
from video_processor import VideoProcessor

# ========== CRITICAL UNICODE FIX ==========
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ========== TRANSCRIBER SELECTION ==========
USE_SMART_TRANSCRIBER = True

try:
    if USE_SMART_TRANSCRIBER:
        from smart_transcriber import transcribe_audio
        TRANSCRIBER_TYPE = "smart_v2"
        print("✅ Loaded Smart Transcriber v2")
    else:
        raise ImportError("Manual fallback")
except ImportError:
    try:
        from fast_transcriber import Transcriber
        TRANSCRIBER_TYPE = "enhanced"
        print("✅ Loaded Enhanced Transcriber")
    except ImportError:
        from transcriber_legacy import Transcriber
        TRANSCRIBER_TYPE = "legacy"
        print("⚠️ Using Legacy Transcriber")

try:
    from summarizer import Summarizer
except ImportError:
    from summarizer_legacy import Summarizer

from pdf_generator import PDFGenerator


# ========== FLASK APP CONFIGURATION ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['JSON_AS_ASCII'] = False 

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

video_processor = VideoProcessor(
    upload_folder=app.config['UPLOAD_FOLDER'],
    output_folder=app.config['OUTPUT_FOLDER']
)

transcriber = None
summarizer = None
pdf_generator = None

def get_transcriber():
    global transcriber
    if transcriber is None:
        if TRANSCRIBER_TYPE == "smart_v2":
            transcriber = "smart_v2_ready"
        elif TRANSCRIBER_TYPE == "enhanced":
            from fast_transcriber import Transcriber
            transcriber = Transcriber(model_size='base', performance_mode='balanced')
        else:
            from transcriber_legacy import Transcriber
            transcriber = Transcriber(model_size='base')
    return transcriber

def get_summarizer():
    global summarizer
    if summarizer is None:
        # Optimization: use mt5-small for multilingual support if needed
        # but the current Summarizer class handles model selection internally usually
        summarizer = Summarizer(model_name='google/mt5-small')
    return summarizer

def get_pdf_generator():
    global pdf_generator
    if pdf_generator is None:
        pdf_generator = PDFGenerator()
    return pdf_generator

jobs = {}

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv', 'webm'}

def generate_job_id(): return str(uuid.uuid4())

def create_job(job_type, source):
    job_id = generate_job_id()
    jobs[job_id] = {
        'id': job_id, 'type': job_type, 'source': source,
        'stage': 'uploading', 'progress': 0, 'status': 'processing',
        'created_at': datetime.now().isoformat(), 'updated_at': datetime.now().isoformat(),
        'language': None, 'word_count': 0, 'summary_word_count': 0
    }
    return job_id

def normalize_language_code(language):
    if not language: return None
    normalized = language.strip()
    if not normalized or normalized.lower() in {"auto", "detect", "none"}: return None
    return normalized

def update_job(job_id, updates):
    if job_id in jobs:
        jobs[job_id].update(updates)
        jobs[job_id]['updated_at'] = datetime.now().isoformat()
        return True
    return False

def get_job(job_id): return jobs.get(job_id)

def transcribe_with_fallback(audio_path, language=None):
    language = normalize_language_code(language)
    try:
        if TRANSCRIBER_TYPE == "smart_v2":
            result = transcribe_audio(audio_path, language=language)
            return {
                'success': True, 'text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'word_count': len(result.get('text', '').split()),
                'processing_time': result.get('total_time', 0),
                'speed_ratio': result.get('speed_ratio', 0),
                'strategy': result.get('strategy', 'unknown')
            }
    except Exception as e:
        print(f"[App] Smart Transcriber crashed, attempting fallback: {e}")
        pass # Fall through to basic error block or fallback

    return {'success': False, 'error': 'Transcriber failed or encountered hardware constraint'}

def save_utf8(path, text):
    with open(path, 'w', encoding='utf-8') as f: f.write(text)

def clean_transcript_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def process_video_job(job_id, filepath, language=None):
    """Uploaded video processing"""
    try:
        update_job(job_id, {'stage': 'extracting_audio', 'progress': 20})
        result = video_processor.extract_audio_from_video(filepath)
        if not result['success']: raise Exception(result['error'])
        
        audio_path = result['audio_path']
        update_job(job_id, {'stage': 'transcribing', 'progress': 40, 'audio_path': audio_path, 'duration': result.get('duration', 0)})
        
        trans_result = transcribe_with_fallback(audio_path, language=language)
        if not trans_result['success']: raise Exception(trans_result.get('error', 'Transcription failed'))
        
        finalize_common_processing(job_id, trans_result['text'], trans_result.get('language', 'unknown'), result.get('duration', 0), trans_result)
        
    except Exception as e:
        update_job(job_id, {'stage': 'failed', 'status': 'failed', 'error': str(e)})

def process_youtube_job(job_id, youtube_url, language=None):
    """
    Antigravity Phase 1: Caption-first YouTube processing
    """
    try:
        update_job(job_id, {'stage': 'analyzing', 'progress': 10})
        
        # PHASE 1: Try to get captions first
        info = video_processor.get_youtube_info(youtube_url)
        if info['success']:
            duration = info.get('duration', 0)
            video_title = info.get('title', 'Unknown')
            update_job(job_id, {'duration': duration, 'video_title': video_title})
            
            has_subs = len(info.get('subtitles', {})) > 0 or len(info.get('automatic_captions', {})) > 0
            if has_subs:
                update_job(job_id, {'stage': 'extracting_captions', 'progress': 30})
                sub_result = video_processor.download_youtube_subtitles(youtube_url, lang=language or 'en', output_filename=f"{job_id}_subs")
                if sub_result['success']:
                    text = video_processor.extract_text_from_subtitles(sub_result['subtitle_path'])
                    if len(text.strip()) > 50:
                        print(f"[Job {job_id}] Antigravity success: Used captions")
                        finalize_common_processing(job_id, text, sub_result.get('language', 'unknown'), duration, {'strategy': 'captions_direct'})
                        return

        # PHASE 2: Fallback to audio download
        update_job(job_id, {'stage': 'downloading_audio', 'progress': 30})
        result = video_processor.download_youtube_video(youtube_url, audio_only=True)
        if not result['success']: raise Exception(result['error'])
        
        update_job(job_id, {'stage': 'transcribing', 'progress': 45, 'audio_path': result['audio_path']})
        trans_result = transcribe_with_fallback(result['audio_path'], language=language)
        if not trans_result['success']: raise Exception(trans_result.get('error', 'Transcription failed'))
        
        finalize_common_processing(job_id, trans_result['text'], trans_result.get('language', 'unknown'), result.get('duration', 0), trans_result)
        
    except Exception as e:
        update_job(job_id, {'stage': 'failed', 'status': 'failed', 'error': str(e)})

def finalize_common_processing(job_id, transcript_text, language, duration, trans_meta):
    """Common final steps: cleanup, summarize, PDF"""
    cleaned_text = clean_transcript_text(transcript_text)
    transcript_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_transcript.txt")
    save_utf8(transcript_path, transcript_text)
    
    update_job(job_id, {
        'stage': 'summarizing', 'progress': 60,
        'transcript': transcript_text, 'transcript_path': transcript_path,
        'language': language, 'word_count': len(transcript_text.split()),
        'transcription_strategy': trans_meta.get('strategy'),
        'transcription_time': trans_meta.get('processing_time', 0),
        'speed_ratio': trans_meta.get('speed_ratio', 0)
    })
    
    # Summarization
    try:
        summ = get_summarizer()
        summary_result = summ.summarize_text(cleaned_text, language=(language or 'en'))
        if not summary_result.get('success', False):
            summary_text = summ.summarize(cleaned_text)
            provider = "fallback"
            method = "fallback"
        else:
            summary_text = summary_result.get('summary', '')
            provider = summary_result.get('provider', 'unknown')
            method = summary_result.get('method', 'unknown')
    except Exception as e:
        print(f"[App] Summarization error: {e}")
        summary_text = ' '.join(cleaned_text.split()[:200]) + "..."
        provider = "error"
        method = "error"
    
    summary_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_summary.txt")
    save_utf8(summary_path, summary_text)
    
    update_job(job_id, {
        'stage': 'generating_output', 'progress': 80,
        'summary': summary_text, 'summary_path': summary_path,
        'summary_word_count': len(summary_text.split())
    })
    
    # PDF
    pdf_gen = get_pdf_generator()
    meta = {'language': language, 'duration': duration, 'date': datetime.now().strftime('%B %d, %Y')}
    t_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_transcript.pdf")
    s_pdf = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_summary.pdf")
    pdf_gen.generate_transcript_pdf(transcript_text, t_pdf, meta)
    pdf_gen.generate_summary_pdf(summary_text, s_pdf, meta)
    
    update_job(job_id, {
        'stage': 'completed', 'progress': 100, 'status': 'completed',
        'transcript_pdf_path': t_pdf, 'summary_pdf_path': s_pdf
    })

# ========== ROUTES ==========
@app.route('/')
def home(): return render_template('home.html')

@app.route('/processing')
def processing(): return render_template('processing.html', job_id=request.args.get('job_id'))

@app.route('/results')
def results(): return render_template('results.html', job_id=request.args.get('job_id'))

@app.route('/api/status/<job_id>')
def check_status(job_id):
    job = get_job(job_id)
    return jsonify({'success': True, **job}) if job else (jsonify({'error': 'Job not found'}), 404)

@app.route('/api/cancel/<job_id>', methods=['POST'])
def cancel_processing_job(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    update_job(job_id, {'stage': 'failed', 'status': 'failed', 'error': 'Cancelled by user'})
    return jsonify({'success': True, 'message': 'Job cancelled'})

@app.route('/api/health')
def health_check():
    summ = get_summarizer()
    # Check if summarizer has a get_provider_status or just return ok
    provider_info = {}
    try:
        from summarizer import get_provider_status
        provider_info = get_provider_status()
    except:
        pass
    return jsonify({'status': 'ok', 'message': 'API is running', **provider_info})


@app.route('/api/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file: return jsonify({'error': 'No file'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    job_id = create_job('upload', filepath)
    threading.Thread(target=process_video_job, args=(job_id, filepath, request.form.get('language'))).start()
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/api/youtube', methods=['POST'])
def process_youtube():
    url = request.get_json().get('url')
    if not url: return jsonify({'error': 'No URL'}), 400
    job_id = create_job('youtube', url)
    threading.Thread(target=process_youtube_job, args=(job_id, url, request.get_json().get('language'))).start()
    return jsonify({'success': True, 'job_id': job_id})

@app.route('/api/download/transcript/<job_id>')
def download_transcript(job_id):
    job = get_job(job_id)
    path = job.get('transcript_pdf_path') if request.args.get('format') == 'pdf' else job.get('transcript_path')
    return send_file(path, as_attachment=True) if path and os.path.exists(path) else (jsonify({'error': 'File not found'}), 404)

@app.route('/api/download/summary/<job_id>')
def download_summary(job_id):
    job = get_job(job_id)
    path = job.get('summary_pdf_path') if request.args.get('format') == 'pdf' else job.get('summary_path')
    return send_file(path, as_attachment=True) if path and os.path.exists(path) else (jsonify({'error': 'File not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
