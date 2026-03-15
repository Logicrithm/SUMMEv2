"""
VidSummarize - Video Processing Module
Handles video download, audio extraction, and file processing
"""

import os
import subprocess
import yt_dlp
import ffmpeg
import re
from datetime import datetime


class VideoProcessor:
    """Handles all video and audio processing operations"""
    
    def __init__(self, upload_folder='uploads', output_folder='outputs'):
        self.upload_folder = upload_folder
        self.output_folder = output_folder
        
        # Create folders if they don't exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path for output audio file (optional)
            
        Returns:
            dict: Result with audio_path and metadata
        """
        try:
            print(f"[VideoProcessor] Extracting audio from: {video_path}")
            
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(self.output_folder, f"{base_name}_audio.wav")
            
            # Extract audio using ffmpeg-python
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            
            # Get video metadata
            metadata = self._get_video_metadata(video_path)
            
            print(f"[VideoProcessor] Audio extracted successfully: {output_path}")
            
            return {
                'success': True,
                'audio_path': output_path,
                'duration': metadata.get('duration', 0),
                'size': os.path.getsize(output_path)
            }
            
        except ffmpeg.Error as e:
            error_message = str(e)
            if hasattr(e, 'stderr') and e.stderr:
                try:
                    error_message = e.stderr.decode()
                except:
                    error_message = str(e.stderr)
            print(f"[VideoProcessor] FFmpeg error: {error_message}")
            return {
                'success': False,
                'error': f"Audio extraction failed: {error_message}"
            }
        except Exception as e:
            print(f"[VideoProcessor] Error: {str(e)}")
            return {
                'success': False,
                'error': f"Audio extraction failed: {str(e)}"
            }
    
    def get_youtube_info(self, youtube_url):
        """
        Antigravity Move: Get metadata and caption info from YouTube without downloading
        """
        try:
            print(f"[VideoProcessor] Fetching YouTube info: {youtube_url}")
            ydl_opts = {
                'format': 'best',
                'cookiefile': r"C:\Users\KIIT\Downloads\www.youtube.com_cookies.txt",
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return {
                    'success': True,
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'subtitles': info.get('subtitles', {}),
                    'automatic_captions': info.get('automatic_captions', {}),
                    'description': info.get('description', ''),
                    'raw_info': info
                }
        except Exception as e:
            print(f"[VideoProcessor] Error fetching YouTube info: {str(e)}")
            return {'success': False, 'error': str(e)}

    def download_youtube_subtitles(self, youtube_url, lang='en', output_filename=None):
        """
        Antigravity Move: Download subtitles with improved language support
        """
        try:
            if output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"subs_{timestamp}"
            
            output_template = os.path.join(self.output_folder, output_filename)
            
            # Expanded language list to catch regional variants (hi-IN, etc)
            target_langs = [lang, 'en', 'hi', 'ta', 'te', 'bn', 'ml', 'kn', 'mr', 'gu']
            lang_str = ','.join(set(target_langs))
            
            ydl_opts = {
                'format': 'best',
                'skip_download': True,
                'write_auto_subs': True,
                'write_subs': True,
                'subtitleslangs': target_langs,
                'outtmpl': f"{output_template}.%(ext)s",
                'cookiefile': r"C:\Users\KIIT\Downloads\www.youtube.com_cookies.txt",
                'quiet': True,
                'no_warnings': True
            }
            
            print(f"[VideoProcessor] Downloading subs for {youtube_url} (preferring {lang})")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            
            # Robust file discovery: look for the most relevant subtitle file
            # Prioritize: 1. Requested lang, 2. Any manual sub, 3. Any auto sub
            possible_files = os.listdir(self.output_folder)
            found_path = None
            found_lang = 'unknown'
            
            # First pass: look for exact language match
            for file in possible_files:
                if file.startswith(output_filename) and (file.endswith('.vtt') or file.endswith('.srt')):
                    # Check if this file contains our target language code
                    if f".{lang}." in file:
                        found_path = os.path.join(self.output_folder, file)
                        found_lang = lang
                        break
            
            # Second pass: if no exact match, take any that starts with our filename
            if not found_path:
                for file in possible_files:
                    if file.startswith(output_filename) and (file.endswith('.vtt') or file.endswith('.srt')):
                        found_path = os.path.join(self.output_folder, file)
                        # Extract language from filename (e.g., subs_123.hi.vtt -> hi)
                        parts = file.split('.')
                        if len(parts) >= 3:
                            found_lang = parts[-2]
                        break
            
            if found_path:
                print(f"[VideoProcessor] Found subtitle: {found_path} ({found_lang})")
                return {
                    'success': True,
                    'subtitle_path': found_path,
                    'language': found_lang
                }
            
            return {'success': False, 'error': f"No subtitles found for mapping: {lang}"}
            
        except Exception as e:
            print(f"[VideoProcessor] Subtitle download error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def extract_text_from_subtitles(self, subtitle_path):
        """
        Clean VTT/SRT file to plain text
        """
        try:
            if not os.path.exists(subtitle_path):
                return ""
                
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove WEBVTT header
            content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
            # Remove timestamps
            content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
            content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}.*?\n', '', content)
            # Remove HTML-like tags
            content = re.sub(r'<[^>]*>', '', content)
            # Remove duplicate lines
            lines = content.split('\n')
            unique_lines = []
            prev_line = ""
            for line in lines:
                clean_line = line.strip()
                if clean_line and clean_line != prev_line:
                    unique_lines.append(clean_line)
                    prev_line = clean_line
            
            return ' '.join(unique_lines)
        except Exception as e:
            print(f"[VideoProcessor] Error parsing subtitles: {e}")
            return ""

    def download_youtube_video(self, youtube_url, output_filename=None, audio_only=True):
        """
        Prefer audio-only for transcription efficiency
        """
        try:
            print(f"[VideoProcessor] Downloading YouTube ({'audio-only' if audio_only else 'video'}): {youtube_url}")
            
            if output_filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"youtube_{timestamp}"
            
            # yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best' if audio_only else 'best',
                'outtmpl': os.path.join(self.upload_folder, f"{output_filename}.%(ext)s"),
                'cookiefile': r"C:\Users\KIIT\Downloads\www.youtube.com_cookies.txt",
                'quiet': False,
            }
            
            # Download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                video_path = ydl.prepare_filename(info)
                audio_path = os.path.join(self.output_folder, f"{output_filename}_audio.wav")
                
                # Some formats might map slightly differently after merge, but prepare_filename is usually robust.
                if not os.path.exists(video_path):
                    # fallback search
                    base_name = os.path.join(self.upload_folder, output_filename)
                    for file in os.listdir(self.upload_folder):
                        if file.startswith(output_filename):
                            video_path = os.path.join(self.upload_folder, file)
                            break
            
            # Extract audio to WAV
            audio_result = self.extract_audio_from_video(video_path, audio_path)
            
            if audio_result['success']:
                audio_result['video_path'] = video_path
                audio_result['video_title'] = title
                audio_result['duration'] = duration
                return audio_result
            else:
                return audio_result
                
        except Exception as e:
            print(f"[VideoProcessor] YouTube error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _get_video_metadata(self, video_path):
        """Get video metadata using ffprobe"""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
            
            return {
                'duration': duration,
                'size': int(probe['format']['size'])
            }
        except Exception as e:
            print(f"[VideoProcessor] Metadata extraction error: {str(e)}")
            return {'duration': 0, 'size': 0}
    
    def get_audio_duration(self, audio_path):
        """Get duration of audio file"""
        try:
            probe = ffmpeg.probe(audio_path)
            return float(probe['format']['duration'])
        except Exception as e:
            return 0
    
    def cleanup_file(self, file_path):
        """Delete a file safely"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            return False


if __name__ == "__main__":
    processor = VideoProcessor()
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        print("✅ FFmpeg is installed")
    except FileNotFoundError:
        print("❌ FFmpeg is not installed")
