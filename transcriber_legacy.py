"""
VidSummarize - Transcription Module
Handles speech-to-text transcription using OpenAI Whisper
"""

import whisper
import torch
import os
from datetime import timedelta


class Transcriber:
    """Handles audio transcription using Whisper"""
    
    def __init__(self, model_size='base'):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size (str): Whisper model size
                - 'tiny': Fastest, least accurate (~1GB RAM)
                - 'base': Fast, good accuracy (~1GB RAM) - DEFAULT
                - 'small': Balanced (~2GB RAM)
                - 'medium': High accuracy (~5GB RAM)
                - 'large': Best accuracy (~10GB RAM)
        """
        print(f"[Transcriber] Loading Whisper model: {model_size}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Transcriber] Using device: {self.device}")
        
        # Load Whisper model
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            print(f"[Transcriber] Model loaded successfully")
        except Exception as e:
            print(f"[Transcriber] Error loading model: {str(e)}")
            raise
        
        self.model_size = model_size
    
    def transcribe_audio(self, audio_path, language=None, task='transcribe'):
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to audio file
            language (str): Language code (e.g., 'en', 'es', 'fr') or None for auto-detect
            task (str): 'transcribe' or 'translate' (translate to English)
            
        Returns:
            dict: Transcription result with text, language, and segments
        """
        try:
            print(f"[Transcriber] Transcribing: {audio_path}")
            print(f"[Transcriber] Language: {language or 'auto-detect'}")
            print(f"[Transcriber] Task: {task}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Transcribe options
            options = {
                'task': task,
                'verbose': False
            }
            
            # Add language if specified
            if language:
                options['language'] = language
            
            # Perform transcription
            result = self.model.transcribe(audio_path, **options)
            
            # Extract information
            transcript_text = result['text'].strip()
            detected_language = result['language']
            segments = result.get('segments', [])
            
            print(f"[Transcriber] Transcription complete")
            print(f"[Transcriber] Detected language: {detected_language}")
            print(f"[Transcriber] Text length: {len(transcript_text)} characters")
            print(f"[Transcriber] Segments: {len(segments)}")
            
            return {
                'success': True,
                'text': transcript_text,
                'language': detected_language,
                'segments': segments,
                'word_count': len(transcript_text.split())
            }
            
        except Exception as e:
            print(f"[Transcriber] Error: {str(e)}")
            return {
                'success': False,
                'error': f"Transcription failed: {str(e)}"
            }
    
    def transcribe_with_timestamps(self, audio_path, language=None):
        """
        Transcribe audio with detailed timestamps
        
        Args:
            audio_path (str): Path to audio file
            language (str): Language code or None
            
        Returns:
            dict: Result with formatted transcript including timestamps
        """
        result = self.transcribe_audio(audio_path, language)
        
        if not result['success']:
            return result
        
        # Format segments with timestamps
        formatted_segments = []
        for segment in result['segments']:
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            formatted_segments.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
        
        # Create formatted transcript with timestamps
        formatted_transcript = self._create_formatted_transcript(formatted_segments)
        
        result['formatted_transcript'] = formatted_transcript
        result['formatted_segments'] = formatted_segments
        
        return result
    
    def _format_timestamp(self, seconds):
        """
        Convert seconds to HH:MM:SS format
        
        Args:
            seconds (float): Time in seconds
            
        Returns:
            str: Formatted timestamp
        """
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _create_formatted_transcript(self, segments):
        """
        Create a formatted transcript with timestamps
        
        Args:
            segments (list): List of segment dictionaries
            
        Returns:
            str: Formatted transcript text
        """
        lines = []
        for segment in segments:
            line = f"[{segment['start']} --> {segment['end']}] {segment['text']}"
            lines.append(line)
        
        return '\n\n'.join(lines)
    
    def save_transcript(self, transcript_text, output_path):
        """
        Save transcript to text file
        
        Args:
            transcript_text (str): Transcript text
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            print(f"[Transcriber] Transcript saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"[Transcriber] Save error: {str(e)}")
            return False
    
    def get_supported_languages(self):
        """
        Get list of supported languages
        
        Returns:
            list: List of language codes
        """
        return list(whisper.tokenizer.LANGUAGES.keys())
    
    def get_language_name(self, language_code):
        """
        Get language name from code
        
        Args:
            language_code (str): Language code (e.g., 'en')
            
        Returns:
            str: Language name (e.g., 'English')
        """
        return whisper.tokenizer.LANGUAGES.get(language_code, language_code)


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("VidSummarize - Transcriber Test")
    print("=" * 60)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Initialize transcriber
    try:
        print("\nInitializing Whisper model (this may take a minute)...")
        transcriber = Transcriber(model_size='base')
        
        print("\n✅ Transcriber initialized successfully!")
        print(f"Model size: {transcriber.model_size}")
        print(f"Device: {transcriber.device}")
        
        # Show supported languages (first 10)
        languages = transcriber.get_supported_languages()
        print(f"\nSupported languages: {len(languages)}")
        print("Sample languages:", ', '.join(languages[:10]))
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Transcriber module loaded successfully!")
    print("=" * 60)