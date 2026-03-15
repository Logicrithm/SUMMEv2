"""
VidSummarize - Smart Transcriber v2
Implements the "Fast + Smart" philosophy:
- Audio profiling and analysis
- Silence-based chunking
- Early language detection
- Adaptive model selection
- Parallel chunk processing
- Structured output for summarization
"""

try:
    from faster_whisper import WhisperModel
    _HAS_FASTER = True
except Exception:
    WhisperModel = None
    _HAS_FASTER = False
    try:
        import whisper
    except Exception:
        whisper = None

import os
import time
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import json


# ========== CONFIGURATION ==========

# Model selection by strategy
MODELS = {
    "direct": "tiny",      # Short videos, fast
    "chunked": "base",     # Medium videos, balanced
    "streaming": "small"   # Long videos, quality
}

# Indian language codes
INDIAN_LANGS = {
    "hi", "ta", "te", "bn", "ml", "kn", "mr", "gu", "pa", "ur"
}

# Performance mode
PERFORMANCE_MODE = os.getenv("PERFORMANCE_MODE", "balanced")

# Parallel workers for chunking (capped by device)
MAX_WORKERS = int(os.getenv("TRANSCRIBE_WORKERS", "4"))
CPU_WORKERS = 2
GPU_WORKERS = 4

# Silence detection threshold (dB)
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "-40"))

# Chunk parameters
MIN_CHUNK_DURATION = 10  # seconds
MAX_CHUNK_DURATION = 180  # seconds

VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


class AudioProfiler:
    """Analyzes audio to determine optimal processing strategy"""
    
    @staticmethod
    def profile(audio_path: str) -> Dict:
        """
        Profile audio file to determine characteristics
        Returns: {duration, avg_loudness, silence_ratio, strategy}
        """
        start = time.time()
        
        try:
            # Get duration using ffprobe
            duration = AudioProfiler._get_duration(audio_path)
            
            # Get loudness statistics
            loudness_stats = AudioProfiler._get_loudness(audio_path)
            
            # Determine strategy based on duration
            if duration < 600:  # < 10 minutes
                strategy = "direct"
            elif duration < 2400:  # 10-40 minutes
                strategy = "chunked"
            else:  # > 40 minutes
                strategy = "streaming"
            
            profile_time = time.time() - start
            
            profile = {
                "duration": duration,
                "duration_minutes": duration / 60,
                "avg_loudness": loudness_stats.get("mean_volume", -20),
                "strategy": strategy,
                "profile_time": profile_time
            }
            
            if VERBOSE:
                print(f"[Profiler] Duration: {duration:.1f}s ({duration/60:.1f}m)")
                print(f"[Profiler] Strategy: {strategy}")
                print(f"[Profiler] Profile time: {profile_time:.2f}s")
            
            return profile
            
        except Exception as e:
            print(f"[Profiler] Error: {e}")
            # Fallback to direct strategy
            return {
                "duration": 0,
                "duration_minutes": 0,
                "avg_loudness": -20,
                "strategy": "direct",
                "profile_time": 0,
                "error": str(e)
            }
    
    @staticmethod
    def _get_duration(audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except Exception as e:
            if VERBOSE:
                print(f"[Profiler] Duration detection failed: {e}")
            return 0
    
    @staticmethod
    def _get_loudness(audio_path: str) -> Dict:
        """Get loudness statistics using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'volumedetect',
                '-f', 'null', '-'
            ]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            # Parse mean_volume from stderr
            stderr = result.stderr
            mean_volume = -20  # default
            
            for line in stderr.split('\n'):
                if 'mean_volume' in line:
                    try:
                        mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                    except:
                        pass
            
            return {"mean_volume": mean_volume}
            
        except Exception as e:
            if VERBOSE:
                print(f"[Profiler] Loudness detection failed: {e}")
            return {"mean_volume": -20}


class AudioChunker:
    """Smart audio chunking based on silence detection"""
    
    @staticmethod
    def chunk_by_silence(audio_path: str, max_duration: int = MAX_CHUNK_DURATION) -> List[Dict]:
        """
        Split audio into chunks based on silence
        Returns: [{start, end, duration, chunk_path}]
        """
        start = time.time()
        
        try:
            # Detect silence points
            silence_points = AudioChunker._detect_silence(audio_path)
            
            if not silence_points:
                # No silence detected, fallback to time-based chunking
                if VERBOSE:
                    print("[Chunker] No silence detected, using time-based chunking")
                return AudioChunker._chunk_by_time(audio_path, max_duration)
            
            # Create chunks from silence points
            chunks = AudioChunker._create_chunks_from_silence(
                audio_path, 
                silence_points, 
                max_duration
            )
            
            chunk_time = time.time() - start
            
            if VERBOSE:
                print(f"[Chunker] Created {len(chunks)} chunks in {chunk_time:.2f}s")
            
            return chunks
            
        except Exception as e:
            print(f"[Chunker] Error: {e}")
            # Fallback to time-based chunking
            return AudioChunker._chunk_by_time(audio_path, max_duration)
    
    @staticmethod
    def _detect_silence(audio_path: str) -> List[float]:
        """Detect silence points in audio"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', f'silencedetect=noise={SILENCE_THRESHOLD}dB:d=0.5',
                '-f', 'null', '-'
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Parse silence_end timestamps
            silence_points = []
            for line in result.stderr.split('\n'):
                if 'silence_end' in line:
                    try:
                        timestamp = float(line.split('silence_end:')[1].split('|')[0].strip())
                        silence_points.append(timestamp)
                    except:
                        pass
            
            return sorted(silence_points)
            
        except Exception as e:
            if VERBOSE:
                print(f"[Chunker] Silence detection failed: {e}")
            return []
    
    @staticmethod
    def _create_chunks_from_silence(
        audio_path: str, 
        silence_points: List[float], 
        max_duration: int
    ) -> List[Dict]:
        """Create chunks using silence points as boundaries"""
        chunks = []
        start_time = 0
        
        base_name = os.path.splitext(audio_path)[0]
        output_dir = os.path.dirname(audio_path)
        
        for i, silence_point in enumerate(silence_points):
            chunk_duration = silence_point - start_time
            
            # If chunk is long enough and under max duration
            if MIN_CHUNK_DURATION <= chunk_duration <= max_duration:
                chunk_path = f"{base_name}_chunk_{i:03d}.wav"
                chunks.append({
                    "start": start_time,
                    "end": silence_point,
                    "duration": chunk_duration,
                    "chunk_path": chunk_path,
                    "index": i
                })
                start_time = silence_point
            
            # If chunk is too long, split it further
            elif chunk_duration > max_duration:
                # Split into smaller time-based chunks
                current = start_time
                while current < silence_point:
                    end = min(current + max_duration, silence_point)
                    chunk_path = f"{base_name}_chunk_{len(chunks):03d}.wav"
                    chunks.append({
                        "start": current,
                        "end": end,
                        "duration": end - current,
                        "chunk_path": chunk_path,
                        "index": len(chunks)
                    })
                    current = end
                start_time = silence_point
        
        # Add final chunk if needed
        duration = AudioProfiler._get_duration(audio_path)
        if duration - start_time >= MIN_CHUNK_DURATION:
            chunk_path = f"{base_name}_chunk_{len(chunks):03d}.wav"
            chunks.append({
                "start": start_time,
                "end": duration,
                "duration": duration - start_time,
                "chunk_path": chunk_path,
                "index": len(chunks)
            })
        
        return chunks
    
    @staticmethod
    def _chunk_by_time(audio_path: str, max_duration: int) -> List[Dict]:
        """Fallback: chunk by fixed time intervals"""
        duration = AudioProfiler._get_duration(audio_path)
        chunks = []
        
        base_name = os.path.splitext(audio_path)[0]
        
        current = 0
        index = 0
        while current < duration:
            end = min(current + max_duration, duration)
            chunk_path = f"{base_name}_chunk_{index:03d}.wav"
            chunks.append({
                "start": current,
                "end": end,
                "duration": end - current,
                "chunk_path": chunk_path,
                "index": index
            })
            current = end
            index += 1
        
        return chunks
    
    @staticmethod
    def extract_chunk(audio_path: str, chunk_info: Dict) -> str:
        """Extract a single chunk to file"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_path,
                '-ss', str(chunk_info['start']),
                '-t', str(chunk_info['duration']),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                chunk_info['chunk_path']
            ]
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            return chunk_info['chunk_path']
        except Exception as e:
            if VERBOSE:
                print(f"[Chunker] Extract failed for chunk {chunk_info['index']}: {e}")
            return None


class SmartTranscriber:
    """
    Intelligent transcriber with adaptive strategies
    Phase 1: Profile audio
    Phase 2: Detect language once
    Phase 3: Choose optimal model
    Phase 4: Process (direct/chunked/streaming)
    Phase 5: Clean and merge
    """
    
    def __init__(self):
        self.models = {}  # Cache loaded models
        self.device = self._detect_device()
        self.max_workers = CPU_WORKERS if self.device == "cpu" else GPU_WORKERS
        print(f"[Transcriber] Device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detect optimal device"""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except:
            return "cpu"
    
    def _select_model_name(self, strategy: str, language: Optional[str]) -> str:
        """Select model based on language and device"""
        if language:
            lang = language.lower()
            if lang in INDIAN_LANGS:
                return "tiny" if self.device == "cpu" else "small"
            return "small" if self.device == "cpu" else "base"
        return MODELS.get(strategy, "base")

    def _get_model(self, strategy: str, language: Optional[str] = None):
        """Get or load model for strategy"""
        model_name = self._select_model_name(strategy, language)
        
        if model_name in self.models:
            if VERBOSE:
                print(f"[Transcriber] Using cached model: {model_name}")
            return self.models[model_name]
        
        print(f"[Transcriber] Loading model: {model_name} for {strategy} strategy")
        
        if _HAS_FASTER and WhisperModel is not None:
            compute_type = "int8"
            model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=compute_type,
                num_workers=4 if self.device == "cpu" else 1
            )
            backend = "faster_whisper"
        else:
            if whisper is None:
                raise ImportError("No Whisper installation found")
            model = whisper.load_model(model_name, device=self.device)
            backend = "openai_whisper"
        
        self.models[model_name] = (model, backend)
        print(f"[Transcriber] Loaded {backend} model: {model_name}")
        
        return model, backend
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        """
        Main transcription pipeline with smart strategies
        """
        total_start = time.time()
        
        print("="*70)
        print("🎙️  Smart Transcriber v2 - Starting")
        print("="*70)
        
        # PHASE 1: Profile Audio
        print("\n[Phase 1] Profiling audio...")
        profile = AudioProfiler.profile(audio_path)
        
        strategy = profile['strategy']
        duration = profile['duration']
        
        print(f"✓ Strategy: {strategy}")
        print(f"✓ Duration: {duration:.1f}s ({duration/60:.1f}m)")
        
        # PHASE 2: Detect Language (probe first 30 seconds)
        print("\n[Phase 2] Detecting language...")
        detected_lang = self._detect_language(audio_path, language)
        print(f"✓ Language: {detected_lang}")
        
        # PHASE 3: Choose Model
        print("\n[Phase 3] Loading model...")
        model_name = self._select_model_name(strategy, detected_lang)
        model, backend = self._get_model(strategy, detected_lang)
        print(f"✓ Model: {model_name} ({backend})")
        
        # PHASE 4: Process based on strategy
        print(f"\n[Phase 4] Transcribing ({strategy} mode)...")
        
        if strategy == "direct":
            result = self._transcribe_direct(model, backend, audio_path, detected_lang)
        elif strategy == "chunked":
            result = self._transcribe_chunked(model, backend, audio_path, detected_lang)
        else:  # streaming
            result = self._transcribe_streaming(model, backend, audio_path, detected_lang)
        
        # PHASE 5: Post-process
        print("\n[Phase 5] Post-processing...")
        result = self._post_process(result)
        
        total_time = time.time() - total_start
        speed_ratio = duration / total_time if duration > 0 else 0
        
        print("\n" + "="*70)
        print("✅ Transcription Complete")
        print("="*70)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Speed: {speed_ratio:.2f}x realtime")
        print(f"Text length: {len(result['text'])} characters")
        print(f"Segments: {len(result['segments'])}")
        print("="*70)
        
        result.update({
            "profile": profile,
            "strategy": strategy,
            "language": detected_lang,
            "total_time": total_time,
            "speed_ratio": speed_ratio,
            "backend": backend
        })
        
        return result
    
    def _detect_language(self, audio_path: str, forced_lang: Optional[str]) -> str:
        """Detect language from first 30 seconds"""
        if forced_lang:
            return forced_lang
        
        try:
            # Create a 30-second probe file
            probe_path = audio_path.replace('.wav', '_probe.wav')
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-t', '30',
                '-acodec', 'pcm_s16le',
                probe_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=15, check=True)
            
            # Quick transcription for language detection
            model, backend = self._get_model("direct")
            
            if backend == "faster_whisper":
                segments, info = model.transcribe(
                    probe_path,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False
                )
                # Consume first segment
                next(segments, None)
                lang = getattr(info, 'language', 'en')
            else:
                result = model.transcribe(
                    probe_path,
                    task="transcribe",
                    beam_size=1,
                    condition_on_previous_text=False
                )
                lang = result.get('language', 'en')
            
            # Clean up probe file
            try:
                os.remove(probe_path)
            except:
                pass
            
            return lang
            
        except Exception as e:
            if VERBOSE:
                print(f"[Transcriber] Language detection failed: {e}")
            return "en"
    
    def _transcribe_direct(self, model, backend: str, audio_path: str, language: str) -> Dict:
        """Direct transcription (no chunking)"""
        start = time.time()
        
        if backend == "faster_whisper":
            segments, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )
            
            text_parts = []
            segment_list = []
            
            for seg in segments:
                text_parts.append(seg.text)
                segment_list.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                })
            
            full_text = " ".join(text_parts).strip()
            
        else:  # openai_whisper
            result = model.transcribe(
                audio_path,
                language=language,
                beam_size=1,
                condition_on_previous_text=False
            )
            full_text = result.get("text", "").strip()
            segment_list = [
                {
                    "start": s.get("start", 0),
                    "end": s.get("end", 0),
                    "text": s.get("text", "").strip()
                }
                for s in result.get("segments", [])
            ]
        
        process_time = time.time() - start
        print(f"✓ Transcribed in {process_time:.1f}s")
        
        return {
            "text": full_text,
            "segments": segment_list,
            "process_time": process_time
        }
    
    def _transcribe_chunked(self, model, backend: str, audio_path: str, language: str) -> Dict:
        """Chunked transcription with parallel processing"""
        start = time.time()
        
        # Create chunks
        chunks = AudioChunker.chunk_by_silence(audio_path)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Extract chunks to files
        print("✓ Extracting chunks...")
        for chunk in chunks:
            AudioChunker.extract_chunk(audio_path, chunk)
        
        # Transcribe chunks in parallel
        print(f"✓ Transcribing {len(chunks)} chunks in parallel...")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._transcribe_chunk,
                    model, backend, chunk, language
                ): chunk
                for chunk in chunks
            }
            
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  ✓ Chunk {chunk['index'] + 1}/{len(chunks)}")
                except Exception as e:
                    print(f"  ✗ Chunk {chunk['index']} failed: {e}")
        
        # Sort by index and merge
        results.sort(key=lambda x: x['index'])
        
        full_text = " ".join(r['text'] for r in results)
        all_segments = []
        for r in results:
            all_segments.extend(r['segments'])
        
        # Cleanup chunk files
        for chunk in chunks:
            try:
                os.remove(chunk['chunk_path'])
            except:
                pass
        
        process_time = time.time() - start
        print(f"✓ All chunks processed in {process_time:.1f}s")
        
        return {
            "text": full_text,
            "segments": all_segments,
            "process_time": process_time,
            "chunks_processed": len(results)
        }
    
    def _transcribe_chunk(self, model, backend: str, chunk: Dict, language: str) -> Dict:
        """Transcribe a single chunk"""
        chunk_path = chunk['chunk_path']
        
        if backend == "faster_whisper":
            segments, _ = model.transcribe(
                chunk_path,
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )
            
            text_parts = []
            segment_list = []
            
            for seg in segments:
                # Adjust timestamps relative to chunk start
                adjusted_start = chunk['start'] + seg.start
                adjusted_end = chunk['start'] + seg.end
                
                text_parts.append(seg.text)
                segment_list.append({
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": seg.text.strip()
                })
            
            text = " ".join(text_parts).strip()
            
        else:  # openai_whisper
            result = model.transcribe(
                chunk_path,
                language=language,
                beam_size=1,
                condition_on_previous_text=False
            )
            text = result.get("text", "").strip()
            
            segment_list = []
            for s in result.get("segments", []):
                adjusted_start = chunk['start'] + s.get("start", 0)
                adjusted_end = chunk['start'] + s.get("end", 0)
                segment_list.append({
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "text": s.get("text", "").strip()
                })
        
        return {
            "index": chunk['index'],
            "text": text,
            "segments": segment_list
        }
    
    def _transcribe_streaming(self, model, backend: str, audio_path: str, language: str) -> Dict:
        """Streaming transcription (same as chunked for now)"""
        # For very long videos, use chunked approach with smaller model
        return self._transcribe_chunked(model, backend, audio_path, language)
    
    def _post_process(self, result: Dict) -> Dict:
        """Clean up transcription results"""
        text = result['text']
        segments = result['segments']
        
        # Remove repeated lines
        seen_text = set()
        cleaned_segments = []
        
        for seg in segments:
            seg_text = seg['text'].strip()
            if seg_text and seg_text not in seen_text:
                seen_text.add(seg_text)
                cleaned_segments.append(seg)
        
        cleaned_text = " ".join(s['text'] for s in cleaned_segments)
        
        # Normalize whitespace
        cleaned_text = " ".join(cleaned_text.split())
        
        result['text'] = cleaned_text
        result['segments'] = cleaned_segments
        
        return result


# ========== PUBLIC API ==========

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict:
    """
    Main public API for transcription
    Returns structured transcript optimized for summarization
    """
    transcriber = SmartTranscriber()
    return transcriber.transcribe(audio_path, language)


# ========== TEST ==========

if __name__ == "__main__":
    print("Smart Transcriber v2 - Test Mode")
    print("="*70)
    print("\nEnvironment Variables:")
    print(f"  PERFORMANCE_MODE: {PERFORMANCE_MODE}")
    print(f"  TRANSCRIBE_WORKERS: {MAX_WORKERS}")
    print(f"  SILENCE_THRESHOLD: {SILENCE_THRESHOLD}dB")
    print(f"  VERBOSE: {VERBOSE}")
    print("="*70)
    
    # Test profiler
    print("\n[Test] Audio Profiler")
    print("  profile(audio_path) -> {duration, strategy, ...}")
    
    print("\n[Test] Audio Chunker")
    print("  chunk_by_silence(audio_path) -> [chunks]")
    
    print("\n[Test] Smart Transcriber")
    print("  transcribe(audio_path) -> {text, segments, ...}")
    
    print("\n✅ All modules loaded successfully!")
    print("\nUsage:")
    print("  from smart_transcriber import transcribe_audio")
    print("  result = transcribe_audio('video.wav')")
