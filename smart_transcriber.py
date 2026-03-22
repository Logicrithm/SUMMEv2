"""
VidSummarize - Smart Transcriber v2
ANTIGRAVITY UPDATED: Memory-efficient streaming and early sampling
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
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import json


# ========== CONFIGURATION ==========

MODELS = {
    "direct": "tiny",
    "chunked": "base",
    "streaming": "small"
}

INDIAN_LANGS = {"hi", "ta", "te", "bn", "ml", "kn", "mr", "gu", "pa", "ur"}
SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "-40"))
MIN_CHUNK_DURATION = 10
MAX_CHUNK_DURATION = 180
VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"


class AudioProfiler:
    """Analyzes audio to determine optimal processing strategy"""
    
    @staticmethod
    def profile(audio_path: str) -> Dict:
        start = time.time()
        try:
            duration = AudioProfiler._get_duration(audio_path)
            loudness_stats = AudioProfiler._get_loudness(audio_path)
            
            if duration < 600:
                strategy = "direct"
            elif duration < 2400:
                strategy = "chunked"
            else:
                strategy = "streaming"
            
            return {
                "duration": duration,
                "avg_loudness": loudness_stats.get("mean_volume", -20),
                "strategy": strategy,
                "profile_time": time.time() - start
            }
        except Exception as e:
            return {"duration": 0, "avg_loudness": -20, "strategy": "direct", "error": str(e)}

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        return float(subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or 0)

    @staticmethod
    def _get_loudness(audio_path: str) -> Dict:
        cmd = ['ffmpeg', '-i', audio_path, '-af', 'volumedetect', '-f', 'null', '-']
        stderr = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stderr
        mean_volume = -20
        for line in stderr.split('\n'):
            if 'mean_volume' in line:
                try: mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except: pass
        return {"mean_volume": mean_volume}


class AudioChunker:
    """Smart audio chunking based on silence detection"""
    
    @staticmethod
    def chunk_by_silence(audio_path: str, max_duration: int = MAX_CHUNK_DURATION) -> List[Dict]:
        try:
            silence_points = AudioChunker._detect_silence(audio_path)
            if not silence_points:
                return AudioChunker._chunk_by_time(audio_path, max_duration)
            return AudioChunker._create_chunks_from_silence(audio_path, silence_points, max_duration)
        except:
            return AudioChunker._chunk_by_time(audio_path, max_duration)

    @staticmethod
    def _detect_silence(audio_path: str) -> List[float]:
        cmd = ['ffmpeg', '-i', audio_path, '-af', f'silencedetect=noise={SILENCE_THRESHOLD}dB:d=0.5', '-f', 'null', '-']
        stderr = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stderr
        points = []
        for line in stderr.split('\n'):
            if 'silence_end' in line:
                try: points.append(float(line.split('silence_end:')[1].split('|')[0].strip()))
                except: pass
        return sorted(points)

    @staticmethod
    def _create_chunks_from_silence(audio_path, points, max_duration):
        chunks = []
        start = 0
        base = os.path.splitext(audio_path)[0]
        for i, point in enumerate(points):
            dur = point - start
            if MIN_CHUNK_DURATION <= dur <= max_duration:
                chunks.append({"start": start, "end": point, "duration": dur, "chunk_path": f"{base}_c{i:03d}.wav", "index": i})
                start = point
            elif dur > max_duration:
                curr = start
                while curr < point:
                    end = min(curr + max_duration, point)
                    chunks.append({"start": curr, "end": end, "duration": end - curr, "chunk_path": f"{base}_c{len(chunks):03d}.wav", "index": len(chunks)})
                    curr = end
                start = point
        # Final chunk
        dur_total = AudioProfiler._get_duration(audio_path)
        if dur_total - start >= 1:
            chunks.append({"start": start, "end": dur_total, "duration": dur_total - start, "chunk_path": f"{base}_c{len(chunks):03d}.wav", "index": len(chunks)})
        return chunks

    @staticmethod
    def _chunk_by_time(audio_path, max_duration):
        dur = AudioProfiler._get_duration(audio_path)
        chunks = []
        base = os.path.splitext(audio_path)[0]
        curr = 0
        idx = 0
        while curr < dur:
            end = min(curr + max_duration, dur)
            chunks.append({"start": curr, "end": end, "duration": end - curr, "chunk_path": f"{base}_c{idx:03d}.wav", "index": idx})
            curr = end
            idx += 1
        return chunks

    @staticmethod
    def extract_chunk(audio_path, chunk):
        try:
            cmd = ['ffmpeg', '-y', '-i', audio_path, '-ss', str(chunk['start']), '-t', str(chunk['duration']), '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', chunk['chunk_path']]
            subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        except Exception as e:
            print(f"[SmartTranscriber] FFmpeg extract failed for chunk {chunk['index']}: {e}")
            # Keep an empty trace so it doesn't crash
            with open(chunk['chunk_path'], 'wb') as f:
                f.write(b"")


class SmartTranscriber:
    """Intelligent transcriber with adaptive strategies"""
    
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_workers = 2 if self.device == "cpu" else 4

    def _get_model(self, strategy: str, lang: Optional[str] = None):
        model_name = "small" if (lang in INDIAN_LANGS or strategy == "streaming") else "base"
        if model_name in self.models: return self.models[model_name]
        
        if _HAS_FASTER and WhisperModel:
            try:
                model = WhisperModel(model_name, device=self.device, compute_type="int8")
            except Exception as e:
                # Fallback to float32 or default if int8 compute is not supported
                print(f"[SmartTranscriber] int8 constraint failed on {self.device}: {e}. Retrying with float32.")
                try:
                    model = WhisperModel(model_name, device=self.device, compute_type="float32")
                except Exception as e2:
                    print(f"[SmartTranscriber] float32 failed, falling back to default compute_type")
                    model = WhisperModel(model_name, device=self.device, compute_type="default")
            backend = "faster_whisper"
        else:
            import whisper
            model = whisper.load_model(model_name, device=self.device)
            backend = "openai_whisper"
        
        self.models[model_name] = (model, backend)
        return model, backend

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict:
        start_time = time.time()
        print(f"[Transcriber] Starting: {audio_path}")
        
        profile = AudioProfiler.profile(audio_path)
        
        # Sampling Check: If incredibly quiet, exit early
        if profile['avg_loudness'] < -55:
            return {"text": "[Audio is too quiet to transcribe]", "segments": [], "strategy": "skipped_silence"}

        # Detection Phase
        detected_lang = self._detect_language(audio_path, language)
        strategy = profile['strategy']
        model, backend = self._get_model(strategy, detected_lang)
        
        print(f"[Transcriber] Strategy: {strategy}, Lang: {detected_lang}, Backend: {backend}")
        
        if strategy == "direct":
            result = self._transcribe_direct(model, backend, audio_path, detected_lang)
        elif strategy == "chunked":
            result = self._transcribe_chunked(model, backend, audio_path, detected_lang)
        else:
            result = self._transcribe_streaming(model, backend, audio_path, detected_lang)
            
        total_time = time.time() - start_time
        result.update({
            "language": detected_lang, "total_time": total_time,
            "speed_ratio": profile['duration'] / total_time if total_time > 0 else 0,
            "strategy": strategy, "profile": profile
        })
        return result

    def _detect_language(self, audio_path, forced_lang):
        if forced_lang: return forced_lang
        # Probe first 30s
        probe = audio_path.replace('.wav', '_probe.wav')
        subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-t', '30', '-acodec', 'pcm_s16le', probe], capture_output=True)
        try:
            model, backend = self._get_model("direct")
            if backend == "faster_whisper":
                _, info = model.transcribe(probe, beam_size=1)
                lang = info.language
            else:
                lang = model.transcribe(probe, beam_size=1).get('language', 'en')
            os.remove(probe)
            return lang
        except:
            return "en"

    def _transcribe_direct(self, model, backend, path, lang):
        try:
            if backend == "faster_whisper":
                segs, _ = model.transcribe(path, language=lang, beam_size=1, vad_filter=True)
                res = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segs]
                return {"text": " ".join(s['text'] for s in res), "segments": res}
            else:
                res = model.transcribe(path, language=lang)
                return {"text": res['text'], "segments": res['segments']}
        except Exception as e:
            print(f"[SmartTranscriber] Direct transcription failed on {path}: {e}")
            return {"text": "", "segments": []}

    def _transcribe_chunked(self, model, backend, path, lang):
        chunks = AudioChunker.chunk_by_silence(path)
        for c in chunks: AudioChunker.extract_chunk(path, c)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(self._transcribe_chunk, model, backend, c, lang): c for c in chunks}
            for f in as_completed(futures): results.append(f.result())
        
        results.sort(key=lambda x: x['index'])
        for c in chunks: 
            try: os.remove(c['chunk_path'])
            except: pass
            
        return {
            "text": " ".join(r['text'] for r in results),
            "segments": [s for r in results for s in r['segments']]
        }

    def _transcribe_streaming(self, model, backend, path, lang):
        """Streaming mode: Process batches to save memory and disk"""
        start = time.time()
        chunks = AudioChunker.chunk_by_silence(path)
        all_text = []
        all_segs = []
        batch_size = 5
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            for c in batch: AudioChunker.extract_chunk(path, c)
            
            batch_res = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
                futures = {exe.submit(self._transcribe_chunk, model, backend, c, lang): c for c in batch}
                for f in as_completed(futures): batch_res.append(f.result())
            
            batch_res.sort(key=lambda x: x['index'])
            for r in batch_res:
                all_text.append(r['text'])
                all_segs.extend(r['segments'])
            
            for c in batch:
                try: os.remove(c['chunk_path'])
                except: pass
        
        return {"text": " ".join(all_text), "segments": all_segs}

    def _transcribe_chunk(self, model, backend, chunk, lang):
        res = self._transcribe_direct(model, backend, chunk['chunk_path'], lang)
        # Adjust timestamps
        for s in res['segments']:
            s['start'] += chunk['start']
            s['end'] += chunk['start']
        return {"index": chunk['index'], "text": res['text'], "segments": res['segments']}


def transcribe_audio(audio_path: str, language: Optional[str] = None) -> Dict:
    return SmartTranscriber().transcribe(audio_path, language)

if __name__ == "__main__":
    print("VidSummarize Smart Transcriber Loaded.")
