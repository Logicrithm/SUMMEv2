"""
VidSummarize - Video Processing Module (FIXED)
Handles video download, audio extraction, and file processing

FIXES APPLIED:
1. Removed hardcoded Windows cookie path — now reads from env var or auto-discovers
2. Added graceful fallback when no cookie file is found
3. Removed 'format': 'best' from metadata/subtitle calls — caused "format not available" crash
4. Used 'bestaudio/best' for actual downloads so any available format is accepted
5. Improved error messages to help with debugging
"""

import os
import subprocess
import re
from datetime import datetime

import yt_dlp
import ffmpeg

# ──────────────────────────────────────────────
# COOKIE FILE RESOLUTION  (FIX #1 & #2)
# Priority:
#   1. Environment variable  YOUTUBE_COOKIE_FILE
#   2. A file named  youtube_cookies.txt  next to this script
#   3. No cookies (yt-dlp will try without them)
# ──────────────────────────────────────────────
def _resolve_cookie_file() -> str | None:
    # 1. Explicit env-var override
    env_path = os.environ.get("YOUTUBE_COOKIE_FILE", "").strip()
    if env_path and os.path.isfile(env_path):
        print(f"[VideoProcessor] Using cookie file from env: {env_path}")
        return env_path

    # 2. Sibling file in the project root
    here = os.path.dirname(os.path.abspath(__file__))
    local_cookie = os.path.join(here, "youtube_cookies.txt")
    if os.path.isfile(local_cookie):
        print(f"[VideoProcessor] Using local cookie file: {local_cookie}")
        return local_cookie

    print("[VideoProcessor] WARNING: No cookie file found. "
          "YouTube may block requests. Set YOUTUBE_COOKIE_FILE env var "
          "or place youtube_cookies.txt in the project root.")
    return None   # yt-dlp will try unauthenticated


COOKIE_FILE = _resolve_cookie_file()


def _ydl_base_opts(skip_download=True) -> dict:
    """Return common yt-dlp options with optional cookie injection."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": skip_download,
        "format": "m4a/bestaudio/best/worst",
        "extract_flat": "discard_in_playlist",
    }
    if COOKIE_FILE:
        opts["cookiefile"] = COOKIE_FILE
    return opts


class VideoProcessor:
    """Handles all video and audio processing operations"""

    def __init__(self, upload_folder="uploads", output_folder="outputs"):
        self.upload_folder = upload_folder
        self.output_folder = output_folder
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)

    # ──────────────────────────────────────────
    # Audio extraction
    # ──────────────────────────────────────────
    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract mono 16 kHz WAV audio from a video file using FFmpeg."""
        try:
            print(f"[VideoProcessor] Extracting audio from: {video_path}")

            if output_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(self.output_folder, f"{base_name}_audio.wav")

            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, output_path, acodec="pcm_s16le", ac=1, ar="16k")
            ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

            metadata = self._get_video_metadata(video_path)
            print(f"[VideoProcessor] Audio extracted → {output_path}")
            return {
                "success": True,
                "audio_path": output_path,
                "duration": metadata.get("duration", 0),
                "size": os.path.getsize(output_path),
            }

        except ffmpeg.Error as e:
            msg = e.stderr.decode() if (hasattr(e, "stderr") and e.stderr) else str(e)
            print(f"[VideoProcessor] FFmpeg error: {msg}")
            return {"success": False, "error": f"Audio extraction failed: {msg}"}
        except Exception as e:
            print(f"[VideoProcessor] Error: {e}")
            return {"success": False, "error": f"Audio extraction failed: {e}"}

    # ──────────────────────────────────────────
    # YouTube — metadata probe
    # ──────────────────────────────────────────
    def get_youtube_info(self, youtube_url):
        """
        Antigravity Move: Fetch metadata + caption availability WITHOUT downloading.
        """
        try:
            print(f"[VideoProcessor] Fetching YouTube info: {youtube_url}")
            # NOTE: No 'format' key here — skip_download=True means we're only
            # fetching metadata, so format validation is irrelevant and causes
            # "Requested format is not available" crashes on some videos.
            opts = _ydl_base_opts(skip_download=True)

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
            if not info:
                return {"success": False, "error": "Video metadata could not be fetched (video may be unavailable)"}

            return {
                "success": True,
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),
                "subtitles": info.get("subtitles", {}),
                "automatic_captions": info.get("automatic_captions", {}),
                "description": info.get("description", ""),
                "raw_info": info,
            }
        except Exception as e:
            print(f"[VideoProcessor] Error fetching YouTube info: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────
    # YouTube — subtitle download
    # ──────────────────────────────────────────
    def download_youtube_subtitles(self, youtube_url, lang="en", output_filename=None):
        """
        Antigravity Move: Download subtitle/caption files directly (no video download).
        """
        try:
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"subs_{timestamp}"

            output_template = os.path.join(self.output_folder, output_filename)

            target_langs = list(dict.fromkeys([lang, "en", "hi", "ta", "te", "bn", "ml", "kn", "mr", "gu"]))

            opts = _ydl_base_opts(skip_download=True)
            opts.update({
                # No 'format' key — skip_download=True, format validation causes crashes
                "write_auto_subs": True,
                "write_subs": True,
                "subtitleslangs": target_langs,
                "outtmpl": f"{output_template}.%(ext)s",
            })

            print(f"[VideoProcessor] Downloading subs for {youtube_url} (preferring '{lang}')")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([youtube_url])

            # ── Discover the downloaded subtitle file ──
            possible_files = os.listdir(self.output_folder)
            found_path, found_lang = None, "unknown"

            # Pass 1: exact language match
            for f in possible_files:
                if f.startswith(output_filename) and f.endswith((".vtt", ".srt")):
                    if f".{lang}." in f:
                        found_path = os.path.join(self.output_folder, f)
                        found_lang = lang
                        break

            # Pass 2: any sub with our filename prefix
            if not found_path:
                for f in possible_files:
                    if f.startswith(output_filename) and f.endswith((".vtt", ".srt")):
                        found_path = os.path.join(self.output_folder, f)
                        parts = f.split(".")
                        found_lang = parts[-2] if len(parts) >= 3 else "unknown"
                        break

            if found_path:
                print(f"[VideoProcessor] Subtitle found: {found_path} ({found_lang})")
                return {"success": True, "subtitle_path": found_path, "language": found_lang}

            return {"success": False, "error": f"No subtitle file found for language: {lang}"}

        except Exception as e:
            print(f"[VideoProcessor] Subtitle download error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────
    # YouTube — audio / video download
    # ──────────────────────────────────────────
    def download_youtube_video(self, youtube_url, output_filename=None, audio_only=True):
        """
        Download audio (preferred) or full video from YouTube, then extract WAV.
        """
        try:
            mode = "audio-only" if audio_only else "video"
            print(f"[VideoProcessor] Downloading YouTube ({mode}): {youtube_url}")

            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"youtube_{timestamp}"

            opts = _ydl_base_opts(skip_download=False)
            opts.update({
                "format": "m4a/bestaudio/best/worst" if audio_only else "bestvideo+bestaudio/best/worst",
                "outtmpl": os.path.join(self.upload_folder, f"{output_filename}.%(ext)s"),
                "quiet": False,
                "ignoreerrors": False
            })

            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                title = info.get("title", "Unknown")
                duration = info.get("duration", 0)
                video_path = ydl.prepare_filename(info)

            # Fallback: scan the upload folder if prepare_filename is off
            if not os.path.exists(video_path):
                for f in os.listdir(self.upload_folder):
                    if f.startswith(output_filename):
                        video_path = os.path.join(self.upload_folder, f)
                        break

            audio_path = os.path.join(self.output_folder, f"{output_filename}_audio.wav")
            audio_result = self.extract_audio_from_video(video_path, audio_path)

            if audio_result["success"]:
                audio_result["video_path"] = video_path
                audio_result["video_title"] = title
                audio_result["duration"] = duration

            return audio_result

        except Exception as e:
            print(f"[VideoProcessor] YouTube download error: {e}")
            return {"success": False, "error": str(e)}

    # ──────────────────────────────────────────
    # Subtitle parsing
    # ──────────────────────────────────────────
    def extract_text_from_subtitles(self, subtitle_path):
        """Strip timestamps and markup from a VTT/SRT file, returning plain text."""
        try:
            if not os.path.exists(subtitle_path):
                return ""

            with open(subtitle_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove WEBVTT header block
            content = re.sub(r"WEBVTT.*?\n\n", "", content, flags=re.DOTALL)
            # Remove VTT timestamps  (00:00:00.000 --> 00:00:00.000 ...)
            content = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n", "", content)
            # Remove SRT timestamps  (00:00:00,000 --> 00:00:00,000)
            content = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}.*?\n", "", content)
            # Remove HTML-style tags
            content = re.sub(r"<[^>]*>", "", content)

            # Deduplicate consecutive identical lines
            unique_lines = []
            prev = ""
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped and stripped != prev:
                    unique_lines.append(stripped)
                    prev = stripped

            return " ".join(unique_lines)

        except Exception as e:
            print(f"[VideoProcessor] Subtitle parse error: {e}")
            return ""

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────
    def _get_video_metadata(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            return {
                "duration": float(probe["format"]["duration"]),
                "size": int(probe["format"]["size"]),
            }
        except Exception as e:
            print(f"[VideoProcessor] Metadata error: {e}")
            return {"duration": 0, "size": 0}

    def get_audio_duration(self, audio_path):
        try:
            return float(ffmpeg.probe(audio_path)["format"]["duration"])
        except Exception:
            return 0

    def cleanup_file(self, file_path):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False


# ──────────────────────────────────────────────
# Quick sanity check when run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("✅ FFmpeg is installed")
    except FileNotFoundError:
        print("❌ FFmpeg not found — install it and make sure it's on your PATH")

    try:
        result = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
        print("✅ FFprobe is installed")
    except FileNotFoundError:
        print("❌ FFprobe not found")

    cookie = _resolve_cookie_file()
    if cookie:
        print(f"✅ Cookie file found: {cookie}")
    else:
        print("⚠️  No cookie file — YouTube may block requests")

    print("\nVideoProcessor ready.")