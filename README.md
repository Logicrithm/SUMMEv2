# VidSummarize: The "Antigravity" Multilingual Video Summarizer 🚀

VidSummarize is an advanced, production-grade Flask web application designed to instantly transcribe and summarize videos, with a strong focus on **Indian regional languages** and extracting meaning from extremely long content (1 hour+).

By introducing the **Antigravity Architecture**, the system "floats above the weight" of standard transcription, bypassing massive downloads wherever possible, and intelligently scaling logic for ultra-long videos.

---

## 🌟 Key Features

### 1. The "Antigravity" Backend Architecture
A completely re-engineered backend designed to process massive videos in seconds rather than hours:
- **Phase 1: Caption-First Extraction** (`video_processor.py`)
  - Instead of automatically downloading 5GB+ video files, the system first probes YouTube metadata.
  - If subtitles or automatic captions exist (including regional codes like `hi` or `hi-IN`), it downloads the raw `.vtt`/`.srt` files directly.
  - **Result:** Hours of video processed in milliseconds.
- **Phase 2: Listen Briefly & Batch Streaming** (`smart_transcriber.py`)
  - If no captions exist, the system extracts the audio.
  - **Audio Profiler:** Samples the audio duration and loudness. If the file is absolute silence, it skips processing.
  - **True Streaming:** Extremely long audio is chunked intelligently based on **silence detection** (via `ffmpeg`), and processed in small batches to maintain a tiny memory footprint.
- **Phase 3: Hierarchical Summarization** (`summarizer.py`)
  - Prevents token-limit crashes for huge transcripts.
  - Uses a **Two-Pass Recursive Strategy**: Splits long transcripts into `~1200` character chunks -> summarizes each chunk -> summarizes the combined summaries into a final coherent output.

### 2. Deep Multilingual Support
Native support for complex Indian languages alongside English:
- Full Unicode support across the entire pipeline.
- Handles Hindi (`hi`), Tamil (`ta`), Telugu (`te`), Bengali (`bn`), Malayalam (`ml`), Kannada (`kn`), Marathi (`mr`), Gujarati (`gu`), Punjabi (`pa`), and Urdu (`ur`).
- Adaptive model selection: Automatically upgrades Whisper to `base` or `small` parameters for Indian languages to ensure higher accuracy.

### 3. Vibrant User Interface
A sleek, modern frontend designed to wow the user at first glance:
- Clean, responsive glassmorphism aesthetic.
- Interactive progress steps during the processing pipeline (Uploading -> Extracting -> Transcribing -> Summarizing).
- Extensible CSS variables (`variables.css`) allowing instant aesthetic shifts (e.g., Deep Sea, Frost Light, Aurora).

### 4. Professional Exporting
- Export the final transcript and AI summary directly to **TXT** or **PDF** (`pdf_generator.py`).
- Outputs map metadata accurately, including detected language, word count, speed ratio, and original video duration.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.11+, Flask
- **Audio Extraction & Metadata:** `yt-dlp`, FFmpeg
- **Transcription (AI):** `faster-whisper` (optimized for fast CPU/GPU inference), with OpenAI `whisper` fallback.
- **Summarization (AI):** HuggingFace `pipeline`, `google/mt5-small` (multilingual T5)
- **Frontend:** HTML5, Vanilla Deep CSS (No Tailwind dependency), JavaScript (Fetch APIs, SSE for progress)

---

## 📂 Project Structure

```text
VidSummarize/
├── app.py                     # Main Flask Application & Job Orchestrator
├── video_processor.py         # YouTube fetching, subtitle processing, FFmpeg
├── smart_transcriber.py       # Audio silence chunking, whisper inference
├── summarizer.py              # Recursive/Hierarchical mT5 summarization
├── pdf_generator.py           # Generation of cleanly formatted PDFs
├── templates/
│   ├── home.html              # Main upload/URL input page
│   ├── processing.html        # Interactive loading & progress tracking
│   ├── results.html           # Outputs display and download actions
│   └── base.html              # Core HTML wrapper
├── static/
│   ├── css/
│   │   ├── variables.css      # Core color palettes and glass themes
│   │   ├── home.css           # UI styles for the index
│   │   ├── components.css     # Buttons, inputs, structural UI
│   │   ├── processing.css     # Animation steps
│   │   └── results.css        # Typography and cards
│   └── js/                    # Client-side logic for API polling
├── uploads/                   # Temporary directory for local videos
└── outputs/                   # Cached transcripts, summaries, and PDFs
```

---

## 🚀 How It Works internally (The YouTube Flow)

1. **User Submits URL** to `/api/youtube`.
2. **`app.py` Process Job:** Initiates an asynchronous worker thread.
3. **`video_processor.get_youtube_info()`:** Probes metadata.
4. If captions are present: **Extracts `vtt`/`srt` directly** -> Parses into raw text -> Skips to Step 7.
5. If no captions: **Downloads audio (`m4a`/`wav`)** via `yt-dlp`.
6. **`smart_transcriber.transcribe()`:**
   - Detects language in the first 30 seconds.
   - Splits audio at silent points using `ffmpeg silencedetect`.
   - Feeds chunks concurrently to `faster-whisper`.
7. **`summarizer.summarize()`:** Takes final transcript, dynamically adjusts chunks, and prompts mT5.
8. **Outputs Saved:** `.txt` and `.pdf` files are generated.
9. **UI Updates:** The client `processing.html` polls `/api/status` until `completed`, then navigates to `results.html`.

---

## 💡 Status & Future Improvements
- **Currently Implemented:** Antigravity backend, multilingual processing, basic UI/UX structural revamps.
- **Next Steps:** Polish the `variables.css` aesthetic themes and add an interactive theme switcher.