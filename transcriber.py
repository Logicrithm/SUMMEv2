"""
VidSummarize - Main Application
Fixed to use fast_transcriber instead of old transcriber
"""

import os
import sys
from pathlib import Path

# Import the FAST transcriber (not the old one!)
from fast_transcriber import Transcriber
from video_processor import VideoProcessor
from summarizer import Summarizer


class VidSummarizeApp:
    def __init__(self, whisper_model="base", summarizer_model="facebook/bart-large-cnn"):
        print("[App] Initializing VidSummarize...")
        
        # Load Faster-Whisper (NOT OpenAI Whisper!)
        print("[App] Loading Faster-Whisper model...")
        self.transcriber = Transcriber(model_size=whisper_model)
        
        # Verify we're using the fast version
        if self.transcriber.backend != "faster_whisper":
            print("⚠️  WARNING: Not using faster-whisper!")
            print(f"   Current backend: {self.transcriber.backend}")
            print("   Expected: faster_whisper")
        else:
            print("✅ Faster-Whisper loaded successfully!")
        
        print("[App] Loading Summarizer...")
        self.summarizer = Summarizer(model_name=summarizer_model)
        
        self.video_processor = VideoProcessor()
        print("[App] Initialization complete\n")

    def process_video(self, video_path, output_dir="output", language=None):
        """Main pipeline: video -> transcript -> summary"""
        
        print("=" * 60)
        print("STARTING VIDEO PROCESSING PIPELINE")
        print("=" * 60)
        
        # Verify input
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        
        # Step 1: Extract audio
        print("\n[STEP 1/3] Extracting audio...")
        audio_path = os.path.join(output_dir, f"{video_name}_audio.wav")
        
        if not self.video_processor.extract_audio(video_path, audio_path):
            print("❌ Audio extraction failed")
            return None
        
        print(f"✅ Audio extracted: {audio_path}")
        
        # Step 2: Transcribe (with Faster-Whisper!)
        print("\n[STEP 2/3] Transcribing audio with Faster-Whisper...")
        print(f"Backend: {self.transcriber.backend}")
        
        transcript_result = self.transcriber.transcribe_with_timestamps(
            audio_path, 
            language=language
        )
        
        if not transcript_result["success"]:
            print(f"❌ Transcription failed: {transcript_result.get('error')}")
            return None
        
        # Save transcript
        transcript_path = os.path.join(output_dir, f"{video_name}_transcript.txt")
        self.transcriber.save_transcript(
            transcript_result["formatted_transcript"], 
            transcript_path
        )
        
        print(f"✅ Transcript saved: {transcript_path}")
        print(f"   Words: {transcript_result['word_count']}")
        print(f"   Language: {transcript_result['language']}")
        if transcript_result.get('duration'):
            print(f"   Duration: {transcript_result['duration']:.2f}s")
        
        # Step 3: Summarize
        print("\n[STEP 3/3] Generating summary...")
        
        summary_result = self.summarizer.summarize(
            transcript_result["text"],
            max_length=250,
            min_length=100
        )
        
        if not summary_result["success"]:
            print(f"❌ Summarization failed: {summary_result.get('error')}")
            return None
        
        # Save summary
        summary_path = os.path.join(output_dir, f"{video_name}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_result["summary"])
        
        print(f"✅ Summary saved: {summary_path}")
        print(f"   Summary length: {len(summary_result['summary'])} characters")
        
        # Final result
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Transcript: {transcript_path}")
        print(f"📝 Summary: {summary_path}")
        print("=" * 60 + "\n")
        
        return {
            "success": True,
            "transcript": transcript_result,
            "summary": summary_result,
            "files": {
                "audio": audio_path,
                "transcript": transcript_path,
                "summary": summary_path
            }
        }


def main():
    """Command line interface"""
    
    if len(sys.argv) < 2:
        print("Usage: python app.py <video_path> [output_dir] [language]")
        print("\nExample:")
        print("  python app.py video.mp4")
        print("  python app.py video.mp4 my_output")
        print("  python app.py video.mp4 my_output en")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    language = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Get model sizes from environment or use defaults
    whisper_model = os.getenv("WHISPER_MODEL", "base")
    summarizer_model = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
    
    print(f"\n🎬 VidSummarize - Fast Video Transcription & Summarization")
    print(f"📹 Video: {video_path}")
    print(f"📂 Output: {output_dir}")
    print(f"🎤 Whisper Model: {whisper_model}")
    print(f"📝 Summarizer: {summarizer_model}")
    if language:
        print(f"🌍 Language: {language}")
    print()
    
    # Initialize and run
    app = VidSummarizeApp(
        whisper_model=whisper_model,
        summarizer_model=summarizer_model
    )
    
    result = app.process_video(video_path, output_dir, language)
    
    if result and result["success"]:
        print("\n✅ SUCCESS!")
        print("\n📋 SUMMARY:")
        print("-" * 60)
        print(result["summary"]["summary"])
        print("-" * 60)
    else:
        print("\n❌ FAILED - Check logs above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()