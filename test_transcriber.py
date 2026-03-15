print("Testing faster-whisper installation...")

try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper is installed!")
    print(f"   WhisperModel: {WhisperModel}")
except Exception as e:
    print(f"❌ faster-whisper import failed: {e}")

print("\nTesting fast_transcriber.py...")
try:
    from fast_transcriber import Transcriber
    t = Transcriber(model_size='base')
    print(f"✅ Transcriber loaded")
    print(f"   Backend: {t.backend}")
    print(f"   Device: {t.device}")
except Exception as e:
    print(f"❌ fast_transcriber failed: {e}")
    import traceback
    traceback.print_exc()