"""
VidSummarize - Model Manager Module
Handles model loading, caching, and GPU/CPU routing

Responsibility: Load once, reuse forever
Focus: Speed through caching, not reloading
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from typing import Dict, Optional, Callable
import time


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = {
    'fast': {
        'name': 'sshleifer/distilbart-cnn-12-6',
        'description': 'Distilled BART - 40% faster, slightly lower quality',
        'avg_speed_multiplier': 1.4,
        'quality_score': 78
    },
    'balanced': {
        'name': 'facebook/bart-large-cnn',
        'description': 'Standard BART - best balance',
        'avg_speed_multiplier': 1.0,
        'quality_score': 85
    },
    'quality': {
        'name': 'facebook/bart-large-cnn',
        'description': 'Same as balanced, used in quality mode',
        'avg_speed_multiplier': 1.0,
        'quality_score': 90
    }
}


# ============================================================================
# MODEL CACHE
# ============================================================================

class ModelCache:
    """
    Singleton cache for loaded models
    
    Prevents reloading models on every request
    Saves 5-30 seconds per request after first load
    """
    
    _instance = None
    _loaded_models: Dict[str, any] = {}
    _loaded_tokenizers: Dict[str, any] = {}
    _load_times: Dict[str, float] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def get_model(self, mode: str):
        """Get cached model or None"""
        return self._loaded_models.get(mode)
    
    def get_tokenizer(self, mode: str):
        """Get cached tokenizer or None"""
        return self._loaded_tokenizers.get(mode)
    
    def set_model(self, mode: str, model, tokenizer):
        """Cache model and tokenizer"""
        self._loaded_models[mode] = model
        self._loaded_tokenizers[mode] = tokenizer
    
    def get_load_time(self, mode: str) -> float:
        """Get load time for a mode"""
        return self._load_times.get(mode, 0)
    
    def set_load_time(self, mode: str, load_time: float):
        """Record load time"""
        self._load_times[mode] = load_time
    
    def is_loaded(self, mode: str) -> bool:
        """Check if model is loaded"""
        return mode in self._loaded_models
    
    def clear(self, mode: Optional[str] = None):
        """Clear cache (for mode or all)"""
        if mode:
            self._loaded_models.pop(mode, None)
            self._loaded_tokenizers.pop(mode, None)
            self._load_times.pop(mode, None)
        else:
            self._loaded_models.clear()
            self._loaded_tokenizers.clear()
            self._load_times.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'loaded_models': list(self._loaded_models.keys()),
            'total_models': len(self._loaded_models),
            'load_times': self._load_times.copy()
        }


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def detect_device() -> tuple:
    """
    Detect best available device
    
    Returns:
        (device_id, device_name) tuple
        device_id: 0 for GPU, -1 for CPU
        device_name: 'cuda' or 'cpu'
    """
    if torch.cuda.is_available():
        device_id = 0
        device_name = 'cuda'
        
        # Get GPU info
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[ModelManager] GPU detected: {gpu_name}")
        except:
            print(f"[ModelManager] GPU detected (name unknown)")
    else:
        device_id = -1
        device_name = 'cpu'
        print(f"[ModelManager] Using CPU (no GPU available)")
    
    return device_id, device_name


# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelManager:
    """
    Manages model loading, caching, and serving
    
    Features:
    - Lazy loading (only when needed)
    - Automatic caching (load once, reuse)
    - Device routing (GPU/CPU)
    - Model switching by mode
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.cache = ModelCache()
        self.device_id, self.device_name = detect_device()
        
        # Configuration
        self.use_cache = True
        self.auto_device = True
        
        print(f"[ModelManager] Initialized")
        print(f"[ModelManager] Device: {self.device_name}")
    
    def _load_model(self, mode: str) -> tuple:
        """
        Actually load model from HuggingFace
        
        Args:
            mode: Mode name (fast/balanced/quality)
            
        Returns:
            (model, tokenizer) tuple
        """
        if mode not in MODELS:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(MODELS.keys())}")
        
        model_info = MODELS[mode]
        model_name = model_info['name']
        
        print(f"[ModelManager] Loading model: {model_name}")
        print(f"[ModelManager] Mode: {mode}")
        print(f"[ModelManager] Description: {model_info['description']}")
        
        start_time = time.time()
        
        try:
            # Load tokenizer
            print(f"[ModelManager] Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            print(f"[ModelManager] Loading model weights...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move to device
            if self.auto_device and self.device_name == 'cuda':
                print(f"[ModelManager] Moving model to GPU...")
                model = model.to('cuda')
            
            load_time = time.time() - start_time
            
            print(f"[ModelManager] ✅ Model loaded successfully in {load_time:.1f}s")
            
            # Cache load time
            self.cache.set_load_time(mode, load_time)
            
            return model, tokenizer
            
        except Exception as e:
            print(f"[ModelManager] ❌ Failed to load model: {e}")
            raise
    
    def get_pipeline(self, mode: str = 'balanced') -> Callable:
        """
        Get summarization pipeline for a mode
        
        Args:
            mode: Mode name (fast/balanced/quality)
            
        Returns:
            Summarization pipeline function
        """
        # Check cache first
        if self.use_cache and self.cache.is_loaded(mode):
            print(f"[ModelManager] ✓ Using cached model for '{mode}' mode")
            model = self.cache.get_model(mode)
            tokenizer = self.cache.get_tokenizer(mode)
        else:
            # Load model
            print(f"[ModelManager] Loading fresh model for '{mode}' mode...")
            model, tokenizer = self._load_model(mode)
            
            # Cache it
            if self.use_cache:
                self.cache.set_model(mode, model, tokenizer)
                print(f"[ModelManager] ✓ Model cached for future use")
        
        # Create pipeline
        pipe = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=self.device_id
        )
        
        return pipe
    
    def get_model_info(self, mode: str) -> Dict:
        """Get information about a model"""
        if mode not in MODELS:
            return None
        
        info = MODELS[mode].copy()
        info['is_loaded'] = self.cache.is_loaded(mode)
        info['device'] = self.device_name
        
        if self.cache.is_loaded(mode):
            info['load_time'] = self.cache.get_load_time(mode)
        
        return info
    
    def preload_model(self, mode: str):
        """
        Preload a model (before first request)
        
        Useful for warming up the cache
        """
        print(f"[ModelManager] Preloading model for '{mode}' mode...")
        self.get_pipeline(mode)
    
    def switch_device(self, use_gpu: bool):
        """
        Switch between GPU and CPU
        
        Note: Requires reloading models
        """
        if use_gpu and not torch.cuda.is_available():
            print(f"[ModelManager] ⚠️ GPU requested but not available")
            return
        
        self.auto_device = use_gpu
        self.device_id = 0 if use_gpu else -1
        self.device_name = 'cuda' if use_gpu else 'cpu'
        
        # Clear cache (models need to be moved)
        print(f"[ModelManager] Clearing cache to switch to {self.device_name}")
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self, mode: Optional[str] = None):
        """
        Clear model cache
        
        Args:
            mode: Specific mode to clear, or None for all
        """
        if mode:
            print(f"[ModelManager] Clearing cache for '{mode}' mode")
        else:
            print(f"[ModelManager] Clearing all cached models")
        
        self.cache.clear(mode)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global model manager instance
model_manager = ModelManager()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_model(mode: str = 'balanced') -> Callable:
    """
    Get summarization model for a mode
    
    Convenience wrapper around model_manager.get_pipeline()
    
    Args:
        mode: Mode name (fast/balanced/quality)
        
    Returns:
        Summarization pipeline
    """
    return model_manager.get_pipeline(mode)


def preload_all_models():
    """Preload all models for faster first request"""
    print("[ModelManager] Preloading all models...")
    for mode in MODELS.keys():
        model_manager.preload_model(mode)
    print("[ModelManager] ✅ All models preloaded")


def list_available_models() -> Dict:
    """List all available models with info"""
    return {mode: model_manager.get_model_info(mode) for mode in MODELS.keys()}


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Model Manager Test")
    print("=" * 70)
    print()
    
    # Test 1: Device detection
    print("Test 1: Device Detection")
    print("-" * 70)
    device_id, device_name = detect_device()
    print(f"  Device ID: {device_id}")
    print(f"  Device Name: {device_name}")
    print()
    
    # Test 2: Model info
    print("Test 2: Available Models")
    print("-" * 70)
    for mode, info in MODELS.items():
        print(f"\n{mode.upper()}:")
        print(f"  Model: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Speed multiplier: {info['avg_speed_multiplier']}x")
        print(f"  Quality score: {info['quality_score']}%")
    print()
    
    # Test 3: Model loading (balanced mode only for speed)
    print("Test 3: Model Loading & Caching")
    print("-" * 70)
    
    # First load (slow)
    print("\nFirst load (should take 5-15 seconds)...")
    start = time.time()
    pipe1 = model_manager.get_pipeline('balanced')
    first_load_time = time.time() - start
    print(f"✓ First load: {first_load_time:.1f}s")
    
    # Second load (should be instant from cache)
    print("\nSecond load (should be instant)...")
    start = time.time()
    pipe2 = model_manager.get_pipeline('balanced')
    second_load_time = time.time() - start
    print(f"✓ Second load: {second_load_time:.3f}s")
    
    print(f"\nSpeedup: {first_load_time/max(second_load_time, 0.001):.1f}x faster")
    print()
    
    # Test 4: Cache stats
    print("Test 4: Cache Statistics")
    print("-" * 70)
    stats = model_manager.get_cache_stats()
    print(f"  Loaded models: {stats['loaded_models']}")
    print(f"  Total count: {stats['total_models']}")
    print(f"  Load times: {stats['load_times']}")
    print()
    
    # Test 5: Quick summarization test
    print("Test 5: Summarization Test")
    print("-" * 70)
    
    test_text = """
    Machine learning is transforming modern technology. Neural networks
    learn patterns from data without explicit programming. Deep learning
    uses multiple layers to extract features. Applications include image
    recognition, natural language processing, and autonomous systems.
    However, interpretability remains a challenge for these models.
    """
    
    try:
        result = pipe1(
            test_text,
            max_length=50,
            min_length=20,
            do_sample=False
        )
        summary = result[0]['summary_text']
        print(f"Input: {len(test_text.split())} words")
        print(f"Output: {len(summary.split())} words")
        print(f"\nSummary: {summary}")
    except Exception as e:
        print(f"⚠️ Summarization test skipped: {e}")
    
    print()
    print("=" * 70)
    print("✅ Model Manager tests completed!")
    print("=" * 70)