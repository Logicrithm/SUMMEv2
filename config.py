"""
VidSummarize - Configuration Module
Central location for all modes, budgets, thresholds, and settings

Responsibility: Define what each mode means, what limits exist
Focus: Single source of truth for all configuration
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


# ============================================================================
# PERFORMANCE MODES
# ============================================================================

@dataclass
class ModeConfig:
    """Configuration for a summarization mode"""
    name: str
    description: str
    max_time_budget_ms: int
    target_quality: int  # 0-100 scale
    use_filtering: bool
    use_structured_output: bool
    chunk_size_tokens: int
    parallel_workers: int
    retry_attempts: int


# Define all available modes
MODES = {
    'fast': ModeConfig(
        name='fast',
        description='Ultra-fast processing, minimal structure, good for previews',
        max_time_budget_ms=3000,  # 3 seconds
        target_quality=75,
        use_filtering=False,
        use_structured_output=False,
        chunk_size_tokens=1024,  # Larger chunks = fewer calls
        parallel_workers=2,
        retry_attempts=1  # No retries for speed
    ),
    
    'balanced': ModeConfig(
        name='balanced',
        description='Best speed/quality trade-off, structured output',
        max_time_budget_ms=6000,  # 6 seconds
        target_quality=85,
        use_filtering=False,  # Skip filtering for speed
        use_structured_output=True,
        chunk_size_tokens=512,
        parallel_workers=3,
        retry_attempts=2
    ),
    
    'quality': ModeConfig(
        name='quality',
        description='Maximum quality, full filtering and structure',
        max_time_budget_ms=12000,  # 12 seconds
        target_quality=93,
        use_filtering=True,  # Enable idea filtering
        use_structured_output=True,
        chunk_size_tokens=512,
        parallel_workers=3,
        retry_attempts=3
    )
}


# ============================================================================
# TIMING BUDGETS (milliseconds)
# ============================================================================

TIMING_BUDGETS = {
    'preprocessing': {
        'clean': 100,
        'split': 200,
        'domain_detect': 50,
        'filter': 300,
        'chunk': 100,
        'total': 500  # All preprocessing should be under 500ms
    },
    
    'summarization': {
        'chunk_summary': 5000,  # Per-chunk summarization
        'format': 100,
        'merge': 50
    },
    
    'overall': {
        'fast': 3000,
        'balanced': 6000,
        'quality': 12000
    }
}


# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

QUALITY_THRESHOLDS = {
    # Minimum length for valid summary
    'min_summary_words': 20,
    
    # Maximum length multiplier (summary shouldn't exceed this * target)
    'max_length_multiplier': 1.5,
    
    # Minimum compression ratio (original / summary)
    'min_compression_ratio': 2.0,
    
    # Sentence filtering - keep ratio
    'filter_keep_ratio': {
        'fast': 1.0,      # Keep all
        'balanced': 0.85, # Keep 85%
        'quality': 0.65   # Keep 65%
    },
    
    # Long word ratio threshold (words > 6 chars)
    'long_word_ratio': 0.2,
    
    # Importance score weights
    'importance_weights': {
        'long_word_ratio': 1.0,
        'has_marker': 0.5,
        'length_score': 0.3
    }
}


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'default': {
        'model_name': 'facebook/bart-large-cnn',
        'description': 'Standard BART model, good balance',
        'avg_speed_tokens_per_sec': 50,  # Rough estimate on CPU
        'quality_score': 85
    },
    
    'fast': {
        'model_name': 'sshleifer/distilbart-cnn-12-6',
        'description': 'Distilled BART, 40% faster, slightly lower quality',
        'avg_speed_tokens_per_sec': 80,
        'quality_score': 78
    },
    
    'quality': {
        'model_name': 'facebook/bart-large-cnn',
        'description': 'Same as default, best for quality mode',
        'avg_speed_tokens_per_sec': 50,
        'quality_score': 90
    }
}


# ============================================================================
# DOMAIN-SPECIFIC SETTINGS
# ============================================================================

DOMAIN_SETTINGS = {
    'tech': {
        'keep_technical_terms': True,
        'preserve_code_blocks': True,
        'format_style': 'lecture',
        'emphasis': ['implementation', 'architecture', 'performance']
    },
    
    'science': {
        'keep_technical_terms': True,
        'preserve_formulas': True,
        'format_style': 'lecture',
        'emphasis': ['hypothesis', 'results', 'conclusions']
    },
    
    'business': {
        'keep_technical_terms': False,
        'preserve_numbers': True,
        'format_style': 'lecture',
        'emphasis': ['metrics', 'strategy', 'outcomes']
    },
    
    'history': {
        'keep_technical_terms': False,
        'preserve_dates': True,
        'format_style': 'narrative',
        'emphasis': ['events', 'impact', 'context']
    },
    
    'tutorial': {
        'keep_technical_terms': True,
        'preserve_steps': True,
        'format_style': 'tutorial',
        'emphasis': ['steps', 'order', 'actions']
    },
    
    'economics': {
        'keep_technical_terms': True,
        'preserve_numbers': True,
        'format_style': 'lecture',
        'emphasis': ['concepts', 'mechanisms', 'effects']
    },
    
    'general': {
        'keep_technical_terms': False,
        'preserve_nothing': True,
        'format_style': 'general',
        'emphasis': ['key_points', 'main_ideas']
    }
}


# ============================================================================
# DYNAMIC LENGTH CALCULATION
# ============================================================================

def calculate_target_length(word_count: int, duration_seconds: float = 0,
                           mode: str = 'balanced') -> int:
    """
    Calculate target summary length based on input characteristics
    
    Rules:
    - Short videos (<5 min): 100-150 words
    - Medium videos (5-20 min): 200-300 words
    - Long videos (20-60 min): 300-400 words
    - Very long (>60 min): 400-500 words
    
    Also considers total word count
    
    Args:
        word_count: Total words in transcript
        duration_seconds: Video duration in seconds (optional)
        mode: Summarization mode
        
    Returns:
        Target summary length in words
    """
    # Base calculation from duration
    if duration_seconds > 0:
        duration_minutes = duration_seconds / 60
        
        if duration_minutes < 5:
            base_length = 125
        elif duration_minutes < 20:
            base_length = 250
        elif duration_minutes < 60:
            base_length = 350
        else:
            base_length = 450
    else:
        # Fallback to word count
        if word_count < 1000:
            base_length = 125
        elif word_count < 3000:
            base_length = 250
        elif word_count < 8000:
            base_length = 350
        else:
            base_length = 450
    
    # Adjust for mode
    mode_multipliers = {
        'fast': 0.8,      # 20% shorter
        'balanced': 1.0,  # Standard
        'quality': 1.2    # 20% longer
    }
    
    multiplier = mode_multipliers.get(mode, 1.0)
    target = int(base_length * multiplier)
    
    # Ensure minimum and maximum bounds
    return max(100, min(500, target))


# ============================================================================
# ENVIRONMENT VARIABLE HANDLING
# ============================================================================

class ConfigManager:
    """
    Manages configuration from environment variables and defaults
    
    Priority:
    1. Environment variables
    2. Mode defaults
    3. Global defaults
    """
    
    def __init__(self):
        """Initialize configuration manager"""
        self.mode = self._get_mode()
        self.mode_config = MODES.get(self.mode, MODES['balanced'])
    
    def _get_mode(self) -> str:
        """Get mode from environment or default"""
        mode = os.getenv('SUMMARY_MODE', 'balanced').lower()
        
        if mode not in MODES:
            print(f"[Config] Warning: Unknown mode '{mode}', using 'balanced'")
            return 'balanced'
        
        return mode
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Try environment variable first
        env_key = f"SUMMARY_{key.upper().replace('.', '_')}"
        env_value = os.getenv(env_key)
        
        if env_value is not None:
            # Try to convert to appropriate type
            if env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            try:
                return int(env_value)
            except ValueError:
                try:
                    return float(env_value)
                except ValueError:
                    return env_value
        
        # Try mode config
        if hasattr(self.mode_config, key):
            return getattr(self.mode_config, key)
        
        # Return default
        return default
    
    def get_mode_config(self) -> ModeConfig:
        """Get current mode configuration"""
        return self.mode_config
    
    def get_timing_budget(self, stage: str) -> int:
        """Get timing budget for a stage in milliseconds"""
        if stage in TIMING_BUDGETS['preprocessing']:
            return TIMING_BUDGETS['preprocessing'][stage]
        elif stage in TIMING_BUDGETS['summarization']:
            return TIMING_BUDGETS['summarization'][stage]
        elif stage in TIMING_BUDGETS['overall']:
            return TIMING_BUDGETS['overall'][stage]
        else:
            return 1000  # Default 1 second
    
    def get_domain_settings(self, domain: str) -> Dict:
        """Get settings for a specific domain"""
        return DOMAIN_SETTINGS.get(domain, DOMAIN_SETTINGS['general'])
    
    def should_use_filtering(self) -> bool:
        """Check if idea filtering should be used"""
        return self.mode_config.use_filtering
    
    def should_use_structured_output(self) -> bool:
        """Check if structured output should be used"""
        return self.mode_config.use_structured_output
    
    def get_parallel_workers(self) -> int:
        """Get number of parallel workers"""
        # Environment variable overrides mode config
        env_workers = os.getenv('SUMMARY_WORKERS')
        if env_workers:
            return int(env_workers)
        return self.mode_config.parallel_workers
    
    def get_model_config(self) -> Dict:
        """Get model configuration for current mode"""
        return MODEL_CONFIGS.get(self.mode, MODEL_CONFIGS['default'])
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 70)
        print("VidSummarize Configuration")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Description: {self.mode_config.description}")
        print(f"Time Budget: {self.mode_config.max_time_budget_ms}ms")
        print(f"Target Quality: {self.mode_config.target_quality}%")
        print(f"Parallel Workers: {self.get_parallel_workers()}")
        print(f"Use Filtering: {self.should_use_filtering()}")
        print(f"Use Structure: {self.should_use_structured_output()}")
        print(f"Chunk Size: {self.mode_config.chunk_size_tokens} tokens")
        print(f"Retry Attempts: {self.mode_config.retry_attempts}")
        print("=" * 70)


# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

# Create a global config instance
config = ConfigManager()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_mode_info(mode_name: str) -> Dict:
    """
    Get information about a specific mode
    
    Args:
        mode_name: Name of the mode
        
    Returns:
        Dictionary with mode information
    """
    if mode_name not in MODES:
        return None
    
    mode = MODES[mode_name]
    return {
        'name': mode.name,
        'description': mode.description,
        'time_budget_s': mode.max_time_budget_ms / 1000,
        'quality': mode.target_quality,
        'features': {
            'filtering': mode.use_filtering,
            'structure': mode.use_structured_output,
            'workers': mode.parallel_workers
        }
    }


def list_all_modes() -> Dict[str, Dict]:
    """Get information about all available modes"""
    return {name: get_mode_info(name) for name in MODES.keys()}


def validate_config() -> bool:
    """
    Validate current configuration
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check mode exists
        mode = config.mode
        if mode not in MODES:
            print(f"[Config] Error: Invalid mode '{mode}'")
            return False
        
        # Check workers is reasonable
        workers = config.get_parallel_workers()
        if workers < 1 or workers > 10:
            print(f"[Config] Warning: Unusual worker count: {workers}")
        
        # Check timing budgets are positive
        for stage in ['clean', 'split', 'chunk']:
            budget = config.get_timing_budget(stage)
            if budget <= 0:
                print(f"[Config] Error: Invalid timing budget for {stage}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[Config] Validation error: {e}")
        return False


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Configuration Module Test")
    print("=" * 70)
    print()
    
    # Test 1: Config manager
    print("Test 1: Configuration Manager")
    print("-" * 70)
    config.print_config()
    print()
    
    # Test 2: Mode information
    print("Test 2: All Available Modes")
    print("-" * 70)
    modes = list_all_modes()
    for mode_name, info in modes.items():
        print(f"\n{mode_name.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Time budget: {info['time_budget_s']}s")
        print(f"  Quality: {info['quality']}%")
        print(f"  Features: {info['features']}")
    print()
    
    # Test 3: Dynamic length calculation
    print("Test 3: Dynamic Length Calculation")
    print("-" * 70)
    test_cases = [
        (500, 180, 'balanced'),   # 3 min video, 500 words
        (3000, 1200, 'balanced'), # 20 min video, 3000 words
        (8000, 3600, 'quality'),  # 1 hour video, 8000 words
    ]
    
    for word_count, duration, mode in test_cases:
        target = calculate_target_length(word_count, duration, mode)
        print(f"  {word_count} words, {duration//60} min, {mode}: "
              f"→ {target} word summary")
    print()
    
    # Test 4: Domain settings
    print("Test 4: Domain-Specific Settings")
    print("-" * 70)
    for domain in ['tech', 'business', 'tutorial']:
        settings = config.get_domain_settings(domain)
        print(f"\n{domain.upper()}:")
        print(f"  Format: {settings['format_style']}")
        print(f"  Emphasis: {settings['emphasis']}")
    print()
    
    # Test 5: Validation
    print("Test 5: Configuration Validation")
    print("-" * 70)
    is_valid = validate_config()
    print(f"Configuration is {'valid' if is_valid else 'invalid'}")
    print()
    
    print("=" * 70)
    print("✅ Configuration module tests completed!")
    print("=" * 70)