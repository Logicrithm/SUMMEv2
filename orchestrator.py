"""
VidSummarize - Orchestrator Module
Coordinates the complete summarization pipeline

Responsibility: Wire all modules together
Focus: Timing, mode control, error isolation
"""

import time
from typing import Dict, Optional
import os

# Import all modules
try:
    from preprocess import (
        safe_clean, safe_split, detect_domain,
        light_idea_filter, sentence_chunks,
        fast_chunks, get_text_stats
    )
    from chunk_summarizer import ChunkSummarizer
    from config import config, calculate_target_length
    from model_manager import model_manager
    from quality_guard import quality_guard
    _ALL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[Orchestrator] Warning: Some modules not available: {e}")
    _ALL_MODULES_AVAILABLE = False


# ============================================================================
# MODE CONFIGURATIONS
# ============================================================================

PIPELINE_CONFIGS = {
    'fast': {
        'use_cleaning': True,
        'use_splitting': False,
        'use_filtering': False,
        'use_chunking': False,
        'parallel_workers': 1,
        'max_retries': 1,
        'quality_check': False,
        'timeout_per_chunk': 10
    },
    'balanced': {
        'use_cleaning': True,
        'use_splitting': True,
        'use_filtering': False,  # Skip for speed
        'use_chunking': True,
        'parallel_workers': 3,
        'max_retries': 2,
        'quality_check': True,
        'timeout_per_chunk': 30
    },
    'quality': {
        'use_cleaning': True,
        'use_splitting': True,
        'use_filtering': True,  # Enable for quality
        'use_chunking': True,
        'parallel_workers': 3,
        'max_retries': 3,
        'quality_check': True,
        'timeout_per_chunk': 45
    }
}


# ============================================================================
# ORCHESTRATOR CLASS
# ============================================================================

class Orchestrator:
    """
    Master coordinator for the summarization pipeline
    
    Wires together:
    - preprocess.py (cleaning, splitting, chunking)
    - chunk_summarizer.py (parallel processing)
    - model_manager.py (model loading)
    - quality_guard.py (validation)
    - config.py (mode settings)
    """
    
    def __init__(self, mode: str = 'balanced'):
        """
        Initialize orchestrator
        
        Args:
            mode: Pipeline mode (fast/balanced/quality)
        """
        self.mode = mode
        self.pipeline_config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['balanced'])
        
        # Get model manager
        if _ALL_MODULES_AVAILABLE:
            self.model_manager = model_manager
        else:
            self.model_manager = None
        
        # Statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_time': 0,
            'avg_time': 0
        }
        
        print(f"[Orchestrator] Initialized in '{mode}' mode")
    
    def _timed_stage(self, name: str, func, *args, 
                    budget_ms: int = 1000, **kwargs):
        """
        Execute stage with timing
        
        Args:
            name: Stage name
            func: Function to execute
            budget_ms: Time budget in milliseconds
            
        Returns:
            (result, elapsed_ms) tuple
        """
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        
        status = "✓" if elapsed_ms <= budget_ms else "⚠"
        print(f"[Orchestrator] {status} {name:15} {elapsed_ms:6.1f}ms")
        
        return result, elapsed_ms
    
    def run(self, text: str, target_words: int = 300) -> Dict:
        """
        Execute complete summarization pipeline
        
        Args:
            text: Input text to summarize
            target_words: Target summary length in words
            
        Returns:
            Dict with 'summary', 'stats', 'quality', 'metadata'
        """
        pipeline_start = time.time()
        stage_times = {}
        
        print(f"\n[Orchestrator] ⚡ Starting pipeline (mode: {self.mode})")
        print(f"[Orchestrator] Input: {len(text.split())} words")
        print(f"[Orchestrator] Target: {target_words} words")
        
        try:
            # ================================================================
            # STAGE 1: PREPROCESSING
            # ================================================================
            
            # 1.1: Clean text
            if self.pipeline_config['use_cleaning']:
                cleaned, elapsed = self._timed_stage(
                    'Clean',
                    safe_clean,
                    text,
                    budget_ms=100
                )
                stage_times['clean'] = elapsed
            else:
                cleaned = text
            
            # 1.2: Domain detection (always do this)
            domain, elapsed = self._timed_stage(
                'Domain',
                detect_domain,
                cleaned.split(),
                budget_ms=50
            )
            stage_times['domain'] = elapsed
            print(f"[Orchestrator] 🎯 Detected domain: {domain}")
            
            # 1.3: Sentence splitting (if enabled)
            if self.pipeline_config['use_splitting']:
                sentences, elapsed = self._timed_stage(
                    'Split',
                    safe_split,
                    cleaned,
                    budget_ms=200
                )
                stage_times['split'] = elapsed
                
                # 1.4: Filtering (if enabled)
                if self.pipeline_config['use_filtering']:
                    sentences, elapsed = self._timed_stage(
                        'Filter',
                        light_idea_filter,
                        sentences,
                        domain,
                        budget_ms=300
                    )
                    stage_times['filter'] = elapsed
                
                # 1.5: Chunking
                if self.pipeline_config['use_chunking']:
                    chunks, elapsed = self._timed_stage(
                        'Chunk',
                        sentence_chunks,
                        sentences,
                        budget_ms=100
                    )
                    stage_times['chunk'] = elapsed
                else:
                    chunks = [' '.join(sentences)]
            else:
                # Fast mode - single chunk
                chunks, elapsed = self._timed_stage(
                    'FastChunk',
                    fast_chunks,
                    cleaned,
                    budget_ms=50
                )
                stage_times['chunk'] = elapsed
            
            print(f"[Orchestrator] 📦 Created {len(chunks)} chunks")
            
            # ================================================================
            # STAGE 2: SUMMARIZATION
            # ================================================================
            
            # Get model
            summarizer_func = self.model_manager.get_pipeline(self.mode)
            
            # Create chunk summarizer
            chunk_sum = ChunkSummarizer(
                summarizer_func,
                max_workers=self.pipeline_config['parallel_workers']
            )
            
            # Calculate per-chunk target
            target_per_chunk = max(30, target_words // max(1, len(chunks)))
            
            # Summarize chunks
            print(f"[Orchestrator] 🔄 Summarizing with {self.pipeline_config['parallel_workers']} workers")
            start = time.time()
            chunk_summaries = chunk_sum.summarize_chunks(chunks, target_per_chunk)
            stage_times['summarize'] = (time.time() - start) * 1000
            
            if not chunk_summaries:
                raise Exception("Summarization produced no output")
            
            # ================================================================
            # STAGE 3: MERGING & FORMATTING
            # ================================================================
            
            # Merge summaries
            if len(chunk_summaries) == 1:
                merged_summary = chunk_summaries[0]
            else:
                merged_summary, elapsed = self._timed_stage(
                    'Merge',
                    chunk_sum.merge_summaries,
                    chunk_summaries,
                    'concatenate',
                    budget_ms=50
                )
                stage_times['merge'] = elapsed
            
            # Apply structured formatting (if not fast mode)
            if self.mode != 'fast':
                final_summary = self._format_by_domain(merged_summary, domain, chunk_summaries)
                stage_times['format'] = 20  # Estimate
            else:
                final_summary = merged_summary
            
            # ================================================================
            # STAGE 4: QUALITY CHECK
            # ================================================================
            
            quality_result = None
            
            if self.pipeline_config['quality_check']:
                quality_result, elapsed = self._timed_stage(
                    'QualityCheck',
                    quality_guard.evaluate,
                    final_summary,
                    target_words,
                    text,
                    budget_ms=100
                )
                stage_times['quality'] = elapsed
                
                print(f"[Orchestrator] 📊 Quality score: {quality_result['score']}/100")
                
                # Handle quality issues
                if quality_result['decision'] == 'retry':
                    print(f"[Orchestrator] ⚠️ Quality check suggests retry")
                    print(f"[Orchestrator] Issues: {quality_result['issues']}")
                    
                    # For now, we'll accept it anyway
                    # In production, you might retry with different params
                elif quality_result['decision'] == 'reject':
                    print(f"[Orchestrator] ❌ Quality check rejected summary")
                    # Use fallback - first N words of original
                    final_summary = self._fallback_summary(text, target_words)
            
            # ================================================================
            # FINALIZE
            # ================================================================
            
            total_time = (time.time() - pipeline_start) * 1000
            
            # Update statistics
            self.stats['total_runs'] += 1
            self.stats['successful_runs'] += 1
            self.stats['total_time'] += total_time
            self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_runs']
            
            # Log completion
            word_count = len(final_summary.split())
            print(f"[Orchestrator] ✅ Complete: {word_count} words in {total_time:.0f}ms")
            
            # Check against budget
            mode_budget = config.get_timing_budget(self.mode)
            if total_time > mode_budget:
                print(f"[Orchestrator] ⚠️ Exceeded time budget ({mode_budget}ms)")
            
            return {
                'summary': final_summary,
                'stats': {
                    'total_time_ms': total_time,
                    'stage_times': stage_times,
                    'word_count': word_count,
                    'chunk_count': len(chunks),
                    'mode': self.mode
                },
                'quality': quality_result,
                'metadata': {
                    'domain': domain,
                    'original_words': len(text.split()),
                    'compression_ratio': len(text.split()) / max(1, word_count)
                }
            }
            
        except Exception as e:
            # Handle errors
            self.stats['total_runs'] += 1
            self.stats['failed_runs'] += 1
            
            print(f"[Orchestrator] ❌ Pipeline failed: {str(e)}")
            
            # Return fallback summary
            fallback = self._fallback_summary(text, target_words)
            
            return {
                'summary': fallback,
                'stats': {
                    'total_time_ms': (time.time() - pipeline_start) * 1000,
                    'stage_times': stage_times,
                    'word_count': len(fallback.split()),
                    'error': str(e)
                },
                'quality': None,
                'metadata': {
                    'domain': 'unknown',
                    'error': True
                }
            }
    
    def _format_by_domain(self, summary: str, domain: str, 
                         chunk_summaries: list) -> str:
        """Apply domain-specific formatting"""
        # Simple formatting - just return as-is for now
        # Could be enhanced with structured templates
        return summary
    
    def _fallback_summary(self, text: str, target_words: int) -> str:
        """Generate fallback summary when everything fails"""
        import re
        
        # Extract first N sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Accumulate up to target words
        summary_words = []
        for sent in sentences:
            words = sent.split()
            if len(summary_words) + len(words) <= target_words:
                summary_words.extend(words)
            else:
                break
        
        summary = ' '.join(summary_words)
        if summary and summary[-1] not in '.!?':
            summary += '...'
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Get orchestrator statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'total_time': 0,
            'avg_time': 0
        }


# ============================================================================
# CONVENIENCE API
# ============================================================================

def summarize(text: str, mode: str = 'balanced', 
             target_words: int = 300) -> str:
    """
    High-level API for summarization
    
    Args:
        text: Input text
        mode: Pipeline mode (fast/balanced/quality)
        target_words: Target summary length
        
    Returns:
        Summary text
    """
    orch = Orchestrator(mode=mode)
    result = orch.run(text, target_words)
    return result['summary']


def summarize_with_stats(text: str, mode: str = 'balanced',
                        target_words: int = 300) -> Dict:
    """
    Summarize and return full result including stats
    
    Args:
        text: Input text
        mode: Pipeline mode
        target_words: Target length
        
    Returns:
        Full result dict
    """
    orch = Orchestrator(mode=mode)
    return orch.run(text, target_words)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Orchestrator Test")
    print("=" * 70)
    print()
    
    # Sample text
    sample_text = """
    Machine learning is transforming modern technology in profound ways.
    Neural networks learn patterns from data without explicit programming,
    enabling computers to recognize images, understand speech, and make
    predictions. Deep learning uses multiple layers to extract increasingly
    abstract features from raw data. Training requires large datasets and
    significant computational power, often using GPUs. Applications include
    image recognition, natural language processing, autonomous vehicles,
    and medical diagnosis. However, models can exhibit bias if training
    data isn't representative. Interpretability remains a major challenge,
    as neural networks are often considered black boxes. Researchers work
    on explainable AI to make decisions more transparent. Despite challenges,
    machine learning continues to advance rapidly with new architectures
    and techniques emerging regularly.
    """
    
    # Test all modes
    for mode in ['fast', 'balanced', 'quality']:
        print(f"\nTesting {mode.upper()} mode")
        print("=" * 70)
        
        try:
            result = summarize_with_stats(sample_text, mode=mode, target_words=80)
            
            print(f"\nSummary ({len(result['summary'].split())} words):")
            print("-" * 70)
            print(result['summary'])
            print()
            
            print("Statistics:")
            print(f"  Total time: {result['stats']['total_time_ms']:.0f}ms")
            print(f"  Chunks: {result['stats']['chunk_count']}")
            if result['quality']:
                print(f"  Quality score: {result['quality']['score']}/100")
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
        
        print()
    
    print("=" * 70)
    print("✅ Orchestrator tests completed!")
    print("=" * 70)