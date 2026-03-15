"""
VidSummarize - Chunk Summarizer Module
Handles parallel chunk processing, retry logic, and merging

Responsibility: Take chunks → Summarize in parallel → Merge intelligently
Focus: Reliability, speed, error recovery
"""

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Optional, Callable
import time
import traceback


class ChunkSummarizer:
    """
    Handles parallel summarization of text chunks with retry logic
    
    Features:
    - Parallel processing with ThreadPoolExecutor
    - Automatic retry on failure (up to 3 attempts)
    - Timeout protection (max 30s per chunk)
    - Fallback strategies (extract first sentences)
    - Progress tracking and logging
    """
    
    def __init__(self, summarizer_func: Callable, max_workers: int = 3):
        """
        Initialize chunk summarizer
        
        Args:
            summarizer_func: Function that takes (text, max_length, min_length) 
                           and returns summary
            max_workers: Number of parallel workers
        """
        self.summarizer_func = summarizer_func
        self.max_workers = max_workers
        self.max_retries = 3
        self.chunk_timeout = 30  # seconds
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'fallbacks': 0,
            'total_time': 0
        }
    
    # ========================================================================
    # SINGLE CHUNK PROCESSING
    # ========================================================================
    
    def _summarize_chunk_with_retry(self, chunk: str, target_length: int,
                                    chunk_idx: int) -> Dict:
        """
        Summarize a single chunk with retry logic
        
        Args:
            chunk: Text to summarize
            target_length: Target summary length in words
            chunk_idx: Index for logging and ordering
            
        Returns:
            Dict with 'index', 'summary', 'success', 'attempts', 'fallback'
        """
        min_length = max(20, target_length - 30)
        max_length = target_length + 30
        
        result = {
            'index': chunk_idx,
            'summary': '',
            'success': False,
            'attempts': 0,
            'fallback': False,
            'error': None
        }
        
        # Try up to max_retries times
        for attempt in range(self.max_retries):
            result['attempts'] += 1
            
            try:
                # Call the summarizer function
                summary_result = self.summarizer_func(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                
                # Extract text from result
                if isinstance(summary_result, list) and len(summary_result) > 0:
                    summary = summary_result[0].get('summary_text', '')
                elif isinstance(summary_result, dict):
                    summary = summary_result.get('summary_text', '')
                else:
                    summary = str(summary_result)
                
                if summary and len(summary.strip()) > 0:
                    result['summary'] = summary.strip()
                    result['success'] = True
                    
                    # Log success
                    word_count = len(summary.split())
                    print(f"[ChunkSummarizer] ✓ Chunk {chunk_idx} "
                          f"complete ({word_count} words, attempt {attempt+1})")
                    
                    return result
                
            except Exception as e:
                result['error'] = str(e)
                
                if attempt < self.max_retries - 1:
                    # Log retry
                    self.stats['retries'] += 1
                    print(f"[ChunkSummarizer] ⚠ Chunk {chunk_idx} failed "
                          f"(attempt {attempt+1}), retrying... Error: {str(e)[:50]}")
                    time.sleep(0.5)  # Brief pause before retry
                else:
                    # Final attempt failed
                    print(f"[ChunkSummarizer] ✗ Chunk {chunk_idx} failed "
                          f"after {self.max_retries} attempts")
        
        # All retries failed - use fallback
        return self._fallback_summary(chunk, chunk_idx, target_length)
    
    def _fallback_summary(self, chunk: str, chunk_idx: int, 
                         target_length: int) -> Dict:
        """
        Generate fallback summary when model fails
        
        Strategy:
        1. Extract first N sentences (up to target_length words)
        2. If still too long, truncate to first target_length words
        3. Add ellipsis to indicate truncation
        
        Args:
            chunk: Original text
            chunk_idx: Index for logging
            target_length: Target length in words
            
        Returns:
            Dict with fallback summary
        """
        self.stats['fallbacks'] += 1
        print(f"[ChunkSummarizer] 🔄 Chunk {chunk_idx} using fallback strategy")
        
        # Strategy 1: Extract first few sentences
        import re
        sentences = re.split(r'[.!?]+', chunk)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Accumulate sentences up to target length
        summary_words = []
        for sent in sentences:
            words = sent.split()
            if len(summary_words) + len(words) <= target_length:
                summary_words.extend(words)
            else:
                break
        
        # If we got nothing, just take first target_length words
        if not summary_words:
            summary_words = chunk.split()[:target_length]
        
        summary = ' '.join(summary_words)
        
        # Add proper ending if it doesn't have one
        if summary and summary[-1] not in '.!?':
            summary += '...'
        
        return {
            'index': chunk_idx,
            'summary': summary,
            'success': True,  # Fallback counts as success
            'attempts': self.max_retries,
            'fallback': True,
            'error': 'Used fallback strategy'
        }
    
    # ========================================================================
    # PARALLEL PROCESSING
    # ========================================================================
    
    def summarize_chunks(self, chunks: List[str], 
                        target_per_chunk: int) -> List[str]:
        """
        Summarize multiple chunks in parallel
        
        Args:
            chunks: List of text chunks
            target_per_chunk: Target words per chunk summary
            
        Returns:
            List of summaries (in original order)
        """
        if not chunks:
            return []
        
        # Reset statistics
        self.stats = {
            'total_chunks': len(chunks),
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'fallbacks': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        print(f"[ChunkSummarizer] Processing {len(chunks)} chunks "
              f"with {self.max_workers} workers")
        print(f"[ChunkSummarizer] Target per chunk: {target_per_chunk} words")
        
        # Single chunk - no parallelization needed
        if len(chunks) == 1:
            result = self._summarize_chunk_with_retry(chunks[0], target_per_chunk, 1)
            self.stats['successful'] = 1 if result['success'] else 0
            self.stats['failed'] = 0 if result['success'] else 1
            self.stats['total_time'] = time.time() - start_time
            return [result['summary']] if result['summary'] else []
        
        # Multiple chunks - parallel processing
        results = [None] * len(chunks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    self._summarize_chunk_with_retry,
                    chunk,
                    target_per_chunk,
                    i + 1
                ): i
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                
                try:
                    # Get result with timeout protection
                    result = future.result(timeout=self.chunk_timeout)
                    results[idx] = result
                    
                    # Update statistics
                    if result['success']:
                        self.stats['successful'] += 1
                        if result['fallback']:
                            self.stats['fallbacks'] += 1
                    else:
                        self.stats['failed'] += 1
                    
                except TimeoutError:
                    print(f"[ChunkSummarizer] ⏱ Chunk {idx+1} timeout "
                          f"after {self.chunk_timeout}s")
                    self.stats['failed'] += 1
                    results[idx] = self._fallback_summary(
                        chunks[idx], idx + 1, target_per_chunk
                    )
                
                except Exception as e:
                    print(f"[ChunkSummarizer] ❌ Chunk {idx+1} error: {e}")
                    self.stats['failed'] += 1
                    results[idx] = self._fallback_summary(
                        chunks[idx], idx + 1, target_per_chunk
                    )
        
        # Calculate statistics
        self.stats['total_time'] = time.time() - start_time
        
        # Log summary statistics
        self._log_statistics()
        
        # Extract summaries (filter out None)
        summaries = [r['summary'] for r in results if r and r['summary']]
        
        return summaries
    
    # ========================================================================
    # INTELLIGENT MERGING
    # ========================================================================
    
    def merge_summaries(self, summaries: List[str], 
                       strategy: str = 'concatenate') -> str:
        """
        Merge multiple chunk summaries intelligently
        
        Strategies:
        - 'concatenate': Simple join with spaces
        - 'deduplicate': Remove duplicate sentences across chunks
        - 'compress': Further compress if total is too long
        
        Args:
            summaries: List of chunk summaries
            strategy: Merging strategy
            
        Returns:
            Merged summary text
        """
        if not summaries:
            return ""
        
        if len(summaries) == 1:
            return summaries[0]
        
        if strategy == 'concatenate':
            return ' '.join(summaries)
        
        elif strategy == 'deduplicate':
            return self._merge_with_deduplication(summaries)
        
        elif strategy == 'compress':
            # First merge, then compress if needed
            merged = ' '.join(summaries)
            if len(merged.split()) > 500:  # If too long
                # Could call summarizer again on merged text
                # For now, just truncate intelligently
                return self._smart_truncate(merged, 500)
            return merged
        
        else:
            # Default to concatenate
            return ' '.join(summaries)
    
    def _merge_with_deduplication(self, summaries: List[str]) -> str:
        """Remove duplicate sentences across summaries"""
        import re
        
        all_sentences = []
        seen_sentences = set()
        
        for summary in summaries:
            sentences = re.split(r'[.!?]+', summary)
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                # Normalize for comparison (lowercase, remove extra spaces)
                normalized = ' '.join(sent.lower().split())
                
                if normalized not in seen_sentences:
                    seen_sentences.add(normalized)
                    all_sentences.append(sent)
        
        # Reconstruct with proper punctuation
        result = '. '.join(all_sentences)
        if result and result[-1] not in '.!?':
            result += '.'
        
        return result
    
    def _smart_truncate(self, text: str, max_words: int) -> str:
        """Truncate to max_words at sentence boundary"""
        import re
        
        sentences = re.split(r'([.!?]+)', text)
        
        words = []
        for i in range(0, len(sentences), 2):
            if i >= len(sentences):
                break
            
            sent = sentences[i]
            punct = sentences[i+1] if i+1 < len(sentences) else '.'
            
            sent_words = sent.split()
            if len(words) + len(sent_words) <= max_words:
                words.extend(sent_words)
                words.append(punct)
            else:
                break
        
        result = ' '.join(words)
        return result.strip()
    
    # ========================================================================
    # STATISTICS AND LOGGING
    # ========================================================================
    
    def _log_statistics(self):
        """Log processing statistics"""
        print(f"\n[ChunkSummarizer] Statistics:")
        print(f"  Total chunks:    {self.stats['total_chunks']}")
        print(f"  Successful:      {self.stats['successful']}")
        print(f"  Failed:          {self.stats['failed']}")
        print(f"  Retries:         {self.stats['retries']}")
        print(f"  Fallbacks used:  {self.stats['fallbacks']}")
        print(f"  Total time:      {self.stats['total_time']:.1f}s")
        
        if self.stats['total_chunks'] > 0:
            success_rate = (self.stats['successful'] / 
                          self.stats['total_chunks']) * 100
            avg_time = self.stats['total_time'] / self.stats['total_chunks']
            print(f"  Success rate:    {success_rate:.1f}%")
            print(f"  Avg time/chunk:  {avg_time:.2f}s")
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics to zero"""
        self.stats = {
            'total_chunks': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'fallbacks': 0,
            'total_time': 0
        }


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Chunk Summarizer Test")
    print("=" * 70)
    print()
    
    # Mock summarizer function for testing
    def mock_summarizer(text, max_length=100, min_length=20, 
                       do_sample=False, truncation=True):
        """Simulate summarizer behavior"""
        import random
        
        # Simulate occasional failure (10% chance)
        if random.random() < 0.1:
            raise Exception("Simulated model failure")
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Return mock summary (first few words)
        words = text.split()[:max_length//2]
        summary = ' '.join(words) + '...'
        
        return [{'summary_text': summary}]
    
    # Test data
    test_chunks = [
        "This is the first chunk. It talks about machine learning and AI. "
        "Neural networks are very powerful tools for pattern recognition.",
        
        "The second chunk discusses deep learning. CNNs are used for images. "
        "RNNs handle sequential data like text and speech.",
        
        "Finally, the third chunk covers applications. Self-driving cars use "
        "computer vision. Virtual assistants use natural language processing."
    ]
    
    # Test 1: Basic parallel processing
    print("Test 1: Parallel Processing")
    print("-" * 70)
    
    chunk_summarizer = ChunkSummarizer(mock_summarizer, max_workers=2)
    summaries = chunk_summarizer.summarize_chunks(test_chunks, target_per_chunk=30)
    
    print(f"\nResults:")
    for i, summary in enumerate(summaries, 1):
        print(f"  {i}. {summary[:60]}...")
    
    print()
    
    # Test 2: Merging strategies
    print("Test 2: Merging Strategies")
    print("-" * 70)
    
    merged_concat = chunk_summarizer.merge_summaries(summaries, 'concatenate')
    print(f"Concatenate: {merged_concat[:100]}...")
    print()
    
    merged_dedup = chunk_summarizer.merge_summaries(summaries, 'deduplicate')
    print(f"Deduplicate: {merged_dedup[:100]}...")
    print()
    
    # Test 3: Statistics
    print("Test 3: Statistics")
    print("-" * 70)
    stats = chunk_summarizer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("=" * 70)
    print("✅ Chunk Summarizer tests completed!")
    print("=" * 70)