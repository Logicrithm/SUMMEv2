"""
VidSummarize - Summarization Module
FIXED VERSION - Better error handling and validation
"""

from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import torch


class Summarizer:
    """Handles text summarization using AI models"""
    
    def __init__(self, model_name='facebook/bart-large-cnn'):
        """
        Initialize summarizer
        
        Args:
            model_name (str): Model to use
                - 'facebook/bart-large-cnn' (default, best quality)
                - 'facebook/bart-base' (faster, smaller)
                - 't5-small' (alternative)
                - 't5-base' (alternative, better)
        """
        print(f"[Summarizer] Loading model: {model_name}")
        
        # Check device
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if self.device == 0 else "cpu"
        print(f"[Summarizer] Using device: {device_name}")
        
        try:
            # Load summarization pipeline
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=self.device
            )
            
            print(f"[Summarizer] Model loaded successfully")
            
        except Exception as e:
            print(f"[Summarizer] Error loading model: {str(e)}")
            raise
        
        self.model_name = model_name
        
        # Model-specific parameters
        if 'bart' in model_name.lower():
            self.max_input_length = 1024
            self.max_output_length = 142
            self.min_output_length = 56
        elif 't5' in model_name.lower():
            self.max_input_length = 512
            self.max_output_length = 150
            self.min_output_length = 30
        else:
            self.max_input_length = 1024
            self.max_output_length = 150
            self.min_output_length = 50
    
    def summarize_text(self, text, max_length=None, min_length=None):
        """
        Summarize text - FIXED with better error handling
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length (optional)
            min_length (int): Minimum summary length (optional)
            
        Returns:
            dict: Summarization result
        """
        try:
            print(f"[Summarizer] Summarizing text...")
            
            # FIXED: Validate input text
            if not text or not isinstance(text, str):
                return {
                    'success': False,
                    'error': 'Invalid input: text must be a non-empty string'
                }
            
            # Clean and validate text
            text = text.strip()
            if not text:
                return {
                    'success': False,
                    'error': 'Text is empty after cleaning'
                }
            
            word_count = len(text.split())
            print(f"[Summarizer] Input length: {len(text)} characters, {word_count} words")
            
            # Use default lengths if not provided
            if max_length is None:
                max_length = self.max_output_length
            if min_length is None:
                min_length = self.min_output_length
            
            # FIXED: Check if text is too short with better threshold
            if word_count < 50:
                print(f"[Summarizer] Text too short ({word_count} words), returning original")
                return {
                    'success': True,
                    'summary': text,
                    'word_count': word_count,
                    'compression_ratio': 1.0,
                    'warning': f'Text too short to summarize ({word_count} words), returning original'
                }
            
            # FIXED: Adjust min_length for shorter texts
            if word_count < 100:
                min_length = max(20, word_count // 3)
                max_length = max(min_length + 20, word_count // 2)
                print(f"[Summarizer] Adjusted lengths for short text: min={min_length}, max={max_length}")
            
            # Split long text into chunks if needed
            if word_count > self.max_input_length:
                print(f"[Summarizer] Text too long ({word_count} words), splitting into chunks...")
                summary = self._summarize_long_text(text, max_length, min_length)
            else:
                # Summarize directly with error handling
                try:
                    result = self.summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True  # FIXED: Added truncation
                    )
                    
                    # FIXED: Validate result
                    if not result or not isinstance(result, list) or len(result) == 0:
                        raise ValueError("Summarizer returned empty result")
                    
                    summary = result[0].get('summary_text', '')
                    
                    # FIXED: Validate summary
                    if not summary or len(summary.strip()) < 5:
                        raise ValueError("Generated summary is too short or empty")
                        
                except Exception as model_error:
                    print(f"[Summarizer] Model error: {str(model_error)}")
                    # Try with more lenient parameters
                    try:
                        print(f"[Summarizer] Retrying with lenient parameters...")
                        result = self.summarizer(
                            text[:1000],  # Use first 1000 chars only
                            max_length=100,
                            min_length=20,
                            do_sample=False,
                            truncation=True
                        )
                        summary = result[0].get('summary_text', '')
                        
                        if not summary or len(summary.strip()) < 5:
                            raise ValueError("Retry also failed")
                            
                    except Exception as retry_error:
                        print(f"[Summarizer] Retry failed: {str(retry_error)}")
                        return {
                            'success': False,
                            'error': f"Summarization failed: {str(model_error)}"
                        }
            
            # FIXED: Validate final summary
            summary = summary.strip()
            if not summary:
                return {
                    'success': False,
                    'error': 'Generated summary is empty'
                }
            
            summary_word_count = len(summary.split())
            compression_ratio = summary_word_count / word_count if word_count > 0 else 0
            
            print(f"[Summarizer] Summary generated successfully")
            print(f"[Summarizer] Summary length: {len(summary)} characters, {summary_word_count} words")
            print(f"[Summarizer] Compression ratio: {compression_ratio:.1%}")
            
            return {
                'success': True,
                'summary': summary,
                'word_count': summary_word_count,
                'compression_ratio': compression_ratio
            }
            
        except Exception as e:
            print(f"[Summarizer] Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f"Summarization failed: {str(e)}"
            }
    
    def _summarize_long_text(self, text, max_length, min_length):
        """
        Summarize long text by splitting into chunks - FIXED
        
        Args:
            text (str): Long text to summarize
            max_length (int): Max summary length per chunk
            min_length (int): Min summary length per chunk
            
        Returns:
            str: Combined summary
        """
        # Split text into sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Create chunks of ~500 words
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > 500 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"[Summarizer] Split into {len(chunks)} chunks")
        
        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"[Summarizer] Processing chunk {i+1}/{len(chunks)}...")
            
            try:
                # FIXED: More robust chunk processing
                chunk_word_count = len(chunk.split())
                
                # Adjust parameters for chunk size
                chunk_max = min(max_length, chunk_word_count // 2)
                chunk_min = min(min_length // 2, chunk_max - 10)
                
                if chunk_min >= chunk_max:
                    chunk_min = max(10, chunk_max - 20)
                
                result = self.summarizer(
                    chunk,
                    max_length=chunk_max,
                    min_length=chunk_min,
                    do_sample=False,
                    truncation=True
                )
                
                chunk_summary = result[0].get('summary_text', '').strip()
                if chunk_summary:
                    summaries.append(chunk_summary)
                    
            except Exception as e:
                print(f"[Summarizer] Chunk {i+1} failed: {str(e)}")
                # Add first few sentences of chunk as fallback
                fallback = '. '.join(chunk.split('.')[:3]) + '.'
                summaries.append(fallback)
                continue
        
        # FIXED: Validate we have summaries
        if not summaries:
            # Return first 200 words of original text as ultimate fallback
            words = text.split()[:200]
            return ' '.join(words) + ('...' if len(text.split()) > 200 else '')
        
        # Combine summaries
        combined = ' '.join(summaries)
        
        # If combined is still too long, summarize again
        combined_word_count = len(combined.split())
        if combined_word_count > max_length * 2:
            print(f"[Summarizer] Combined summary too long ({combined_word_count} words), summarizing again...")
            try:
                result = self.summarizer(
                    combined,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                return result[0].get('summary_text', combined).strip()
            except Exception as e:
                print(f"[Summarizer] Final summarization failed: {str(e)}")
                # Return combined summaries as-is
                return combined
        
        return combined
    
    def summarize_with_bullet_points(self, text):
        """
        Generate summary with bullet points
        
        Args:
            text (str): Text to summarize
            
        Returns:
            dict: Summary with bullet points
        """
        result = self.summarize_text(text)
        
        if not result['success']:
            return result
        
        summary = result['summary']
        
        # Split into sentences and create bullet points
        sentences = summary.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Create bullet points
        bullet_points = ['• ' + sentence + '.' for sentence in sentences[:5]]  # Max 5 points
        
        result['bullet_points'] = bullet_points
        result['formatted_summary'] = '\n'.join(bullet_points)
        
        return result
    
    def save_summary(self, summary_text, output_path):
        """
        Save summary to text file
        
        Args:
            summary_text (str): Summary text
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            print(f"[Summarizer] Summary saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"[Summarizer] Save error: {str(e)}")
            return False


# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("VidSummarize - Summarizer Test (FIXED)")
    print("=" * 60)
    
    # Check PyTorch and CUDA
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test text
    test_text = """
    Artificial intelligence is transforming the way we live and work. Machine learning 
    algorithms are being used in various applications, from healthcare to finance. 
    Natural language processing enables computers to understand and generate human language. 
    Computer vision allows machines to interpret and analyze visual information from the world. 
    These technologies are becoming increasingly important in modern society and will continue 
    to shape our future in significant ways. Companies are investing heavily in AI research 
    and development to stay competitive in the market.
    """
    
    try:
        print("\nInitializing summarizer (this may take a minute)...")
        summarizer = Summarizer(model_name='facebook/bart-large-cnn')
        
        print("\n✅ Summarizer initialized successfully!")
        print(f"Model: {summarizer.model_name}")
        print(f"Device: {'cuda' if summarizer.device == 0 else 'cpu'}")
        
        # Test summarization
        print("\nTest summarization:")
        print(f"Original text ({len(test_text.split())} words):")
        print(test_text[:200] + "...")
        
        result = summarizer.summarize_text(test_text)
        
        if result['success']:
            print(f"\nSummary ({result['word_count']} words):")
            print(result['summary'])
            print(f"\nCompression: {result['compression_ratio']:.1%}")
        else:
            print(f"\n❌ Error: {result['error']}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Summarizer module test complete!")
    print("=" * 60)