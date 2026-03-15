"""
VidSummarize - Quality Guard Module
Validates summary quality and triggers fixes

Responsibility: Catch bad summaries before they reach users
Focus: Safety net for speed-first system
"""

import re
from typing import Dict, List, Tuple
from collections import Counter


# ============================================================================
# QUALITY CHECKS
# ============================================================================

def is_empty(text: str) -> bool:
    """Check if summary is effectively empty"""
    if not text or not text.strip():
        return True
    
    # Check for minimum meaningful content
    words = text.split()
    return len(words) < 10


def is_too_short(text: str, target_words: int, threshold: float = 0.3) -> bool:
    """
    Check if summary is too short relative to target
    
    Args:
        text: Summary text
        target_words: Target length in words
        threshold: Minimum acceptable ratio (default 30%)
        
    Returns:
        True if too short
    """
    words = text.split()
    actual = len(words)
    minimum = target_words * threshold
    
    return actual < minimum


def is_too_long(text: str, target_words: int, threshold: float = 1.5) -> bool:
    """
    Check if summary is too long relative to target
    
    Args:
        text: Summary text
        target_words: Target length in words
        threshold: Maximum acceptable ratio (default 150%)
        
    Returns:
        True if too long
    """
    words = text.split()
    actual = len(words)
    maximum = target_words * threshold
    
    return actual > maximum


def has_repetition(text: str, threshold: float = 0.7) -> bool:
    """
    Check for repetitive sentences
    
    Args:
        text: Summary text
        threshold: Minimum unique sentence ratio (default 70%)
        
    Returns:
        True if too repetitive
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) < 2:
        return False
    
    # Check uniqueness
    unique_sentences = set(sentences)
    uniqueness_ratio = len(unique_sentences) / len(sentences)
    
    return uniqueness_ratio < threshold


def check_idea_density(text: str, threshold: float = 0.15) -> bool:
    """
    Check conceptual density (cheap heuristic)
    
    Uses long word ratio as proxy for meaningful content
    
    Args:
        text: Summary text
        threshold: Minimum long word ratio (default 15%)
        
    Returns:
        True if density is acceptable
    """
    words = text.split()
    if not words:
        return False
    
    # Count words > 6 characters (likely meaningful)
    long_words = sum(1 for w in words if len(w) > 6)
    density = long_words / len(words)
    
    return density >= threshold


def check_completeness(text: str) -> bool:
    """
    Check if summary appears complete
    
    Signs of incompleteness:
    - Ends abruptly without punctuation
    - Contains placeholder markers
    - Has incomplete parentheses/quotes
    
    Returns:
        True if appears complete
    """
    text = text.strip()
    
    if not text:
        return False
    
    # Check ending punctuation
    if text[-1] not in '.!?':
        return False
    
    # Check for common placeholder patterns
    placeholders = ['...', '[', ']', 'TODO', 'FIXME', '###']
    for placeholder in placeholders:
        if placeholder in text:
            return False
    
    # Check balanced parentheses and quotes
    if text.count('(') != text.count(')'):
        return False
    if text.count('"') % 2 != 0:
        return False
    
    return True


def check_word_diversity(text: str, threshold: float = 0.5) -> bool:
    """
    Check vocabulary diversity
    
    Low diversity suggests poor quality (same words repeated)
    
    Args:
        text: Summary text
        threshold: Minimum unique word ratio (default 50%)
        
    Returns:
        True if diversity is acceptable
    """
    words = text.lower().split()
    if len(words) < 10:
        return True  # Too short to judge
    
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    
    return diversity >= threshold


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

class QualityGuard:
    """
    Comprehensive quality evaluation system
    
    Checks multiple quality dimensions and provides actionable feedback
    """
    
    def __init__(self):
        """Initialize quality guard with default thresholds"""
        self.thresholds = {
            'min_words': 10,
            'short_ratio': 0.3,
            'long_ratio': 1.5,
            'repetition_ratio': 0.7,
            'idea_density': 0.15,
            'word_diversity': 0.5
        }
    
    def evaluate(self, summary: str, target_words: int = 300,
                original_text: str = "") -> Dict:
        """
        Comprehensive quality evaluation
        
        Args:
            summary: Generated summary text
            target_words: Target length in words
            original_text: Original text (optional, for compression check)
            
        Returns:
            Dict with 'decision', 'issues', 'score', 'metrics'
        """
        issues = []
        metrics = {}
        
        # Check 1: Empty
        if is_empty(summary):
            return {
                'decision': 'reject',
                'issues': ['empty'],
                'score': 0,
                'metrics': {}
            }
        
        # Check 2: Length
        word_count = len(summary.split())
        metrics['word_count'] = word_count
        
        if is_too_short(summary, target_words, self.thresholds['short_ratio']):
            issues.append('too_short')
        
        if is_too_long(summary, target_words, self.thresholds['long_ratio']):
            issues.append('too_long')
        
        # Check 3: Repetition
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) > 1:
            unique_ratio = len(set(s.lower() for s in sentences)) / len(sentences)
            metrics['unique_sentence_ratio'] = unique_ratio
            
            if unique_ratio < self.thresholds['repetition_ratio']:
                issues.append('repetitive')
        
        # Check 4: Idea density
        words = summary.split()
        long_words = sum(1 for w in words if len(w) > 6)
        density = long_words / max(1, len(words))
        metrics['idea_density'] = density
        
        if density < self.thresholds['idea_density']:
            issues.append('low_density')
        
        # Check 5: Word diversity
        unique_words = set(w.lower() for w in words)
        diversity = len(unique_words) / max(1, len(words))
        metrics['word_diversity'] = diversity
        
        if diversity < self.thresholds['word_diversity']:
            issues.append('low_diversity')
        
        # Check 6: Completeness
        if not check_completeness(summary):
            issues.append('incomplete')
        
        # Check 7: Compression ratio (if original provided)
        if original_text:
            original_words = len(original_text.split())
            compression_ratio = original_words / max(1, word_count)
            metrics['compression_ratio'] = compression_ratio
            
            if compression_ratio < 2.0:
                issues.append('low_compression')
        
        # Calculate quality score (0-100)
        score = self._calculate_score(metrics, issues)
        
        # Make decision
        decision = self._make_decision(issues, score)
        
        return {
            'decision': decision,
            'issues': issues,
            'score': score,
            'metrics': metrics
        }
    
    def _calculate_score(self, metrics: Dict, issues: List[str]) -> int:
        """
        Calculate overall quality score (0-100)
        
        Based on metrics and severity of issues
        """
        # Start at 100
        score = 100
        
        # Deduct for issues (different severities)
        severity_map = {
            'empty': 100,           # Fatal
            'incomplete': 40,       # Severe
            'too_short': 30,
            'too_long': 20,
            'repetitive': 25,
            'low_density': 15,
            'low_diversity': 10,
            'low_compression': 10
        }
        
        for issue in issues:
            score -= severity_map.get(issue, 10)
        
        # Ensure 0-100 range
        return max(0, min(100, score))
    
    def _make_decision(self, issues: List[str], score: int) -> str:
        """
        Make decision based on issues and score
        
        Returns:
            'accept' | 'retry' | 'compress' | 'expand' | 'reject'
        """
        # Fatal issues
        if 'empty' in issues:
            return 'reject'
        
        # High score - accept
        if score >= 75:
            return 'accept'
        
        # Specific actionable issues
        if 'too_short' in issues and 'incomplete' not in issues:
            return 'expand'
        
        if 'too_long' in issues:
            return 'compress'
        
        # Multiple issues or low score - retry
        if score < 50 or len(issues) >= 3:
            return 'retry'
        
        # Minor issues - accept with warning
        if score >= 60:
            return 'accept'
        
        # Default - retry
        return 'retry'
    
    def quick_check(self, summary: str, target_words: int) -> bool:
        """
        Quick quality check (fast, boolean)
        
        Returns:
            True if passes basic quality checks
        """
        # Basic checks only
        if is_empty(summary):
            return False
        
        if is_too_short(summary, target_words, 0.3):
            return False
        
        if is_too_long(summary, target_words, 2.0):
            return False
        
        if not check_completeness(summary):
            return False
        
        return True
    
    def suggest_fixes(self, evaluation: Dict) -> List[str]:
        """
        Suggest fixes based on evaluation
        
        Args:
            evaluation: Result from evaluate()
            
        Returns:
            List of suggested actions
        """
        suggestions = []
        issues = evaluation.get('issues', [])
        
        if 'empty' in issues:
            suggestions.append("Regenerate with different parameters")
        
        if 'too_short' in issues:
            suggestions.append("Increase max_length parameter")
            suggestions.append("Use fewer chunks (larger chunks)")
        
        if 'too_long' in issues:
            suggestions.append("Decrease max_length parameter")
            suggestions.append("Enable idea filtering")
        
        if 'repetitive' in issues:
            suggestions.append("Enable do_sample=True")
            suggestions.append("Increase temperature")
        
        if 'low_density' in issues:
            suggestions.append("Enable idea filtering")
            suggestions.append("Use quality mode")
        
        if 'incomplete' in issues:
            suggestions.append("Check for timeout")
            suggestions.append("Retry with longer timeout")
        
        return suggestions


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create global quality guard
quality_guard = QualityGuard()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_quality(summary: str, target_words: int = 300,
                 original_text: str = "") -> Dict:
    """
    Convenience wrapper for quality evaluation
    
    Args:
        summary: Generated summary
        target_words: Target length
        original_text: Original text (optional)
        
    Returns:
        Evaluation dict
    """
    return quality_guard.evaluate(summary, target_words, original_text)


def is_acceptable(summary: str, target_words: int = 300) -> bool:
    """
    Quick boolean check - is summary acceptable?
    
    Args:
        summary: Generated summary
        target_words: Target length
        
    Returns:
        True if acceptable quality
    """
    return quality_guard.quick_check(summary, target_words)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VidSummarize - Quality Guard Test")
    print("=" * 70)
    print()
    
    # Test cases
    test_cases = [
        {
            'name': 'Good Summary',
            'text': """Machine learning transforms technology through neural 
                      networks that learn patterns from data. Deep learning uses 
                      multiple layers to extract increasingly abstract features. 
                      Applications include image recognition, natural language 
                      processing, and autonomous systems.""",
            'target': 50,
            'expected': 'accept'
        },
        {
            'name': 'Too Short',
            'text': "Machine learning is important.",
            'target': 100,
            'expected': 'expand'
        },
        {
            'name': 'Repetitive',
            'text': """Machine learning is important. Machine learning is important. 
                      Machine learning is important. Machine learning is important.""",
            'target': 30,
            'expected': 'retry'
        },
        {
            'name': 'Low Density',
            'text': """It is a thing. It does stuff. It is good. It works well. 
                      It is nice. It is okay. It does things.""",
            'target': 30,
            'expected': 'retry'
        },
        {
            'name': 'Incomplete',
            'text': "Machine learning transforms technology through neural networks that",
            'target': 30,
            'expected': 'retry'
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 70)
        
        result = quality_guard.evaluate(test['text'], test['target'])
        
        print(f"  Decision: {result['decision']}")
        print(f"  Score: {result['score']}/100")
        print(f"  Issues: {result['issues']}")
        print(f"  Metrics: {result['metrics']}")
        
        if result['decision'] != test['expected']:
            print(f"  ⚠️ Expected '{test['expected']}' but got '{result['decision']}'")
        else:
            print(f"  ✓ Correct decision")
        
        # Show suggestions if issues found
        if result['issues']:
            suggestions = quality_guard.suggest_fixes(result)
            if suggestions:
                print(f"  Suggestions:")
                for suggestion in suggestions[:3]:  # Show max 3
                    print(f"    - {suggestion}")
        
        print()
    
    print("=" * 70)
    print("✅ Quality Guard tests completed!")
    print("=" * 70)