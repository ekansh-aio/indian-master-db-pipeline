"""
Legal Text Cleaner Module
Removes law report headers, judge names, citations, and formatting noise.
"""
import re
import logging
from typing import Optional

# Configure logger
logger = logging.getLogger(__name__)


class LegalTextCleaner:
    """Optimized legal text cleaner with compiled regex patterns."""
    
    def __init__(self):
        """Pre-compile regex patterns for better performance."""
        self.patterns = {
            'law_report': re.compile(
                r'THE SUPREME COURT REPORTS.*?\)',
                flags=re.IGNORECASE
            ),
            'judge_list': re.compile(r'\([A-Z\.\s,;:]+JJ?\.\)'),
            'page_headers': re.compile(
                r'SUPREME COURT REPORTS.*?\]',
                flags=re.IGNORECASE
            ),
            'citations': re.compile(r'\(\d{4}[^)]*[A-Z]{2,}[^)]*\)'),
            'hyphen_breaks': re.compile(r'-\s+'),
            'whitespace': re.compile(r'\s+'),
        }
        logger.debug("LegalTextCleaner initialized with compiled regex patterns")
    
    def clean(self, text: str) -> str:
        """
        Clean legal text by removing headers, citations, and noise.
        
        Args:
            text: Raw legal text to clean
            
        Returns:
            Cleaned text string
        """
        if not text:
            logger.warning("Empty text provided for cleaning")
            return ""
        
        original_length = len(text)
        logger.info(f"Starting text cleaning. Original length: {original_length} chars")
        
        try:
            # Apply cleaning patterns sequentially
            text = self.patterns['law_report'].sub('', text)
            text = self.patterns['judge_list'].sub('', text)
            text = self.patterns['page_headers'].sub('', text)
            text = self.patterns['citations'].sub('', text)
            text = self.patterns['hyphen_breaks'].sub('', text)
            text = self.patterns['whitespace'].sub(' ', text)
            
            cleaned_text = text.strip()
            cleaned_length = len(cleaned_text)
            reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
            
            logger.info(
                f"Text cleaning complete. Final length: {cleaned_length} chars "
                f"({reduction_pct:.1f}% reduction)"
            )
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error during text cleaning: {e}", exc_info=True)
            raise


# Backward compatibility: keep original function interface
_cleaner_instance: Optional[LegalTextCleaner] = None


def clean_legal_text(text: str) -> str:
    """
    Removes law report headers, judge names, citations, and formatting noise.
    Keeps only judgment body.
    
    Args:
        text: Raw legal text
        
    Returns:
        Cleaned text
    """
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = LegalTextCleaner()
    return _cleaner_instance.clean(text)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with sample text
    sample = """
    THE SUPREME COURT REPORTS [1960]
    (S. R. DAS, C.J., B. P. SINHA, J.)
    SUPREME COURT REPORTS [1960(1)]
    Sample judgment text here (1960) 1 SCR 100.
    This is the - actual content.
    """
    
    cleaned = clean_legal_text(sample)
    print("Cleaned text:", cleaned)
