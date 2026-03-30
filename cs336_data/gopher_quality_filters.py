"""Gopher quality filters implementation."""

import ssl
from nltk.tokenize import word_tokenize

# Handle SSL certificate issues for NLTK data download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure punkt tokenizer is available
try:
    word_tokenize("test")
except LookupError:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)


def gopher_quality_filter(text: str) -> bool:
    """
    Check if text passes the Gopher quality filters.
    
    Rejects text that:
    - Contains less than 50 or more than 100,000 words
    - Has mean word length outside the range of 3 to 10 characters
    - Has more than 30% of lines ending with "..."
    - Contains less than 80% of words with at least one alphabetic character
    
    Args:
        text: The input text to filter
        
    Returns:
        True if text passes all filters, False otherwise
    """
    
    # Tokenize text into words
    tokens = word_tokenize(text)
    
    # Filter 1: Check word count (50 to 100,000 words)
    word_count = len(tokens)
    if word_count < 50 or word_count > 100_000:
        return False
    
    # Filter 2: Check mean word length (3 to 10 characters)
    if word_count > 0:
        total_length = sum(len(token) for token in tokens)
        mean_length = total_length / word_count
        if mean_length < 3 or mean_length > 10:
            return False
    
    # Filter 3: Check percentage of lines ending with "..."
    lines = text.split('\n')
    if len(lines) > 0:
        lines_ending_with_ellipsis = sum(1 for line in lines if line.rstrip().endswith('...'))
        ellipsis_percentage = lines_ending_with_ellipsis / len(lines)
        if ellipsis_percentage > 0.3:
            return False
    
    # Filter 4: Check percentage of words with at least one alphabetic character
    words_with_alpha = sum(1 for token in tokens if any(c.isalpha() for c in token))
    if word_count > 0:
        alpha_percentage = words_with_alpha / word_count
        if alpha_percentage < 0.8:
            return False
    
    return True
