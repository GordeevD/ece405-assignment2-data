"""Functions for masking personally identifiable information (PII) in text."""

import re
from typing import Tuple


def mask_emails(text: str) -> Tuple[str, int]:
    """
    Mask email addresses in text by replacing them with |||EMAIL_ADDRESS|||.
    
    Args:
        text: Input text containing potential email addresses.
        
    Returns:
        A tuple containing:
        - The text with all email addresses replaced
        - The count of email addresses that were masked
    """
    # Pattern for matching email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    
    # Find all matches to count them
    matches = re.finditer(email_pattern, text)
    count = len(list(re.finditer(email_pattern, text)))
    
    # Replace all matches
    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    
    return masked_text, count


def mask_phone_numbers(text: str) -> Tuple[str, int]:
    """
    Mask phone numbers in text by replacing them with |||PHONE_NUMBER|||.
    
    Handles various US phone number formats:
    - 10 digits: 2831823829
    - With dashes: 283-182-3829
    - With parentheses and dash: (283)-182-3829
    - With parentheses and spaces: (283) 182 3829
    
    Args:
        text: Input text containing potential phone numbers.
        
    Returns:
        A tuple containing:
        - The text with all phone numbers replaced
        - The count of phone numbers that were masked
    """
    # Pattern for matching US phone numbers in various formats
    # Handles: 10 digits, dashes, parentheses, spaces
    phone_pattern = r'(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    # Find all matches to count them
    count = len(list(re.finditer(phone_pattern, text)))
    
    # Replace all matches
    masked_text = re.sub(phone_pattern, '|||PHONE_NUMBER|||', text)
    
    return masked_text, count


def mask_ips(text: str) -> Tuple[str, int]:
    """
    Mask IPv4 addresses in text by replacing them with |||IP_ADDRESS|||.
    
    Args:
        text: Input text containing potential IPv4 addresses.
        
    Returns:
        A tuple containing:
        - The text with all IPv4 addresses replaced
        - The count of IPv4 addresses that were masked
    """
    # Pattern for matching IPv4 addresses (0-255 for each octet)
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    
    # Find all matches to count them
    count = len(list(re.finditer(ip_pattern, text)))
    
    # Replace all matches
    masked_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)
    
    return masked_text, count
