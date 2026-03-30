def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """Extract text from HTML bytes.

    Args:
        html_bytes: The HTML content as bytes.  
    Returns:
        The extracted text as a string, or None if extraction fails.
    """
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding

    try:
        html_text = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        detected = detect_encoding(html_bytes)
        if detected is None:
            return None

        try:
            html_text = html_bytes.decode(detected)
        except (LookupError, UnicodeDecodeError):
            return None

    return extract_plain_text(html_text)
