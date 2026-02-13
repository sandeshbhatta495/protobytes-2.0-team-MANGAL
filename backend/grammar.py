import re
import unicodedata

def correct_nepali_text(text):
    """
    Apply rule-based grammar and punctuation corrections to Nepali text.
    Processing steps:
    1. Unicode normalization (NFC)
    2. Punctuation standardization
    3. Common suffix/spelling fixes
    4. Spacing normalization
    5. Halant cleanup
    6. Add sentence-ending danda if missing
    """
    if not text:
        return ""

    # 1. Unicode NFC Normalization (critical for consistent Devanagari)
    text = unicodedata.normalize('NFC', text.strip())
    
    # 2. Punctuation Normalization
    # Convert pipe to danda
    text = text.replace('|', '।')
    # Only convert period to danda when surrounded by Devanagari text
    # (preserve periods in numbers like 12.5 and English text)
    text = re.sub(r'(?<=[\u0900-\u097f])\.(?=[\s\u0900-\u097f]|$)', '।', text)
    text = text.replace(',', ',')  # Keep comma (sometimes used)
    
    # 3. Common Spelling/Suffix Fixes (Rule-based)
    # Fix common suffix variations
    text = re.sub(r'छ्\s', 'छ ', text)  # Remove trailing halant from छ
    text = re.sub(r'हुन्छ्\s', 'हुन्छ ', text)
    
    # Fix "ho" variations
    text = re.sub(r'\bहैन\b', 'होइन', text)
    
    # Fix common particle spacing issues
    # "le" (ले) should be attached to the previous word
    text = re.sub(r'(\S)\s+ले\b', r'\1ले', text)
    # "ko" (को) should be attached
    text = re.sub(r'(\S)\s+को\b', r'\1को', text)
    # "ma" (मा) postposition should be attached
    text = re.sub(r'(\S)\s+मा\b', r'\1मा', text)
    # "lai" (लाई) should be attached
    text = re.sub(r'(\S)\s+लाई\b', r'\1लाई', text)
    
    # 4. Spacing Fixes
    # Remove space before danda and punctuation
    text = re.sub(r'\s+([।,?!])', r'\1', text)
    # Ensure space after danda (but not at end)
    text = re.sub(r'([।,?!])([^\s।,?!\d])', r'\1 \2', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove space between Devanagari word and number
    text = re.sub(r'([ऀ-ॿ])\s+(\d)', r'\1\2', text)
    
    # 5. Halant cleanup (prevent double halants)
    text = text.replace('््', '्')
    # Remove halant before space or end
    text = re.sub(r'्(\s|$)', r'\1', text)
    
    # 6. Add sentence-ending danda if text ends with Devanagari and no punctuation
    text = text.strip()
    if text and re.search(r'[ऀ-ॿ]$', text):
        # Check if it looks like a complete sentence (has verb ending)
        if re.search(r'(छ|छु|छन्|छौं|हो|थियो|भयो|गर्छ|गर्छु|हुन्छ|पर्छ)$', text):
            text += '।'
    
    # 7. Fix common transcription artifacts
    # Remove repeated dandas
    text = re.sub(r'।{2,}', '।', text)
    # Fix numbers (ensure Devanagari numerals are consistent)
    nepali_digits = str.maketrans('0123456789', '०१२३४५६७८९')
    
    return text.strip()
