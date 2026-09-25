"""
High-performance text preprocessing and standardization for Business Entity Resolution.
Supports multilingual records (US, India, France) across Latin, Devanagari, and Tamil scripts.
"""

import re
import unicodedata
from typing import List, Optional, Set

# Regex patterns
RE_WHITESPACE = re.compile(r"\s+")
RE_NON_ALPHANUM_BASIC = re.compile(r"[^\w\s]", re.UNICODE)
RE_NUMBERS = re.compile(r"\b\d+\b")

# Legal Suffixes by Country / Language
US_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "co", "company", "enterprises", "enterprise", "holdings", "holding",
    "services", "group", "pc", "pllc", "assoc", "associates"
}

INDIA_SUFFIXES = {
    "pvt ltd", "private limited", "pvt limited", "private ltd", "limited", "ltd",
    "llp", "enterprises", "enterprise", "traders", "trading", "services",
    "industries", "industry", "solutions", "associates",
    # Devanagari
    "प्राइवेट लिमिटेड", "प्रा लिमिटेड", "लिमिटेड", "एलएलपी", "इंटरप्राइजेज",
    # Tamil
    "எல்எல்பி", "பிரைவேட் லிமிடெட்"
}

FRANCE_SUFFIXES = {
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "scop",
    "et fils", "& fils", "fils", "freres", "cie", "compagnie",
    "societe", "et cie", "groupe"
}

ALL_SUFFIXES = US_SUFFIXES | INDIA_SUFFIXES | FRANCE_SUFFIXES

# Address standardizations (common across US, India, France)
ADDRESS_SUBSTITUTIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bpl\b": "place",
    r"\bter\b": "terrace",
    r"\bterrace\b": "terrace",
    r"\bpkwy\b": "parkway",
    r"\bct\b": "court",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bpo box\b": "pobox",
    r"\bp o box\b": "pobox",
    # India address markers
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\brd\.\b": "road",
    # France address markers
    r"\ball\b": "allee",
    r"\bimp\b": "impasse",
}


def normalize_unicode(text: Optional[str]) -> str:
    """Normalize unicode, strip excess whitespace and lowercases string."""
    if not text:
        return ""
    # NFKD separates base characters from diacritical marks
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    return RE_WHITESPACE.sub(" ", text)


def clean_business_name(name: Optional[str], country: Optional[str] = None) -> str:
    """Clean business name by stripping punctuation, standardizing tokens,

    and removing legal company suffixes for blocking and matching.
    """
    if not name:
        return ""
    s = normalize_unicode(name)
    # Remove leading noise symbols (e.g. '-- ', '<< ', '++ ')
    s = re.sub(r"^[^a-zA-Z0-9\u0900-\u097F\u0B80-\u0BFF]+", "", s)
    # Replace common separators with spaces
    s = re.sub(r"[\.,\-/_+*&()]", " ", s)
    s = RE_WHITESPACE.sub(" ", s).strip()

    # Determine suffix list based on country
    if country == "France":
        suffixes = FRANCE_SUFFIXES | {"ltd", "inc"}
    elif country == "India":
        suffixes = INDIA_SUFFIXES | {"ltd", "inc", "co"}
    elif country == "US":
        suffixes = US_SUFFIXES
    else:
        suffixes = ALL_SUFFIXES

    tokens = s.split()
    if not tokens:
        return s

    # Strip multi-word suffixes (e.g. 'private limited', 'pvt ltd')
    two_word_end = " ".join(tokens[-2:]) if len(tokens) >= 2 else ""
    if two_word_end in suffixes and len(tokens) > 2:
        tokens = tokens[:-2]
    elif tokens[-1] in suffixes and len(tokens) > 1:
        tokens = tokens[:-1]

    cleaned = " ".join(tokens).strip()
    return cleaned if cleaned else s


def clean_business_address(address: Optional[str], country: Optional[str] = None) -> str:
    """Clean business address by standardizing street/locality types and removing noise."""
    if not address or address.lower() in ("none", "null", "nan"):
        return ""
    s = normalize_unicode(address)
    # Clean punctuation
    s = re.sub(r"[\.,\-/_+*&()]", " ", s)
    s = RE_WHITESPACE.sub(" ", s).strip()

    # Apply standard abbreviation replacements
    for pattern, replacement in ADDRESS_SUBSTITUTIONS.items():
        s = re.sub(pattern, replacement, s)

    return RE_WHITESPACE.sub(" ", s).strip()


def extract_numbers(text: Optional[str]) -> List[str]:
    """Extract all digit sequences (PIN codes, street numbers, suite numbers)."""
    if not text:
        return []
    return RE_NUMBERS.findall(text)


def extract_significant_tokens(text: Optional[str], min_length: int = 3) -> Set[str]:
    """Extract non-trivial word tokens for inverted indexing / blocking."""
    if not text:
        return set()
    cleaned = RE_NON_ALPHANUM_BASIC.sub(" ", text.lower())
    tokens = {t for t in cleaned.split() if len(t) >= min_length}
    return tokens
