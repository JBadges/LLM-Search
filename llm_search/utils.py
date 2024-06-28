from typing import Optional


def safe_str_to_int(s: str) -> Optional[int]:
    """Try to cast a string to an integer. If it fails, return None."""
    try:
        return int(s)
    except ValueError:
        return None