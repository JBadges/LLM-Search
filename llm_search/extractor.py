import mimetypes
import logging

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = [
    'site-packages', 'vendor', 'get-pip', 'node_modules', 'dist',
    'build', 'venv', 'env', 'target', 'docker', 'tmp'
]

def is_text_file(file_path: str) -> bool:
    """Determine if a file is a text file based on its extension or MIME type."""
    for ending in ['.css', '.xml', '.json', '.yaml', '.yml', '.toml']:
        if file_path.endswith(ending):
            return False
    if file_path.endswith('.md'):
        return True
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('text')

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file, returning an empty string if the file is not a text file."""
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in file_path:
            return ""

    if not is_text_file(file_path):
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except (UnicodeDecodeError, IOError) as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""
