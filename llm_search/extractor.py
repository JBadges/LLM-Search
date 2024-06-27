import mimetypes
import logging

logger = logging.getLogger(__name__)

EXCLUDE_DIRS = [
    'site-packages', 'vendor', 'get-pip', 'node_modules', 'dist',
    'build', 'venv', 'env', 'target', 'docker', 'tmp'
]

def is_text_file(file_path):
    for ending in ['.css', '.xml', '.json', '.yaml', '.yml', '.toml']:
        if file_path.endswith(ending):
            return False
    if file_path.endswith('.md'):
        return True
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type and mime_type.startswith('text')

def extract_text_from_file(file_path):
    for exclude_dir in EXCLUDE_DIRS:
        if exclude_dir in file_path:
            return ""

    if is_text_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return content
        except (UnicodeDecodeError, IOError) as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    return ""
