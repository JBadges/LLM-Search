import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_PATH = os.getenv('DB_PATH')
    INDEX_DIRECTORIES = os.getenv('INDEX_DIRECTORIES', '').split(',')
    RE_INDEX_INTERVAL_SECONDS = int(os.getenv('RE_INDEX_INTERVAL_SECONDS'))