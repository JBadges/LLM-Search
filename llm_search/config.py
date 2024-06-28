import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_PATH = os.getenv('DB_PATH', 'persist/embeddings.db')
    INDEX_DIRECTORIES = [dir for dir in os.getenv('INDEX_DIRECTORIES', '').split(',') if dir]
    RE_INDEX_INTERVAL_SECONDS = int(os.getenv('RE_INDEX_INTERVAL_SECONDS', 3600))  # Defaulting to 1 hour

    @staticmethod
    def validate():
        assert Config.DB_PATH, "DB_PATH is not set in the environment variables."
        assert Config.INDEX_DIRECTORIES, "INDEX_DIRECTORIES is not set in the environment variables."
        assert Config.RE_INDEX_INTERVAL_SECONDS > 0, "RE_INDEX_INTERVAL_SECONDS must be greater than 0."

Config.validate()
