import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the application."""
    DB_PATH = os.getenv('DB_PATH', 'persist/embeddings.db')
    INDEX_DIRECTORIES = [dir for dir in os.getenv('INDEX_DIRECTORIES', '').split(',') if dir]

    @staticmethod
    def validate() -> None:
        """Validate the configuration settings."""
        assert Config.DB_PATH, "DB_PATH is not set in the environment variables."
        assert Config.INDEX_DIRECTORIES, "INDEX_DIRECTORIES is not set in the environment variables."

Config.validate()
