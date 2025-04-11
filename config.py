import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
CHROMA_PATH = "./chroma_db"
PAPERS_DIR = "./results"

# Model settings
MODEL_NAME = "gpt-4o-mini"  # Using a smaller model for cost efficiency
