from pathlib import Path

# Root Directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Data Directory and Files
DATA_DIR = BASE_DIR / "data"
ECU_700_PATH = DATA_DIR / "ECU-700_Series_Manual.md"
ECU_800_BASE_PATH = DATA_DIR / "ECU-800_Series_Base.md"
ECU_800_PLUS_PATH = DATA_DIR / "ECU-800_Series_Plus.md"
TEST_QUESTIONS_PATH = DATA_DIR / "test-questions.csv"

# RAG parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

# Model Name
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4o-mini"
