from pathlib import Path

# 工程根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# 数据目录和文件
DATA_DIR = BASE_DIR / "data"
ECU_700_PATH = DATA_DIR / "ECU-700_Series_Manual.md"
ECU_800_BASE_PATH = DATA_DIR / "ECU-800_Series_Base.md"
ECU_800_PLUS_PATH = DATA_DIR / "ECU-800_Series_Plus.md"
TEST_QUESTIONS_PATH = DATA_DIR / "test-questions.csv"

# RAG 参数
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

# 模型名称（先占位，后续你按实际环境换）
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-4o-mini"  # 或你本地/开源的聊天模型
