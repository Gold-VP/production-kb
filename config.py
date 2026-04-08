import os

# --- API ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GITHUB_TOKEN       = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO        = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH      = os.environ.get("GITHUB_BRANCH", "main")

# --- Model ---
MODEL    = "meta-llama/llama-3.3-70b-instruct:free"
BASE_URL = "https://openrouter.ai/api/v1"

# --- Limits ---
MAX_CONTEXT_CHARS = 100_000

# --- Directories ---
KB_DIR    = "./kb"
PDFS_DIR  = "./source-pdfs"
LOGS_DIR  = "./logs"

# --- System prompt ---
SYSTEM_PROMPT = (
    "Ты ассистент производственного предприятия. "
    "Отвечай строго на основе документации в контексте. "
    "Если информации нет — честно сообщи об этом. "
    "Указывай из какого документа взята информация. "
    "Отвечай на русском языке."
)
