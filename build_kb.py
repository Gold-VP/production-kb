import logging
from datetime import date
from pathlib import Path

from openai import OpenAI

from config import OPENROUTER_API_KEY, BASE_URL, MODEL, KB_DIR
from ocr import process_pdf
from storage import GitHubStorage

logger = logging.getLogger(__name__)

_client = OpenAI(
    api_key  = OPENROUTER_API_KEY,
    base_url = BASE_URL,
)

# ---------------------------------------------------------------------------

def build_article(text: str, filename: str) -> str:
    """Send extracted text to LLM and get a structured Markdown article."""
    today = date.today().isoformat()

    prompt = f"""Создай структурированную энциклопедическую статью на русском языке \
на основе следующего документа. Статья должна содержать разделы:

## Краткое описание
## Ключевые факты и нормативы
## Термины и определения
## Важные предупреждения и ограничения
## Связанные темы
## Метаданные
- Исходный файл: {filename}
- Дата обработки: {today}

Текст документа:
{text}"""

    try:
        response = _client.chat.completions.create(
            model    = MODEL,
            messages = [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error("build_article LLM error for %s: %s", filename, e)
        return ""


# ---------------------------------------------------------------------------

def process_document(pdf_path: str, storage: GitHubStorage) -> str:
    """Full pipeline: PDF → text → LLM article → save locally → push to GitHub.

    Returns the name of the created .md file, or empty string on failure.
    """
    pdf_path = Path(pdf_path)

    # 1. Extract text
    logger.info("Extracting text from %s", pdf_path.name)
    text = process_pdf(str(pdf_path))
    if not text.strip():
        logger.error("Empty text extracted from %s — aborting.", pdf_path.name)
        return ""

    # 2. Build structured article via LLM
    logger.info("Building KB article for %s", pdf_path.name)
    article = build_article(text, pdf_path.name)
    if not article.strip():
        logger.error("LLM returned empty article for %s — aborting.", pdf_path.name)
        return ""

    # 3. Save locally
    kb_dir   = Path(KB_DIR)
    kb_dir.mkdir(parents=True, exist_ok=True)
    md_name  = pdf_path.stem + ".md"
    md_path  = kb_dir / md_name
    md_path.write_text(article, encoding="utf-8")
    logger.info("Saved locally: %s", md_path)

    # 4. Push to GitHub
    storage.push_file(str(md_path), article)

    return md_name
