import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from openai import OpenAI

from config import (
    BASE_URL,
    KB_DIR,
    LOGS_DIR,
    MAX_CONTEXT_CHARS,
    MODEL,
    OPENROUTER_API_KEY,
    PDFS_DIR,
    SYSTEM_PROMPT,
)
from build_kb import process_document
from storage import GitHubStorage

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialisation (runs once per session)
# ---------------------------------------------------------------------------

def _ensure_dirs():
    for d in [KB_DIR, PDFS_DIR, LOGS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_storage() -> GitHubStorage:
    storage = GitHubStorage()
    n = storage.pull_kb()
    logger.info("Pulled %d KB files from GitHub.", n)
    return storage


@st.cache_resource
def get_llm_client() -> OpenAI:
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# KB context builder
# ---------------------------------------------------------------------------

def load_kb_context() -> str:
    """Load .md files from KB_DIR into a single context string.

    Order: glossary.md first, then the rest sorted by modification time (newest first).
    If total > MAX_CONTEXT_CHARS, keep only files modified in the last 30 days + glossary.md.
    """
    kb_path   = Path(KB_DIR)
    glossary  = kb_path / "glossary.md"
    cutoff    = datetime.now() - timedelta(days=30)

    md_files = sorted(
        [f for f in kb_path.glob("*.md") if f.name != "glossary.md"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    parts: list[str] = []

    # Always include glossary first
    if glossary.exists():
        parts.append(f"=== glossary.md ===\n{glossary.read_text(encoding='utf-8')}")

    for f in md_files:
        parts.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}")

    full_context = "\n\n".join(parts)

    if len(full_context) <= MAX_CONTEXT_CHARS:
        return full_context

    # Trim: keep glossary + files modified within 30 days
    trimmed: list[str] = []
    if glossary.exists():
        trimmed.append(f"=== glossary.md ===\n{glossary.read_text(encoding='utf-8')}")

    for f in md_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime >= cutoff:
            trimmed.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}")

    return "\n\n".join(trimmed)


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def log_event(question: str, answer: str):
    log_path = Path(LOGS_DIR) / "events.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question":  question,
        "answer":    answer,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Incident form
# ---------------------------------------------------------------------------

def incident_form(storage: GitHubStorage):
    st.subheader("Сообщить об инциденте")
    with st.form("incident_form"):
        dt          = st.text_input("Дата / время", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
        equipment   = st.text_input("Оборудование")
        description = st.text_area("Описание инцидента")
        measures    = st.text_area("Принятые меры")
        responsible = st.text_input("Ответственный")
        submitted   = st.form_submit_button("Сохранить")

    if submitted:
        incident_path = Path(KB_DIR) / "incidents.md"
        entry = (
            f"\n\n---\n"
            f"**Дата/время:** {dt}  \n"
            f"**Оборудование:** {equipment}  \n"
            f"**Описание:** {description}  \n"
            f"**Меры:** {measures}  \n"
            f"**Ответственный:** {responsible}  \n"
        )
        with incident_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        storage.push_file(str(incident_path), incident_path.read_text(encoding="utf-8"))
        st.success("Инцидент сохранён и отправлен в базу знаний.")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    _ensure_dirs()
    st.set_page_config(page_title="База знаний предприятия", layout="wide")

    storage = get_storage()
    client  = get_llm_client()

    # ---- Sidebar ----
    with st.sidebar:
        st.title("База знаний предприятия")
        st.divider()

        # --- PDF upload ---
        uploaded = st.file_uploader("Загрузить PDF-документ", type=["pdf"])
        if st.button("Обработать и добавить в базу", disabled=uploaded is None):
            if uploaded:
                save_path = Path(PDFS_DIR) / uploaded.name
                save_path.write_bytes(uploaded.read())
                with st.spinner(f"Обрабатываю «{uploaded.name}»..."):
                    md_name = process_document(str(save_path), storage)
                if md_name:
                    st.success(f"Добавлено: {md_name}")
                    # Invalidate KB context cache
                    st.cache_data.clear()
                else:
                    st.error("Не удалось обработать файл. Проверьте логи.")

        st.divider()

        # --- File list ---
        st.subheader("Файлы в базе знаний")
        files = storage.list_files()
        if files:
            for f in files:
                lm = f["last_modified"]
                date_str = lm.strftime("%d.%m.%Y") if lm else "—"
                st.markdown(f"- **{f['name']}** — {date_str}")
        else:
            local_mds = sorted(Path(KB_DIR).glob("*.md"))
            if local_mds:
                for f in local_mds:
                    st.markdown(f"- {f.name}")
            else:
                st.info("База знаний пуста.")

        st.divider()

        # --- Incident report ---
        if st.button("Сообщить об инциденте"):
            st.session_state["show_incident"] = not st.session_state.get("show_incident", False)

        if st.session_state.get("show_incident"):
            incident_form(storage)

    # ---- Main area ----
    st.title("Ассистент предприятия")

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Задайте вопрос по документации..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context
        kb_context = load_kb_context()
        system_with_context = (
            SYSTEM_PROMPT
            + "\n\n--- БАЗА ЗНАНИЙ ---\n"
            + kb_context
        )

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    response = client.chat.completions.create(
                        model    = MODEL,
                        messages = [
                            {"role": "system", "content": system_with_context},
                            *st.session_state.messages,
                        ],
                    )
                    answer = response.choices[0].message.content or "Нет ответа от модели."
                except Exception as e:
                    logger.error("LLM request failed: %s", e)
                    answer = f"Ошибка при обращении к модели: {e}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        log_event(user_input, answer)


if __name__ == "__main__":
    main()
