import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import streamlit as st
from openai import OpenAI

from config import (
    BASE_URL, KB_DIR, LOGS_DIR, MAX_CONTEXT_CHARS,
    MODEL, OPENROUTER_API_KEY, PDFS_DIR, SYSTEM_PROMPT,
)
from build_kb import process_document
from storage import GitHubStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Init
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
# KB helpers
# ---------------------------------------------------------------------------

def get_kb_files() -> list[Path]:
    return sorted(Path(KB_DIR).glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)


def load_kb_context() -> str:
    kb_path  = Path(KB_DIR)
    glossary = kb_path / "glossary.md"
    cutoff   = datetime.now() - timedelta(days=30)
    md_files = sorted(
        [f for f in kb_path.glob("*.md") if f.name != "glossary.md"],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    parts = []
    if glossary.exists():
        parts.append(f"=== glossary.md ===\n{glossary.read_text(encoding='utf-8')}")
    for f in md_files:
        parts.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}")
    full = "\n\n".join(parts)
    if len(full) <= MAX_CONTEXT_CHARS:
        return full
    trimmed = parts[:1]
    for f in md_files:
        if datetime.fromtimestamp(f.stat().st_mtime) >= cutoff:
            trimmed.append(f"=== {f.name} ===\n{f.read_text(encoding='utf-8')}")
    return "\n\n".join(trimmed)


def log_event(question: str, answer: str):
    log_path = Path(LOGS_DIR) / "events.jsonl"
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "question": question, "answer": answer}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def kb_size_chars() -> int:
    return sum(len(f.read_text(encoding="utf-8")) for f in Path(KB_DIR).glob("*.md"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(storage: GitHubStorage):
    with st.sidebar:
        st.markdown("## ⚙️ Управление базой")
        st.divider()

        # --- Status ---
        files    = get_kb_files()
        n_docs   = len(files)
        size_kb  = kb_size_chars() // 1000
        pct      = min(int(kb_size_chars() / MAX_CONTEXT_CHARS * 100), 100)

        col1, col2 = st.columns(2)
        col1.metric("Документов", n_docs)
        col2.metric("Размер", f"{size_kb} КБ")

        st.caption(f"Использование контекста: {pct}%")
        st.progress(pct / 100)
        st.divider()

        # --- Upload ---
        st.markdown("**Добавить документ**")
        uploaded = st.file_uploader("PDF-файл", type=["pdf"], label_visibility="collapsed")
        if st.button("Обработать и добавить в базу", use_container_width=True,
                     disabled=uploaded is None, type="primary"):
            save_path = Path(PDFS_DIR) / uploaded.name
            save_path.write_bytes(uploaded.read())
            with st.spinner(f"Обрабатываю «{uploaded.name}»…"):
                md_name = process_document(str(save_path), storage)
            if md_name:
                st.success(f"Добавлено: {md_name}")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Ошибка обработки — проверьте логи.")

        st.divider()

        # --- Sync status ---
        if st.button("🔄 Синхронизировать с GitHub", use_container_width=True):
            with st.spinner("Синхронизирую…"):
                n = storage.pull_kb()
            st.success(f"Скачано файлов: {n}")
            st.rerun()

        st.caption("Хранилище: Gold-VP/production-kb-storage")


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def render_chat(client: OpenAI):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Header row
    col1, col2 = st.columns([6, 1])
    col1.markdown("### 💬 Ассистент предприятия")
    if col2.button("Очистить", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Welcome hint
    if not st.session_state.messages:
        st.info("👋 Задайте вопрос по документации предприятия. "
                "Ассистент отвечает строго по базе знаний и указывает источник.")

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Задайте вопрос по документации…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        kb_context = load_kb_context()
        system_msg = SYSTEM_PROMPT + "\n\n--- БАЗА ЗНАНИЙ ---\n" + kb_context

        with st.chat_message("assistant"):
            placeholder = st.empty()
            answer_parts = []
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        *st.session_state.messages,
                    ],
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    answer_parts.append(delta)
                    placeholder.markdown("".join(answer_parts) + "▌")
                answer = "".join(answer_parts)
                placeholder.markdown(answer)
            except Exception as e:
                logger.error("LLM error: %s", e)
                answer = f"❌ Ошибка модели: {e}"
                placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        log_event(user_input, answer)


# ---------------------------------------------------------------------------
# Tab 2: Knowledge Base browser
# ---------------------------------------------------------------------------

def render_kb():
    st.markdown("### 📚 База знаний")

    files = get_kb_files()
    if not files:
        st.info("База знаний пуста. Загрузите PDF через панель слева.")
        return

    search = st.text_input("🔍 Поиск по названию", placeholder="Введите название файла…")
    filtered = [f for f in files if search.lower() in f.name.lower()] if search else files

    st.caption(f"Найдено: {len(filtered)} из {len(files)} документов")
    st.divider()

    for f in filtered:
        mtime    = datetime.fromtimestamp(f.stat().st_mtime)
        size_kb  = f.stat().st_size // 1000
        content  = f.read_text(encoding="utf-8")
        # Extract first meaningful line as subtitle
        lines    = [l.strip() for l in content.split("\n") if l.strip() and not l.startswith("#")]
        subtitle = lines[0][:120] + "…" if lines else ""

        with st.expander(f"📄 {f.stem}  —  {mtime.strftime('%d.%m.%Y')}  ·  {size_kb} КБ"):
            if subtitle:
                st.caption(subtitle)
            st.markdown(content)


# ---------------------------------------------------------------------------
# Tab 3: Incidents
# ---------------------------------------------------------------------------

def render_incidents(storage: GitHubStorage):
    st.markdown("### 🚨 Журнал инцидентов")

    # Show existing incidents
    incident_path = Path(KB_DIR) / "incidents.md"
    if incident_path.exists():
        content = incident_path.read_text(encoding="utf-8").strip()
        incidents = [b.strip() for b in content.split("---") if b.strip()]
        if incidents:
            st.caption(f"Всего записей: {len(incidents)}")
            for inc in reversed(incidents):
                st.markdown(inc)
                st.divider()

    # Form
    st.markdown("#### Новый инцидент")
    with st.form("incident_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        dt          = col1.text_input("Дата / время", value=datetime.now().strftime("%Y-%m-%d %H:%M"))
        responsible = col2.text_input("Ответственный")
        equipment   = st.text_input("Оборудование / участок")
        description = st.text_area("Описание инцидента", height=100)
        measures    = st.text_area("Принятые меры", height=80)
        submitted   = st.form_submit_button("💾 Сохранить инцидент", use_container_width=True,
                                            type="primary")

    if submitted:
        if not description.strip():
            st.warning("Заполните описание инцидента.")
        else:
            entry = (
                f"\n\n---\n"
                f"**Дата/время:** {dt}  \n"
                f"**Оборудование:** {equipment}  \n"
                f"**Описание:** {description}  \n"
                f"**Меры:** {measures}  \n"
                f"**Ответственный:** {responsible}  \n"
            )
            with incident_path.open("a", encoding="utf-8") as f_out:
                f_out.write(entry)
            storage.push_file(str(incident_path), incident_path.read_text(encoding="utf-8"))
            st.success("✅ Инцидент сохранён и отправлен в базу знаний.")
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _ensure_dirs()
    st.set_page_config(
        page_title="База знаний предприятия",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    storage = get_storage()
    client  = get_llm_client()

    render_sidebar(storage)

    tab_chat, tab_kb, tab_inc = st.tabs(["💬 Ассистент", "📚 База знаний", "🚨 Инциденты"])

    with tab_chat:
        render_chat(client)

    with tab_kb:
        render_kb()

    with tab_inc:
        render_incidents(storage)


if __name__ == "__main__":
    main()
