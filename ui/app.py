"""Docs Assistant - Streamlit UI. Talks to the FastAPI backend over HTTP only.

Run locally (API must be running on :8000):
    streamlit run ui/app.py
"""
import streamlit as st

from api_client import (
    ApiError, DocsApiClient, LLMUnavailable, RateLimited,
    citation_label, escape_markdown, get_api_url,
)

st.set_page_config(page_title="Docs Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def get_client(api_url: str) -> DocsApiClient:
    """One HTTP client per API URL, reused across reruns (Streamlit reruns the script on every click)."""
    return DocsApiClient(api_url)


client = get_client(get_api_url())

# Chat history survives reruns while the browser tab stays open.
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- rendering ----------

def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(escape_markdown(msg["content"]))
            return

        kind = msg.get("kind", "answer")
        if kind == "error":
            st.error(msg["content"], icon="🚫")
        elif kind == "warning":
            st.warning(msg["content"], icon="⏳")
        elif kind == "not_found":
            st.info(f"{msg['content']}\n\nTry rephrasing, or upload a document that covers this topic.", icon="🔎")
        else:
            st.markdown(escape_markdown(msg["content"]))
            if msg["citations"]:
                st.caption("Cited: " + "  ".join(citation_label(s, p) for s, p in msg["citations"])
                           + (f"  ·  model: {msg['model']}" if msg.get("model") else ""))
            if msg["sources"]:
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"**{citation_label(s['source'], s['page'])}**  ·  score {s['score']:.3f}")
                        st.text(s["text"])


def ask(question: str) -> dict:
    """Call POST /ask and turn the result (or error) into a chat message."""
    try:
        r = client.ask(question)
    except RateLimited as exc:
        return {"role": "assistant", "kind": "warning", "content": exc.message}
    except LLMUnavailable as exc:
        return {"role": "assistant", "kind": "warning", "content": exc.message}
    except ApiError as exc:
        return {"role": "assistant", "kind": "error", "content": exc.message}
    if not r.found:
        return {"role": "assistant", "kind": "not_found", "content": r.answer}
    return {
        "role": "assistant", "kind": "answer", "content": r.answer, "model": r.model,
        "citations": r.citations,
        "sources": [{"source": s.source, "page": s.page, "score": s.score, "text": s.text} for s in r.sources],
    }


# ---------- sidebar: status, upload, document list ----------

with st.sidebar:
    st.title("📄 Docs Assistant")
    api_ok = client.health()
    if api_ok:
        st.success(f"API connected · {client.base_url}", icon="✅")
    else:
        st.error(f"API not reachable at {client.base_url}. Start it, then refresh this page.", icon="🔌")

    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
    if st.button("Upload & index", type="primary", disabled=uploaded is None or not api_ok,
                 width="stretch"):
        with st.spinner(f"Indexing {uploaded.name}... (the first upload can take ~20 s)"):
            try:
                result = client.upload(uploaded.name, uploaded.getvalue())
            except ApiError as exc:
                st.error(exc.message)
            else:
                show = {"indexed": st.success, "replaced": st.success,
                        "unchanged": st.info, "no_text": st.warning}[result.status]
                show(f"**{result.filename}** - {result.status}: {result.chunks} chunks "
                     f"from {result.pages_with_text} pages.\n\n{result.message}")

    st.subheader("Indexed files")
    if api_ok:
        try:
            documents = client.list_documents()
        except ApiError as exc:
            st.caption(exc.message)
        else:
            if documents:
                for name, chunks in documents:
                    st.markdown(f"- **{name}** · {chunks} chunks")
                st.caption(f"{len(documents)} file(s), {sum(c for _, c in documents)} chunks")
            else:
                st.caption("No documents yet - upload a PDF above.")

    st.divider()
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# ---------- main area: chat ----------

st.title("Ask your documents")
st.caption("Answers come only from your uploaded PDFs, with a [file, p.N] citation for every fact.")

for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input("Ask a question about your PDFs...", disabled=not api_ok):
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)
    with st.spinner("Searching your documents and asking the AI..."):
        reply = ask(question)
    st.session_state.messages.append(reply)
    render_message(reply)
