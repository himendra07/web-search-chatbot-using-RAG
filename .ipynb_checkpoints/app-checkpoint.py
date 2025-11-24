import streamlit as st
import asyncio
import traceback
from extract_queries import extract_queries
from web_scraper import fetch_web_pages
from db_operations import get_embedding_function
from prompt_generator import generate_prompt
from config import MODEL_NAME

# langchain_ollama chat wrapper (may raise if ollama client not installed)
# using try/except so the app still runs and shows sensible errors
try:
    from langchain_ollama.chat_models import ChatOllama
    import ollama
except Exception:
    ChatOllama = None
    ollama = None

st.set_page_config(page_title="RAGify", page_icon="🤖")
st.title("RAGify — Robust")

# Helper: get safe model list (fallback to config.MODEL_NAME)
def safe_ollama_models():
    try:
        if ollama is None:
            return [MODEL_NAME]
        models = [m.model for m in ollama.list().models if getattr(m, "model", None)]
        # filter out embedding-only model name if present
        models = [m for m in models if "nomic-embed-text" not in m]
        if not models:
            return [MODEL_NAME]
        return models
    except Exception as e:
        # don't crash the app if Ollama list fails
        print("Warning: could not list ollama models:", e)
        return [MODEL_NAME]

# Sidebar controls
with st.sidebar:
    model_options = safe_ollama_models()
    llm_model = st.selectbox(label="Select llm model", options=model_options, index=0)
    search_engine = st.selectbox(label="Select search engine", options=["duckduckgo", "google"])
    n_results = st.number_input(label="Select number of web results", min_value=1, max_value=8, value=4)

# Ensure we always have a valid model string
if not llm_model:
    llm_model = MODEL_NAME
llm_model = str(llm_model)

# session messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# generator for streaming LLM output chunks (keeps same shape as original)
def chunk_generator(llm, query):
    for chunk in llm.stream(query):
        yield chunk

# Small helper to run async coroutines from Streamlit safely
def run_async(coro):
    """
    Try to run an async coroutine; Streamlit runs in a normal Python process
    so asyncio.run is usually fine. We attempt asyncio.run and fallback if needed.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # fallback for environments with running loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

# Main chat input handling
if usr_msg := st.chat_input("Ask me anything (I will search the web)"):
    # append user message to session
    st.session_state.messages.append({"role": "user", "content": usr_msg})
    st.chat_message("user").write(usr_msg)

    # assistant processing
    with st.chat_message("assistant"):
        try:
            with st.spinner("Extracting queries..."):
                # defensive: extract_queries should always return a list
                try:
                    queries = extract_queries(usr_msg, model=llm_model)
                except TypeError:
                    # fallback in case extract_queries signature differs
                    queries = extract_queries(usr_msg)
                if not queries or not isinstance(queries, list):
                    queries = [usr_msg]
                st.write(f"Search queries: {queries}")

            with st.spinner("Searching on the web..."):
                # run web scraping (async) and ignore failures (function is defensive)
                try:
                    run_async(fetch_web_pages(queries, n_results, provider=search_engine))
                except Exception as e:
                    # log and continue — generate_prompt will handle missing docs
                    st.warning("Web fetching failed (continuing): " + str(e))
                    print("fetch_web_pages error:", traceback.format_exc())

                # get embedding function (may raise if embedding model missing)
                try:
                    embedding_function = get_embedding_function()
                except Exception as e:
                    st.error("Embedding function failed: " + str(e))
                    # try to continue with a None placeholder (generate_prompt should handle it)
                    embedding_function = None

            with st.spinner("Extracting info from webpages..."):
                try:
                    prompt, sources = generate_prompt(usr_msg, embedding_function)
                except Exception as e:
                    # If prompt generation fails, show helpful message and continue with fallback
                    st.warning("Could not generate prompt using downloaded pages. Falling back to direct question.")
                    print("generate_prompt error:", traceback.format_exc())
                    prompt = usr_msg
                    sources = "[]"

            with st.spinner("Generating response..."):
                # ensure ChatOllama is available
                if ChatOllama is None:
                    st.error("ChatOllama client not available. Make sure 'langchain_ollama' is installed and 'ollama' package is importable.")
                    # show prompt and sources so user can debug
                    st.write("Prompt sent to model (preview):")
                    st.code(prompt if isinstance(prompt, str) else str(prompt)[:1000])
                    st.write("Sources:", sources)
                else:
                    try:
                        llm = ChatOllama(model=llm_model, stream=True)
                        stream_data = chunk_generator(llm, prompt)
                        # stream to the UI
                        st.write_stream(stream_data)
                        st.write(sources)
                    except Exception as e:
                        # show the error and fallback explanation
                        st.error("LLM generation failed: " + str(e))
                        print("ChatOllama error:", traceback.format_exc())
                        st.write("Model attempted:", llm_model)
                        st.write("Prompt preview:", prompt[:1000] if isinstance(prompt, str) else str(prompt))
        except Exception as outer:
            st.error("Unexpected error: " + str(outer))
            print("Unexpected error in assistant flow:", traceback.format_exc())
