import streamlit as st
import working_backend_final
import tempfile
import os
import json
import io
from PIL import Image
from pdf2image import convert_from_bytes
import streamlit_pdf_viewer as pdf_viewer
from custom_prompt import custom_prompt
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Chat with PDF", page_icon=":material/temp_preferences_eco:")
st.title(":green[**Eco Report RAG Chat** :material/temp_preferences_eco:]")
st.markdown("""*Please do not upload any sensitive or confidential documents, or provide any personal information.*""")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "past_chats" not in st.session_state:
    st.session_state.past_chats = []
if "pdf_wide" not in st.session_state:
    st.session_state["pdf_wide"] = False

# --- Utility Functions ---
def render_pdf_thumbnail(pdf_bytes, preview_width=180):
    """Generates a PNG thumbnail from the first page of a PDF."""
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        if images:
            img = images[0]
            width, height = img.size
            aspect_ratio = width / height if height != 0 else 1
            preview_height = int(preview_width / aspect_ratio)
            img_resized = img.resize((preview_width, preview_height))
            buf = io.BytesIO()
            img_resized.save(buf, format="PNG")
            buf.seek(0)
            return buf
    except Exception as e:
        st.sidebar.error(f"Thumbnail error: {e}")
        return None
    return None

@st.cache_data(show_spinner=False)
def extract_json_for_download(_llm, _retriever, pdf_name):
    """Cached function to extract JSON data from the PDF."""
    return working_backend_final.extract_info_as_json(_llm, _retriever, pdf_name)

@st.cache_data(show_spinner=False)
def convert_json_for_download(data):
    """Cached function to convert JSON data to bytes for download."""
    return json.dumps(data, indent=2).encode("utf-8")

# --- UI Functions ---
def display_past_chats():
    """Renders the list of past chats in the sidebar."""
    st.sidebar.title("Past Chats")
    if st.session_state.past_chats:
        # Display chats in reverse chronological order
        for i, chat in enumerate(reversed(st.session_state.past_chats)):
            idx = len(st.session_state.past_chats) - 1 - i
            pdf_name = chat.get("pdf_name", f"Chat {idx + 1}")
            
            # Display PDF thumbnail
            if chat.get("pdf_bytes"):
                buf = render_pdf_thumbnail(chat["pdf_bytes"])
                if buf:
                    st.sidebar.image(buf, caption=pdf_name)

            # Button to load the past chat
            if st.sidebar.button(f"Load: {pdf_name}", key=f"past_chat_{idx}", use_container_width=True):
                # Save the current chat before switching
                save_current_chat()
                # Restore the selected chat's state
                restore_chat_state(chat)
                st.rerun()
    else:
        st.sidebar.info("No past chats yet.")

def new_chat():
    """
    Handles the logic for starting a new chat session.
    Saves the current chat and clears the session state for a fresh start.
    """
    save_current_chat()
    
    # --- FIX: Clear all relevant keys for a new session ---
    # It's crucial to also clear 'memory' and 'qa_chain' to prevent
    # the LLM from referencing old conversations.
    keys_to_clear = [
        "messages", "retriever", "llm", "pdf_path", "pdf_name", 
        "pdf_file", "pdf_bytes", "index_path", "memory", "qa_chain"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # st.rerun() will stop the current script run and start from the top,
    # effectively resetting the UI to its initial state (e.g., showing the file uploader).
    st.rerun()

def save_current_chat():
    """Saves the current chat state to the past_chats list."""
    if st.session_state.get("messages") and st.session_state.get("pdf_bytes"):
        current_pdf_name = st.session_state.get("pdf_name")
        chat_data = {
            "messages": st.session_state.get("messages", []).copy(),
            "pdf_bytes": st.session_state.get("pdf_bytes"),
            "pdf_name": current_pdf_name,
            "index_path": st.session_state.get("index_path"),
            "pdf_path": st.session_state.get("pdf_path"),
            "retriever": st.session_state.get("retriever"),
            "llm": st.session_state.get("llm"),
            "pdf_file": st.session_state.get("pdf_file"),
        }
        
        # Check if this chat already exists in past_chats to update it
        found = False
        for i, past_chat in enumerate(st.session_state.past_chats):
            if past_chat.get("pdf_name") == current_pdf_name:
                st.session_state.past_chats[i] = chat_data
                found = True
                break
        if not found:
            st.session_state.past_chats.append(chat_data)

def restore_chat_state(chat_data):
    """Restores the session state from a selected past chat."""
    st.session_state["messages"] = chat_data.get("messages", []).copy()
    st.session_state["pdf_bytes"] = chat_data.get("pdf_bytes")
    st.session_state["pdf_name"] = chat_data.get("pdf_name")
    st.session_state["index_path"] = chat_data.get("index_path")
    st.session_state["pdf_path"] = chat_data.get("pdf_path")
    st.session_state["retriever"] = chat_data.get("retriever")
    st.session_state["llm"] = chat_data.get("llm")
    st.session_state["pdf_file"] = chat_data.get("pdf_file")
    
    # Also clear memory and qa_chain to force re-creation with restored state
    if "memory" in st.session_state:
        del st.session_state["memory"]
    if "qa_chain" in st.session_state:
        del st.session_state["qa_chain"]


def toggle_pdf_wide():
    """Toggles the PDF view between wide and narrow."""
    st.session_state["pdf_wide"] = not st.session_state["pdf_wide"]

def display_chat_history(messages):
    """Displays chat history in user/assistant pairs, newest first."""
    for message in reversed(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Main App Logic ---

# Sidebar
with st.sidebar:
    display_past_chats()
    if "pdf_path" in st.session_state:
        st.button("New Chat", on_click=new_chat, use_container_width=True, type="primary")

# Main content layout
if st.session_state["pdf_wide"]:
    col_widths = [3, 2]
    pdf_height = 1000
else:
    col_widths = [1, 3]
    pdf_height = 500

col1, col2 = st.columns(col_widths)

# PDF Viewer Column
with col1:
    if "pdf_path" not in st.session_state:
        uploaded_file = st.file_uploader("Upload a PDF report to begin", type=["pdf"])
        if uploaded_file:
            # Process newly uploaded file
            st.session_state["pdf_bytes"] = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(st.session_state["pdf_bytes"])
                st.session_state["pdf_path"] = tmp_file.name
            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["pdf_file"] = st.session_state["pdf_path"]
            st.rerun() # Rerun to move to the processing stage
    
    if st.session_state.get("pdf_bytes"):
        st.button(
            f"Zoom: {'Shrink' if st.session_state['pdf_wide'] else 'Expand'}",
            icon="➖" if st.session_state['pdf_wide'] else "➕",
            use_container_width=True,
            on_click=toggle_pdf_wide
        )
        file_size_mb = len(st.session_state["pdf_bytes"]) / (1024 * 1024)
        if file_size_mb > 47:
            st.warning(f"⚠️ PDF is {file_size_mb:.2f} MB. Browser embedding may be slow or fail.")
            st.image(render_pdf_thumbnail(st.session_state["pdf_bytes"]), use_container_width=True)
        else:
            pdf_viewer.pdf_viewer(st.session_state["pdf_bytes"], width="100%", height=pdf_height)


# Chat Column
with col2:
    # Knowledge Base Processing
    if "pdf_path" in st.session_state and "retriever" not in st.session_state:
        pdf_name = st.session_state["pdf_name"]
        pdf_path = st.session_state["pdf_path"]
        
        index_dir = "faiss_indexes"
        os.makedirs(index_dir, exist_ok=True)
        pdf_base = os.path.splitext(pdf_name)[0]
        index_path = os.path.join(index_dir, pdf_base)
        st.session_state["index_path"] = index_path

        with st.spinner(f"Processing '{pdf_name}' and building knowledge base..."):
            texts = working_backend_final.read_and_split_pdf(pdf_path, chunk_size=2000, chunk_overlap=200)
            store = working_backend_final.embed_and_store(texts, index_path=index_path)
            st.session_state["retriever"] = working_backend_final.set_retriever(store)
            st.session_state["llm"] = working_backend_final.set_llm()
            st.success("Knowledge base ready! You can now ask questions.")

    # Chat Interface
    if "retriever" in st.session_state and "llm" in st.session_state:
        # --- Chat memory and chain setup (re-initializes if cleared) ---
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer' # Explicitly set for clarity
            )
            # If memory is new, messages should be new too
            st.session_state.messages = []

        if "qa_chain" not in st.session_state:
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=st.session_state["llm"],
                retriever=st.session_state["retriever"],
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )

        # --- Display chat history ---
        display_chat_history(st.session_state.get("messages", []))
        
        # --- Chat input ---
        prompt = st.chat_input(f"Ask about {st.session_state.get('pdf_name', 'the report')}...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain({"question": prompt})
                    answer = response.get("answer", "Sorry, I could not find an answer.")
                    st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun() # Rerun to show the new messages immediately at the top

        # --- Download JSON Button ---
        st.divider()
        with st.spinner("Preparing JSON for download..."):
            extracted_data = extract_json_for_download(
                st.session_state["llm"], 
                st.session_state["retriever"], 
                st.session_state["pdf_name"]
            )
            json_bytes = convert_json_for_download(extracted_data)
        
        st.download_button(
            label="Download Extracted JSON",
            data=json_bytes,
            file_name=f"extracted_{st.session_state.get('pdf_name', 'report')}.json",
            mime="application/json",
            use_container_width=True,
            icon=":material/download:"
        )
    elif "pdf_path" in st.session_state:
         st.info("Processing your document...")
    else:
        st.info("Please upload a PDF file to start chatting.")
