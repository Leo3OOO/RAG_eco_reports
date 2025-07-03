import streamlit as st
import working_backend_final
import tempfile
import os
import json
import streamlit_pdf_viewer as pdf_viewer

st.set_page_config(layout="wide")
st.title("Eco Report RAG Chat")
st.markdown("""Please do not upload any sensitive or confidential documents. or provide any personal information.""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload a PDF report", type=["pdf"])

def toggle_pdf_wide():
    st.session_state["pdf_wide"] = not st.session_state["pdf_wide"]

if "pdf_wide" not in st.session_state:
    st.session_state["pdf_wide"] = False


# Now set icon and col_widths based on the (possibly updated) state
if st.session_state["pdf_wide"]:
    icon = "➖"
    pdf_height = 1000
    col_widths = [3, 2]
else:
    icon = "➕"
    col_widths = [1, 3]
    pdf_height = 500

col1, col2 = st.columns([col_widths[0], col_widths[1]])


with col1:
# Use the callback for the button
    st.button(
        f"Zoom: {'smaller' if st.session_state['pdf_wide'] else 'bigger'}",
        icon=icon,
        use_container_width=True,
        on_click=toggle_pdf_wide
    )
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        pdf_viewer.pdf_viewer(pdf_bytes, width="100%", height=pdf_height)
        uploaded_file.seek(0)

with col2:
    if uploaded_file is not None and "retriever" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        st.session_state["pdf_path"] = pdf_path
        st.session_state["pdf_name"] = uploaded_file.name  # Store original filename
        st.success(f"Uploaded: {uploaded_file.name}")

        # Create a unique index path for this PDF
        index_dir = "faiss_indexes"
        os.makedirs(index_dir, exist_ok=True)
        pdf_base = os.path.splitext(uploaded_file.name)[0]
        index_path = os.path.join(index_dir, pdf_base)

        with st.spinner("Processing PDF and building knowledge base..."):
            # Use larger chunk_size to reduce number of chunks
            texts = working_backend_final.read_and_split_pdf(pdf_path, chunk_size=2000, chunk_overlap=200)
            store = working_backend_final.embed_and_store(texts, index_path=index_path)
            retriever = working_backend_final.set_retriever(store)
            llm = working_backend_final.set_llm()
            st.session_state["retriever"] = retriever
            st.session_state["llm"] = llm
            st.session_state["pdf_file"] = pdf_path  # Store the PDF path for later use
            st.success("Knowledge base ready! You can now ask questions.")
    # Utility functions for JSON extraction and conversion
    @st.cache_data(show_spinner=False)
    def extract_json_for_download(_llm, _retriever, pdf_file):
        return working_backend_final.extract_info_as_json(_llm, _retriever, pdf_file)

    @st.cache_data(show_spinner=False)
    def convert_json_for_download(data):
        return json.dumps(data, indent=2).encode("utf-8")

    # Only show chat input after retriever/llm are ready
    if "retriever" in st.session_state and "llm" in st.session_state:
        prompt = st.chat_input("Ask a question about the uploaded report...")
        if prompt:
            with st.spinner("Thinking..."):
                # Build chat history as context
                chat_history = ""
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        chat_history += f"User: {content}\n"
                    elif role == "assistant":
                        chat_history += f"Assistant: {content}\n"
                chat_history += f"User: {prompt}\nAssistant:"

                # Use RetrievalQA with chat history as context
                qa_chain = working_backend_final.RetrievalQA.from_chain_type(
                    llm=st.session_state["llm"],
                    retriever=st.session_state["retriever"]
                )
                answer = qa_chain.run(chat_history)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": answer})
        # Download JSON button (now works regardless of scope)
        llm = st.session_state["llm"]
        retriever = st.session_state["retriever"]
        pdf_file = st.session_state.get("pdf_file", "uploaded.pdf")
        pdf_name = st.session_state.get("pdf_name", "uploaded.pdf")
        with st.spinner("Extracting information and preparing JSON for download..."):
            extracted = extract_json_for_download(llm, retriever, pdf_name)
            json_bytes = convert_json_for_download(extracted)
        st.download_button(
            label="Download extracted JSON",
            data=json_bytes,
            file_name="extracted_info.json",
            mime="application/json",
            use_container_width=True,
            icon=":material/download:"
        )

    # Display chat history only once
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
