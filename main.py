import streamlit as st
from openai import OpenAI
from docling.document_converter import DocumentConverter
from io import BytesIO
from docling.datamodel.base_models import DocumentStream
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker


st.title("Report Analyzer")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url="https://chat-ai.academiccloud.de/v1"
)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "meta-llama-3.1-8b-instruct"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Wrap the uploaded binary stream with BytesIO
    buf = BytesIO(uploaded_file.read())

    # Create a DocumentStream object
    source = DocumentStream(name=uploaded_file.name, stream=buf)

    # Run the converter
    converter = DocumentConverter()
    result = converter.convert(source)

    st.write(result.document.export_to_markdown())

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    # This block is now INSIDE the "if prompt:"
    with st.chat_message("assistant"):
        # Only make the API call if there are messages (specifically, the user's new message)
        if st.session_state.messages: # Check if messages list is not empty
            try:
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error calling API: {e}")
                # Optionally remove the last user message if the API call failed,
                # or handle the error in a way that makes sense for your app.
                # For now, we'll just display an error.
        else:
            # This case should ideally not be reached if the logic is correct,
            # as we've just added a user message.
            # However, 