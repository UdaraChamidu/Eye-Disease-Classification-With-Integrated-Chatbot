import os
import logging
import warnings
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Suppress unnecessary warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Eye Disease RAG Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Eye Disease Assistant - RAG")

# Initialize session state variables
st.session_state.setdefault("mode", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("vectorstore", None)
st.session_state.setdefault("memory", None)

# Load vectorstore
if st.session_state.vectorstore is None:
    try:
        st.info("🔄 Loading prebuilt medical knowledge base...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        vectorstore = FAISS.load_local("vectorstore/eye_faiss", embeddings, allow_dangerous_deserialization=True)
        st.session_state.vectorstore = vectorstore
        st.success("✅ Vectorstore loaded!")
    except Exception as e:
        st.error(f"❌ Failed to load vectorstore: {e}")

# Load memory
if st.session_state.memory is None and st.session_state.vectorstore:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Mode selection and back navigation
if st.session_state.mode is not None:
    if st.button("🔙 Back to Mode Selection"):
        st.session_state.mode = None
        st.session_state.messages = []
        st.rerun()

if st.session_state.mode is None:
    st.subheader("Choose Input Mode")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🖼️ Image Input"):
            st.session_state.mode = "image"
            st.rerun()
    with col2:
        if st.button("💬 Text Input"):
            st.session_state.mode = "text"
            st.rerun()
    with col3:
        if st.button("🔀 Image + Text Input"):
            st.session_state.mode = "both"
            st.rerun()
    st.stop()

# Shared prompt
prompt_template = PromptTemplate.from_template("""
- You are an AI assistant specialized in Eye Diseases.

Tasks:
- Answer the question using only the following context.
- If the question is not about eye diseases, politely decline.
- Use the context to give a helpful, clear answer.
- If the question is unclear, ask the user for more info.
- Give answers in a friendly and easy-to-read way.

Context:
{context}

Question: {question}
Answer:
""")

# Shared function to create QA chain
def get_qa_chain():
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=False,
        output_key="answer"
    )

# 🖼️ Image-Only Mode
if st.session_state.mode == "image":
    st.subheader("🖼️ Image Input Mode")
    uploaded_file = st.file_uploader("Upload an Eye Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Eye Image", use_column_width=True)
        st.warning("🚧 Image-only model inference not implemented yet.")  # TODO

# 💬 Text-Only Mode
elif st.session_state.mode == "text":
    st.subheader("💬 Text Input Mode")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Describe your symptoms or ask a question...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Error during query: {e}")

# 🔀 Combined Mode
elif st.session_state.mode == "both":
    st.subheader("🔀 Combined Image + Text Input Mode")
    uploaded_file = st.file_uploader("Upload your Eye Image", type=["png", "jpg", "jpeg"])
    query = st.chat_input("Add text symptoms or questions here...")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded OCT Image", use_column_width=True)

    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        try:
            # TODO: Add image model integration logic if needed
            qa_chain = get_qa_chain()
            result = qa_chain.invoke({"question": query})
            answer = result["answer"]
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Error during query: {e}")
