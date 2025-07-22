import os
import logging
import warnings
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI  # <-- updated here
from langchain.prompts import PromptTemplate

# Load API keys
load_dotenv()

# Suppress warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Eye Disease RAG Assistant", page_icon="🧑‍⚕️")
st.title("🧑‍⚕️ Eye Disease Assistant - RAG")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "memory" not in st.session_state:
    st.session_state.memory = None

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

# Initialize memory
if st.session_state.memory is None and st.session_state.vectorstore:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask about eye diseases...")

prompt_template = PromptTemplate.from_template("""
- You are an AI assistant specialized in Eye Diseases.

Tasks:

- Answer the question using only the following context.
- If user ask question that not related to eyes, politely decline to answer.
- Use the context to provide a helpful answer.
- If the user entered question is not clear or not understandable or if you need more data, you can ask questions from the user also.
- Greeting and basic things are normal, but focus on eye diseases.
- Give answers in user friendly way. (paragraphs, points or anything you can use to make it more readable)

Context:
{context}

Question: {question}
Answer:
""")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    try:
        llm = ChatOpenAI(
            openai_api_key=os.environ.get("OPENAI_API_KEY"),  # <-- updated here
            model_name="gpt-4o",  # or "gpt-4" if you have access
            temperature=0
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": prompt_template},
            return_source_documents=False,
            output_key="answer"
        )

        result = qa_chain.invoke({"question": query})
        answer = result["answer"]

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"❌ Error during query: {e}")
