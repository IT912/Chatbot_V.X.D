import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.cache import InMemoryCache
import langchain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI  

# ----------------------------- CẤU HÌNH BAN ĐẦU -----------------------------
load_dotenv()
st.set_page_config(page_title="Chatbot Tri Thức Riêng V.X.D", page_icon="🤖")
st.title("🤖 Chatbot Tri Thức Riêng - V.X.D (Gemini)")

# Cache để tiết kiệm token
langchain.llm_cache = InMemoryCache()

# ----------------------------- KHỞI TẠO EMBEDDINGS -----------------------------
if "embedding" not in st.session_state:
    st.session_state.embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

embedding = st.session_state.embedding
# ----------------------------- THAM SỐ -----------------------------
FAISS_PATH = "faiss_index"
SPLITS_PATH = "splits.pkl"
REBUILD = False  # Đặt True nếu tạo lại index

# ----------------------------- TẢI / TẠO VECTORSTORE VÀ SPLITS -----------------------------
if not REBUILD:
    # Load FAISS và splits
    if "vectorstore" not in st.session_state:
       st.session_state.vectorstore = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    vectorstore = st.session_state.vectorstore
    if "splits" not in st.session_state:
        with open(SPLITS_PATH, "rb") as f:
            st.session_state.splits = pickle.load(f)

    splits = st.session_state.splits
else:
    print(">>> Đang tải tài liệu PDF...")
    loader = DirectoryLoader(
        path="./paper",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    print(f"    Đã load {len(docs)} trang.")

    # Semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    add_start_index=True,
    strip_whitespace=True
)
    splits = text_splitter.split_documents(docs)
    print(f"    Đã chia thành {len(splits)} đoạn văn bản.")

    # Tạo FAISS
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE
    )
    vectorstore.save_local(FAISS_PATH)
    print(f">>> Đã lưu FAISS vào '{FAISS_PATH}'.")

    # Lưu splits
    with open(SPLITS_PATH, "wb") as f:
        pickle.dump(splits, f)
    print(f">>> Đã lưu splits vào '{SPLITS_PATH}'.")

# ----------------------------- RETRIEVERS -----------------------------

if "ensemble_retriever" not in st.session_state:

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    st.session_state.ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )

ensemble_retriever = st.session_state.ensemble_retriever
# ----------------------------- PROMPT -----------------------------
template = (
    "You are a helpful assistant for a private knowledge base.\n"
    "You can both answer knowledge-based questions using provided documents "
    "and engage in simple daily conversation.\n\n"

    "BEHAVIOR RULES:\n"
    "1) If the question requires knowledge from the documents, answer using ONLY the provided context.\n"
    "2) When answering from the context, include citations in the format (source:page).\n"
    "3) Do NOT use external knowledge, assumptions, or web information for knowledge-based questions.\n"
    "4) If the answer to a knowledge-based question is not found in the context, reply exactly:\n"
    "   \"I’m sorry, I cannot answer that based on the provided documents.\"\n"
    "5) If the user is engaging in casual conversation (e.g., greetings, small talk, simple daily questions),\n"
    "   respond naturally, politely, and briefly.\n"
    "6) Keep responses clear, professional, and helpful.\n\n"

    "CONTEXT:\n"
    "{context}\n\n"

    "QUESTION:\n"
    "{question}\n\n"

    "ANSWER:"
)
prompt_template = ChatPromptTemplate.from_template(template)

# ----------------------------- LLM (GEMINI 2.5 FLASH) -----------------------------
if "qa_chain" not in st.session_state:
    print (">>Đang khởi tạo bộ não GEMINI và bộ nhớ...")
    
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1,
        streaming=True
    )

    #LLM riêng để tóm tắt câu hỏi dựa trên lịch sử
    condense_llm=ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
    #bộ nhớ
    memory=ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=2
    )

    st.session_state.memory = memory

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condense_llm,
        retriever=ensemble_retriever, # Đảm bảo biến này đã được khai báo ở phần cache phía trên
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=False
    )

# ----------------------------- GIAO DIỆN STREAMLIT -----------------------------
if "chain" not in st.session_state:
    st.session_state.chain = st.session_state.qa_chain
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lịch sử chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
if user_input := st.chat_input("Hỏi tôi về tài liệu của bạn..."):
    # Thêm tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Trả lời
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        class GeminiStreamCallback(BaseCallbackHandler):
            def __init__(self, placeholder):
                self.placeholder = placeholder
                self.text = ""

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.text += token
                # Dùng write thay vì markdown để tránh lỗi DOM
                self.placeholder.write(self.text)

        stream_handler = GeminiStreamCallback(message_placeholder)

        response = st.session_state.chain.invoke(
            {"question": user_input},
            callbacks=[stream_handler]
        )
        answer = response["answer"]
        # Render lại bằng markdown (tùy chọn)
        message_placeholder.markdown(answer)
        full_response = answer

        st.session_state.messages.append({"role": "assistant", "content": full_response})