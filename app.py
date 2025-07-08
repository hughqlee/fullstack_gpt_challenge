from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message = ""  # 매번 초기화
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./{file.name}"
    # 파일 확장자에 따라 저장 모드 결정
    if file.name.endswith(".txt"):
        with open(file_path, "w", encoding="utf-8") as f:
            # bytes → str 변환 필요
            if isinstance(file_content, bytes):
                file_content = file_content.decode("utf-8")
            f.write(file_content)
    else:
        with open(file_path, "wb") as f:
            f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state.messages:
        send_message(
            message["message"], 
            message["role"], 
            save=False
        )
        
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

st.set_page_config(page_title="DocGPT", page_icon=":book:")
st.title("DocGPT")
st.markdown(
    """
    Welcome!
    """
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ConversationBufferMemory 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

memory = st.session_state.memory

# 사이드바에 API Key 입력 및 Github 링크 추가
with st.sidebar:
    st.markdown("[🔗 Github Repo](https://github.com/hughqlee/fullstack_gpt_challenge)")  # 실제 주소로 변경 필요
    openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY"), type="password")
    file = st.file_uploader(
        "Upload a file", 
        type=["pdf", "docx", "txt"]
    )

# API Key가 입력되지 않으면 진행 불가 안내
if not openai_api_key:
    st.warning("OpenAI API Key를 입력하세요.")
    st.stop()

if file:
    retriever = embed_file(file)
    send_message("I'm ready!", "ai", save=False)
    paint_history()
    
    # ChatCallbackHandler 인스턴스 생성
    chat_handler = ChatCallbackHandler()
    
    # RAG 파이프라인 정의 (retriever가 사용 가능한 시점에서)
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up: {context}"),
        ("user", "Question:\n{question}"),
    ])
    
    map_llm = ChatOpenAI(
        temperature=0.1,
        openai_api_key=openai_api_key,
    )
    
    def map_and_prepare(inputs):
        question = inputs["question"]
        docs = retriever.get_relevant_documents(question)
        snippets = []
        for doc in docs:
            out = map_prompt | map_llm
            resp = out.invoke({"context": doc.page_content, "question": question})
            text = resp.content.strip()
            if text:
                snippets.append(text)
        combined_context = "\n\n".join(snippets)
        return {"context": combined_context, "input": question, "history": inputs["history"]}
    
    map_reduce_step = RunnablePassthrough.assign(history=lambda _: memory.load_memory_variables({})['history'])
    map_reduce_step = map_reduce_step | RunnableLambda(map_and_prepare)
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 제공된 문서 내용과 대화 기록을 바탕으로 질문에 답하는 도움이 되는 어시스턴트입니다. "
         "대화 기록을 참고하여 이전 질문들과 연관지어 답변하세요. "
         "문서에 없는 내용이거나 모르는 것은 '모르겠습니다'라고 답하세요."),
        ("assistant", "문서 내용:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ])
    
    final_llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[chat_handler],
        openai_api_key=openai_api_key,
    )
    
    message = st.chat_input("Ask me anything about the file")
    if message:
        send_message(message, "user")
        
        # 메모리 상태 디버깅
        current_memory = memory.load_memory_variables({})
        st.sidebar.write("🧠 Memory Debug:")
        st.sidebar.write(f"Current history length: {len(current_memory.get('history', []))}")
        if current_memory.get('history'):
            st.sidebar.write("Recent messages:")
            for i, msg in enumerate(current_memory['history'][-3:]):  # 최근 3개만 표시
                st.sidebar.write(f"{i}: {type(msg)} - {str(msg)[:50]}")
        
        chain = map_reduce_step | final_prompt | final_llm
        with st.chat_message("ai"):
            response = chain.invoke({"question": message})
            
            # 응답 디버깅
            st.sidebar.write(f"Response type: {type(response)}")
            st.sidebar.write(f"Response content: {str(response)[:100]}")
            
            # 메모리에 대화 저장
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            memory.save_context(
                {"input": message},
                {"output": response_text}
            )
            
            # 저장 후 메모리 상태 확인
            updated_memory = memory.load_memory_variables({})
            st.sidebar.write(f"After save - history length: {len(updated_memory.get('history', []))}")
else:
    st.session_state.messages = []
        