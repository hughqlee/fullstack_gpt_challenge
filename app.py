import os
import streamlit as st

from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading website...")
def load_website(api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )
    loader = SitemapLoader(
        web_path="https://developers.cloudflare.com/sitemap-0.xml",
        filter_urls=[
            "https://developers.cloudflare.com/ai-gateway",
            "https://developers.cloudflare.com/vectorize",
            "https://developers.cloudflare.com/workers-ai",
        ]
    )
    docs = loader.load_and_split(splitter)
    return FAISS.from_documents(
        docs,
        OpenAIEmbeddings(openai_api_key=api_key),
    ).as_retriever()

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🤖",
)

st.title("SiteGPT")

with st.sidebar:
    st.markdown("[🔗 Github Repo](https://github.com/hughqlee/fullstack_gpt_challenge)")
    openai_api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY"), type="password")
    
# API Key가 입력되지 않으면 진행 불가 안내
if not openai_api_key:
    st.warning("OpenAI API Key를 입력하세요.")
    st.stop()

# LLM 초기화 (API 키가 있을 때만)
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openai_api_key
)

# 웹사이트 로드
retriever = load_website(openai_api_key)

# RAG 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful assistant that answers questions based on the provided context from Cloudflare documentation.
    
    Context: {context}
    
    Please answer the question based on the provided context. If the information is not available in the context, please say so.
    """),
    ("human", "{question}")
])

# RAG 체인 구성
chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)

st.markdown("### Cloudflare AI 문서에 대해 질문해보세요!")

# 샘플 질문들을 버튼으로 제공
st.markdown("**샘플 질문들:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 LLaMA-2 모델 가격"):
        sample_question = "What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
        st.session_state.question = sample_question

with col2:
    if st.button("🚪 AI Gateway 기능"):
        sample_question = "What can I do with Cloudflare's AI Gateway?"
        st.session_state.question = sample_question

with col3:
    if st.button("📊 Vectorize 인덱스 수"):
        sample_question = "How many indexes can a single account have in Vectorize?"
        st.session_state.question = sample_question

# 질문 입력
question = st.text_input(
    "질문을 입력하세요:",
    value=st.session_state.get("question", ""),
    placeholder="예: What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?"
)

if question:
    with st.spinner("답변을 생성하고 있습니다..."):
        try:
            # RAG 체인 실행
            response = chain.invoke(question)
            
            st.markdown("### 답변:")
            st.markdown(response.content)
            
            # 관련 문서 표시 (선택사항)
            with st.expander("참고한 문서들 보기"):
                docs = retriever.get_relevant_documents(question)
                for i, doc in enumerate(docs[:3]):  # 상위 3개 문서만 표시
                    st.markdown(f"**문서 {i+1}:**")
                    st.markdown(f"출처: {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(f"내용: {doc.page_content[:500]}...")
                    st.markdown("---")
                    
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")

# 세션 상태 초기화
if "question" not in st.session_state:
    st.session_state.question = ""
        


