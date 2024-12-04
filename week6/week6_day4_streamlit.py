import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import bs4
from langchain_teddynote import logging

# text = '텍스트를 작성합니다!'
# st.header(text, divider='rainbow')
# st.title(text)
# st.write(text)
# st.write('### 문장을 넣습니다')
# st.write('# 문장을 넣습니다')
# vocab_logits = {'나는': 0.3, '밥을': 0.2, '먹는다': 0.5}
# st.bar_chart(vocab_logits)
# prompt = st.chat_input('메세지를 입력하세요')

load_dotenv()

# GPT 모델 설정
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.2,
    openai_api_key = os.getenv('OPEN_API_KEY')
)

# 타이틀
st.title('뉴스 기반 대화형 챗봇')
st.markdown('뉴스 URL을 입력하면 해당 뉴스 내용을 기반으로 질문에 답변합니다.')

# 상태 관리 (데이터를 안전하게 저장하기 위해 불러와서 쓸 때는 문자열로 선언)
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)  # 새롭게 메모리 구성
if 'messages_displayed' not in st.session_state:
    st.session_state.messages_displayed = []

# 뉴스 로드
news_url = st.text_input('뉴스 URL 입력: ')
if st.button('뉴스 로드'):
    if not news_url:
        st.error('URL을 입력해주세요.')
    else:
        try:
            loader = WebBaseLoader(
                web_path=(news_url,),
                # 튜플 형식으로 입력 받아야 한다
                bs_kwargs=dict(
                    parse_only= bs4.SoupStrainer(
                        'div',
                        attrs={
                            'class': ['newsct_artcle _article_body', 'media_end_head_title']
                        }
                    )
                )
            )
            docs = loader.load()

            if not docs:
                st.error('뉴스 내용을 로드할 수 없습니다. URL을 확인해주세요.')
            else:
                st.success(f'문서를 성공적으로 로드했습니다. 문서 개수: {len(docs)}')
                
                # 문서 분할
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_texts = splitter.split_documents(docs)

                # 임베딩
                embeddings = OpenAIEmbeddings()

                # FAISS 벡터 저장소 생성
                vector_store = FAISS.from_documents(split_texts, embeddings)

                st.session_state.vector_store = vector_store

        except Exception as e:
            st.error(f'오류가 발생했습니다: {str(e)}')

prompt = st.chat_input('메세지를 입력하세요.')
if prompt:
    if st.session_state.vector_store is None:
        st.error('뉴스를 먼저 로드해주세요.')
    else:
        # 사용자 메세지 기록
        st.session_state.memory.chat_memory.add_user_message(prompt)

        try:
            retriever = st.session_state.vector_store.as_retriever()
            chain = ConversationalRetrievalChain.from_llm(
                llm = llm,
                retriever = retriever,
                memory = st.session_state.memory
            )

            # AI 응답 생성
            response = chain({'question': prompt})
            ai_response = response['answer']

            # AI 메세지 기록
            st.session_state.memory.chat_memory.add_ai_message(ai_response)

            # 메세지 표시
            st.session_state.messages_displayed.append({'role': 'user', 'content': prompt})
            st.session_state.messages_displayed.append({'role': 'assistant', 'content': ai_response})
        
        except Exception as e:
            st.error(f'오류가 발생했습니다: {str(e)}')

# 이전 대화를 표시
for message in st.session_state.messages_displayed:
    with st.chat_message(message['role']):
        st.write(message['content'])