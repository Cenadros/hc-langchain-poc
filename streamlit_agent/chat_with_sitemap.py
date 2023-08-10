import os
import tempfile
import pinecone
import nest_asyncio
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter

nest_asyncio.apply()
pinecone.init(
    api_key="d7da79ed-8b66-4f1a-8d57-9c630b5c0412",
    environment="us-west1-gcp"
)

st.set_page_config(page_title="Chatea sobre el contenido de una web", page_icon="🌍")
st.title("🌍 Chatea sobre el contenido de una web")


@st.cache_resource(ttl="1h")
def remove_nav_and_header_elements(_content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = _content.find_all("nav")
    header_elements = _content.find_all("header")

    # Remove each 'nav' and 'header' element from the BeautifulSoup object
    for element in nav_elements + header_elements:
        element.decompose()

    return str(_content.get_text())

def check_index(index_name):
    if index_name not in pinecone.list_indexes():
        return False
    else:
        return True

def configure_retriever(openai_api_key, sitemap_url):
    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # create index name making a hash from sitemap_url
    index_name = "webchat-" + str(hash(sitemap_url))

    if check_index(index_name) == False or pinecone.Index(index_name).describe_index_stats().total_vector_count == 0:
        with st.spinner('Procesando URLs del Sitemap...'):
            # Read documents
            docs = []
            loader = SitemapLoader(
              web_path=sitemap_url,
              filter_urls=[
                  ".*/asistencia/enfermedades/cancer.*",
              ],
              parsing_function=remove_nav_and_header_elements,
            )
            loader.requests_per_second = 2
            loader.requests_kwargs = {"verify": False}
            docs.extend(loader.load())

            # Split documents
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            splits = text_splitter.split_documents(docs)

        with st.spinner('Eliminando índices previos...'):
            # Remove all indexes (until account is upgraded)
            indexes = pinecone.list_indexes()
            for idx, index in enumerate(indexes):
                pinecone.delete_index(index)

        with st.spinner('Creando indice...'):
            pinecone.create_index(index_name, metric="cosine", shards=1, dimension=1536, pod_type="p2.x1")

        with st.spinner('Insertando documentos en el índice...'):
            vectordb = Pinecone.from_documents(splits, embeddings, index_name=index_name)
    else:
        with st.spinner('Cargando indice existente...'):
            vectordb = Pinecone.from_existing_index(index_name, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Contexto de la respuesta")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Pregunta:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        self.container.write(f"**Información extraida de:**")
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**{source}**")
#             self.container.markdown(doc.page_content)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Por favor, introduce tu API Key de OpenAI para continuar.")
    st.stop()

# sitemap_url = st.sidebar.text_input("URL del sitemap")
# if not sitemap_url:
#     st.info("Por favor, introduce la URL del sitemap para continuar.")
#     st.stop()

retriever = configure_retriever(openai_api_key, "https://preclinic.runroom.dev/sitemap-dev.xml")

# Setup memory for contextual conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, max_tokens_limit=2000
)

if "messages" not in st.session_state or st.sidebar.button("Borrar historial de chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "¿En qué puedo ayudarte?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Preguntame cualquier cosa!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.session_state.messages.append({"role": "assistant", "content": response})
