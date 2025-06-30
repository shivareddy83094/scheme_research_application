import streamlit as st
import configparser
import pickle
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Step 1: Load Groq API Key from config
config = configparser.ConfigParser()
config.read('.config')
groq_api_key = config['groq']['api_key']

# ✅ Use supported model from Groq
llm = ChatOpenAI(
    model="llama3-70b-8192",
    temperature=0,
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1"
)

# ✅ Use local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Scheme Research Tool")
st.title("📄 Government Scheme Research Tool")

# Step 2: Input URLs
st.sidebar.header("Step 1: Enter URLs")
urls = st.sidebar.text_area("Paste scheme article URLs (one per line)").splitlines()
process = st.sidebar.button("Process URLs")

if process and urls:
    st.info("🔄 Loading articles from URLs...")
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    st.info("🧠 Splitting text and creating embeddings...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    texts = []
    metadatas = []
    for i, doc in enumerate(docs):
        content = doc.page_content
        if isinstance(content, str) and content.strip():
            texts.append(content)
            metadatas.append(doc.metadata)
        else:
            st.warning(f"⚠️ Skipped invalid doc at index {i}")

    if texts:
        db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        with open("faiss_store_openai.pkl", "wb") as f:
            pickle.dump(db, f)
        st.success("✅ Documents processed and stored!")
    else:
        st.error("❌ No valid content found. Please check the URLs.")

# Step 3: Ask questions
st.header("❓ Ask Questions from the Article")
query = st.text_input("Enter your question here")

if query:
    st.info("🤔 Searching for the answer...")
    try:
        with open("faiss_store_openai.pkl", "rb") as f:
            db = pickle.load(f)

        docs = db.similarity_search(query)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)

        st.subheader("📌 Answer:")
        st.write(answer)

        st.subheader("🔗 Source URLs:")
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            st.markdown(f"- [{source}]({source})")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
