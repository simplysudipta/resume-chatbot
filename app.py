import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# --- Load API keys ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Streamlit Page ---
st.set_page_config(page_title="Sudipta Pal Resume Bot", page_icon="🤖")
st.title("🤖 Resume Q&A - Sudipta Pal")
st.write("Ask questions about Sudipta Pal's resume!")

# --- Load Resume PDF ---
file_path = "SudiptaPal.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load_and_split()

# --- Split text into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# --- Create embeddings + retriever ---
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever()

# --- Custom Prompt ---
custom_prompt = PromptTemplate(
    template="""You are acting as Sudipta Pal. You are answering questions by analyzing his resume. 
    Be professional and engaging, as if talking to a potential client or future employer. 
    If you don't know the answer based on the context, politely suggest contacting via email.

    Context: {context}

    Question: {question}

    Detailed Answer:""",
    input_variables=["context", "question"]
)

# --- LLM (OpenAI GPT) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- User Input ---
user_input = st.chat_input("Ask something about Sudipta Pal's resume...")

if user_input:
    response = qa_chain.invoke({"query": user_input})
    answer = response["result"]

    # Save chat
    st.session_state["messages"].append(("user", user_input))
    st.session_state["messages"].append(("bot", answer))

# --- Display Messages ---
for role, msg in st.session_state["messages"]:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg)
