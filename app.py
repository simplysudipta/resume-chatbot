import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load API keys ---
@st.cache_data
def load_environment():
    """Load environment variables and validate API key"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please add OPENAI_API_KEY to your .env file.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key
    return True

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Sudipta Pal Resume Bot", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Environment ---
load_environment()

# --- Sidebar Information ---
with st.sidebar:
    st.header("📋 About This Bot")
    st.write("This AI assistant analyzes Sudipta Pal's resume and can answer questions about:")
    st.write("• Work experience")
    st.write("• Skills & expertise")
    st.write("• Education")
    st.write("• Projects")
    st.write("• Contact information")
    
    st.divider()
    st.header("💡 Sample Questions")
    sample_questions = [
        "What is Sudipta's current role?",
        "What programming languages does he know?",
        "What are his key achievements?",
        "How can I contact him?",
        "What projects has he worked on?"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}"):
            st.session_state.sample_question = q

# --- Main Page ---
st.title("🤖 Resume Q&A - Sudipta Pal")
st.markdown("### Ask me anything about Sudipta's professional background!")

# --- Load and Process Resume PDF ---
@st.cache_resource
def load_and_process_resume():
    """Load PDF, split into chunks, and create vector store"""
    try:
        file_path = "SudiptaPal.pdf"
        
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"⚠️ Resume file '{file_path}' not found! Please upload the PDF file.")
            st.stop()
        
        # Load PDF
        with st.spinner("📄 Loading resume..."):
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
            
        if not docs:
            st.error("❌ Could not extract text from the PDF file.")
            st.stop()
            
        # Split text into chunks
        with st.spinner("✂️ Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Increased for better context
                chunk_overlap=200,  # Increased overlap
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            
        # Create embeddings and vector store
        with st.spinner("🧠 Creating knowledge base..."):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve more relevant chunks
            )
            
        logger.info(f"Successfully processed {len(chunks)} chunks from resume")
        return retriever, len(docs)
        
    except Exception as e:
        st.error(f"❌ Error processing resume: {str(e)}")
        logger.error(f"Error in load_and_process_resume: {str(e)}")
        st.stop()

# --- Initialize QA Chain ---
@st.cache_resource
def initialize_qa_chain(_retriever):
    """Initialize the QA chain with custom prompt"""
    try:
        custom_prompt = PromptTemplate(
            template="""You are Sudipta Pal's professional AI assistant. Answer questions about his resume professionally and engagingly, as if speaking to potential employers or clients.

Key guidelines:
- Be conversational but professional
- Highlight relevant achievements and skills
- If information isn't in the resume, politely mention that and suggest contacting Sudipta directly
- Keep responses focused and relevant
- Use bullet points for lists when appropriate

Context from resume:
{context}

Question: {question}

Professional Response:""",
            input_variables=["context", "question"]
        )
        
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.3,  # Slightly more creative
            max_tokens=500
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=_retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True  # For debugging/transparency
        )
        
        return qa_chain
    
    except Exception as e:
        st.error(f"❌ Error initializing QA chain: {str(e)}")
        logger.error(f"Error in initialize_qa_chain: {str(e)}")
        st.stop()

# --- Load Resume and Initialize Chain ---
retriever, num_pages = load_and_process_resume()
qa_chain = initialize_qa_chain(retriever)

# Success message
st.success(f"✅ Resume loaded successfully! ({num_pages} pages processed)")

# --- Chat State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    welcome_msg = """👋 Hello! I'm Sudipta Pal's AI assistant. I can help you learn about his professional background, skills, and experience. 

Feel free to ask me anything about his resume - from technical skills to work experience to contact information!"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

# --- Display Chat History ---
for message in st.session_state.messages:
    # Handle both tuple format (old) and dict format (new)
    if isinstance(message, tuple):
        role, content = message
    else:
        role = message["role"]
        content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)

# --- Handle Sample Question from Sidebar ---
if hasattr(st.session_state, 'sample_question'):
    user_input = st.session_state.sample_question
    delattr(st.session_state, 'sample_question')
else:
    user_input = st.chat_input("💬 Ask about Sudipta's background...")

# --- Process User Input ---
if user_input:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing resume..."):
            try:
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]
                
                # Display answer
                st.markdown(answer)
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Optional: Show confidence/source info in expander
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("📚 Source Information", expanded=False):
                        st.write(f"Answer based on {len(response['source_documents'])} relevant sections from the resume.")
                
            except Exception as e:
                error_msg = f"❌ Sorry, I encountered an error while processing your question: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logger.error(f"Error processing query '{user_input}': {str(e)}")

# --- Footer ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🔄 Clear chat history")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.caption("📱 Total messages")
    st.write(f"{len(st.session_state.messages)} messages")

with col3:
    st.caption("🤖 Powered by")
    st.write("OpenAI GPT-4")

# --- Custom CSS for better appearance ---
st.markdown("""
<style>
.stApp > header {
    background-color: transparent;
}

.stApp {
    margin-top: -80px;
}

.main-header {
    padding: 1rem 0;
}

div[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)
