import os
import time
import logging
import streamlit as st
import PyPDF2
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Use the recommended import for Chroma
#from langchain_community.vectorstores import Chroma  # Old import
import chromadb
import chromadb.config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
from typing import List, Dict, Any, Tuple, Optional

# Import utility functions
from utils import reset_chroma, verify_chroma_persistence

# Import OpenAI for embeddings
from langchain.embeddings.base import Embeddings
from openai import OpenAI

class CustomOpenAIEmbeddings(Embeddings):
    """Custom embeddings class that uses the direct OpenAI API."""
    
    def __init__(self, api_key=None, model="text-embedding-3-small", **kwargs):
        """Initialize with API key and model name.
        
        Note: We accept **kwargs to handle any deprecated parameters like 'proxies'
        but we don't use them with the new OpenAI client.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        # Only pass the api_key to the client, ignore other kwargs like 'proxies'
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized CustomOpenAIEmbeddings with model {model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of documents."""
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
            return embeddings
        except Exception as e:
            logger.error(f"Error in embed_documents: {str(e)}")
            # Fallback to random embeddings if OpenAI API fails
            dimension = 1536
            import random
            return [
                [random.uniform(-1, 1) for _ in range(dimension)]
                for _ in texts
            ]
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error in embed_query: {str(e)}")
            # Fallback to random embedding if OpenAI API fails
            dimension = 1536
            import random
            return [random.uniform(-1, 1) for _ in range(dimension)]

# Import configuration
from config import OPENAI_API_KEY, CHROMA_PATH

# Import the PubMed downloader module
from pubmed_downloader import pubmed_downloader_ui

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize session state for ChromaDB
if 'chroma_instance' not in st.session_state:
    st.session_state.chroma_instance = None

# Set page configuration
st.set_page_config(
    page_title="RAG Document Search System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'db' not in st.session_state:
    st.session_state.db = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'db_status' not in st.session_state:
    st.session_state.db_status = "Not initialized"
if 'new_documents' not in st.session_state:
    st.session_state.new_documents = []
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = None

# Initialize additional session state variables for search results and UI state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'accuracy_percentage' not in st.session_state:
    st.session_state.accuracy_percentage = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Document Management"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to process documents
def process_documents(check_for_new=False):
    results_dir = "./results"
    documents = []
    new_documents = []
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        st.warning("Created 'results' directory. Please add PDF documents to it.")
        return [], []
    
    # Get all PDF files from the results directory
    pdf_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning("No PDF files found in the 'results' directory.")
        return [], []
    
    # Get list of already processed documents
    processed_doc_names = [doc["name"] for doc in st.session_state.documents] if check_for_new else []
    
    # Process each PDF file
    with st.spinner("Processing documents..."):
        for pdf_file in pdf_files:
            pdf_path = os.path.join(results_dir, pdf_file)
            try:
                # Check if this is a new document
                is_new = pdf_file not in processed_doc_names
                
                text = extract_text_from_pdf(pdf_path)
                doc_info = {"name": pdf_file, "content": text, "path": pdf_path}
                documents.append(doc_info)
                
                if is_new and check_for_new:
                    new_documents.append(doc_info)
                    
            except Exception as e:
                st.error(f"Error processing {pdf_file}: {str(e)}")
    
    return documents, new_documents

def initialize_chroma() -> Chroma:
    """Initialize ChromaDB with proper handling for conflicts"""
    # If we already have an instance in session state, return it
    if 'chroma_instance' in st.session_state and st.session_state.chroma_instance is not None:
        return st.session_state.chroma_instance
    
    # Check if ChromaDB directory exists - if it does, we'll try to load from it instead of resetting
    chroma_exists = os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0
    logger.info(f"ChromaDB directory exists: {chroma_exists}")
    
    try:
        # Use our custom OpenAI embeddings implementation
        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
        logger.info("Successfully initialized CustomOpenAIEmbeddings")
        
        # Initialize Chroma with proper settings using the updated langchain-chroma package
        try:
            # Create the directory if it doesn't exist
            os.makedirs(CHROMA_PATH, exist_ok=True)
            
            # Initialize ChromaDB instance with the updated approach
            # If the directory exists and has content, this will load the existing DB
            chroma_instance = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_function,
                collection_name="papers"
            )
            
            # Check if we loaded an existing DB or created a new one
            try:
                count = chroma_instance._collection.count()
                if count > 0:
                    logger.info(f"Loaded existing ChromaDB with {count} documents")
                    # Set session state variables to indicate documents are processed
                    st.session_state.processed_docs = True
                    
                    # Also load document list if not already in session state
                    if 'documents' not in st.session_state or not st.session_state.documents:
                        docs, _ = process_documents()
                        st.session_state.documents = docs
                else:
                    logger.info("Created new empty ChromaDB")
            except Exception as count_e:
                logger.warning(f"Could not get document count: {str(count_e)}")
            logger.info("Initialized ChromaDB successfully")
            
            # Store in session state
            st.session_state.chroma_instance = chroma_instance
            st.session_state.db = chroma_instance
            st.session_state.db_status = "Healthy"
            st.session_state.embeddings = embedding_function
            return chroma_instance
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            st.error(f"Error initializing document database: {str(e)}")
            return None
                
    except Exception as outer_e:
        logger.error(f"Unexpected error in initialize_chroma: {str(outer_e)}")
        st.error(f"Unexpected error: {str(outer_e)}")
        return None

class PaperManager:
    def __init__(self):
        self.vectorstore = initialize_chroma()
        
    def clean_title(self, title: str) -> str:
        """Clean and standardize paper title format."""
        # Remove any 'The exact title is' or similar prefixes
        if "the exact title" in title.lower():
            title = title.split("is:")[-1].strip()
            
        # Remove leading/trailing quotes and spaces
        title = title.strip('" ')
        
        # Remove any period at the end
        title = title.rstrip('.')
        
        # Remove any unnecessary spaces
        title = " ".join(title.split())
        
        return title
    
    def extract_first_page_text(self, file_path: str) -> Optional[str]:
        """Extract text from first page of PDF"""
        try:
            with PyPDF2.PdfReader(open(file_path, 'rb')) as pdf:
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    return text.strip() if text else None
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def get_paper_info(self) -> Dict[str, Any]:
        """Get paper counts and titles with improved reliability"""
        files = [f for f in os.listdir("./results") if f.endswith(".pdf")]
        fs_count = len(files)
        
        try:
            # Try multiple approaches to get accurate vector store data
            vs_count = 0
            titles = []
            metadatas = []
            
            # First try the standard get() method
            try:
                stored_data = self.vectorstore.get()
                if stored_data:
                    titles = stored_data.get("documents", [])
                    metadatas = stored_data.get("metadatas", [])
                    vs_count = len(titles)
                    logger.info(f"Retrieved {vs_count} documents using vectorstore.get()")
            except Exception as e:
                logger.warning(f"Error using vectorstore.get(): {str(e)}")
            
            # If that didn't work, try using the collection size if available
            if vs_count == 0 and hasattr(self.vectorstore, '_collection'):
                try:
                    vs_count = self.vectorstore._collection.count()
                    logger.info(f"Retrieved count {vs_count} using _collection.count()")
                    
                    # If we got a count but no documents, try to retrieve them
                    if vs_count > 0 and not titles:
                        collection_data = self.vectorstore._collection.get(limit=vs_count)
                        if collection_data:
                            titles = collection_data.get("documents", [])
                            metadatas = collection_data.get("metadatas", [])
                            logger.info(f"Retrieved {len(titles)} documents using _collection.get()")
                except Exception as e:
                    logger.warning(f"Error using _collection methods: {str(e)}")
            
            # If we still have issues, check the file system
            if vs_count == 0 and os.path.exists(CHROMA_PATH):
                logger.info("Checking ChromaDB filesystem for data existence")
                chroma_files = os.listdir(CHROMA_PATH)
                has_data_files = any(f for f in chroma_files if not f.startswith('.'))
                
                if has_data_files:
                    logger.warning("ChromaDB directory contains files but no documents retrieved")
                    # This suggests ChromaDB has data but we can't access it properly
            
            return {
                "fs_count": fs_count,
                "vs_count": vs_count,
                "titles": titles,
                "metadatas": metadatas,
                "chroma_reliable": vs_count > 0 or not has_data_files  # Flag indicating if ChromaDB seems reliable
            }
        
        except Exception as e:
            logger.error(f"Error getting paper info: {str(e)}")
            return {
                "fs_count": fs_count,
                "vs_count": 0,
                "titles": [],
                "metadatas": [],
                "chroma_reliable": False
            }

    def get_full_text(self, filename: str) -> Optional[str]:
        """Get full text of a paper"""
        try:
            file_path = os.path.join("./results", filename)
            logger.info(f"Extracting text from: {file_path}")
            with PyPDF2.PdfReader(open(file_path, 'rb')) as pdf:
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                logger.info(f"Extracted {len(full_text)} characters from {filename}")
                return full_text
        except Exception as e:
            logger.error(f"Error getting full text: {str(e)}")
            return None
            
    def get_title_from_llm(self, first_page_text: str) -> Optional[str]:
        """Extract and clean paper title using LLM"""
        try:
            prompt = f"""
            Extract the exact title of this research paper from its first page.
            Return only the title without quotes or additional text.
            Return "Unknown Title" if unclear.

            First Page Content:
            {first_page_text[:2000]}
            """
            
            # Use direct OpenAI API approach
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            title = response.choices[0].message.content.strip()
            if title == "Unknown Title":
                return None
                
            # Clean and standardize the title
            cleaned_title = self.clean_title(title)
            logger.info(f"Extracted and cleaned title: {cleaned_title}")
            return cleaned_title
            
        except Exception as e:
            logger.error(f"Error extracting title: {str(e)}")
            return None
            
    def process_paper(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Process a single paper and return (title, error)"""
        try:
            first_page = self.extract_first_page_text(file_path)
            if not first_page:
                return None, "Could not extract text"
                
            title = self.get_title_from_llm(first_page)
            if not title:
                return None, "Could not extract title"
            
            # Additional validation to ensure clean title
            title = self.clean_title(title)
            return title, None
            
        except Exception as e:
            return None, str(e)

    # Sync papers to vector store with persistence
    def sync_papers(self) -> Tuple[int, List]:
        """Sync papers to vector store, checking for existing documents first"""
        files = [f for f in os.listdir("./results") if f.endswith(".pdf")]
        documents = []
        failed_files = []
        processed_count = 0
        
        # Get existing paper information
        paper_info = self.get_paper_info()
        existing_metadatas = paper_info.get("metadatas", [])
        
        # Create a set of already processed filenames for faster lookup
        existing_sources = set()
        for metadata in existing_metadatas:
            if metadata and "source" in metadata:
                existing_sources.add(metadata["source"])
        
        # Log the existing sources for debugging
        logger.info(f"Found {len(existing_sources)} existing papers in vector store")
        
        # Process only new files
        new_files = [f for f in files if f not in existing_sources]
        logger.info(f"Found {len(new_files)} new papers to process")
        
        if not new_files:
            logger.info("No new papers to process")
            return 0, []
        
        for file in new_files:
            file_path = os.path.join("./results", file)
            title, error = self.process_paper(file_path)
            
            if title:
                doc = Document(page_content=title, metadata={"source": file})
                documents.append(doc)
                processed_count += 1
            else:
                failed_files.append((file, error or "Unknown error"))
        
        if documents:
            try:
                self.vectorstore.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
                
                # The newer version of Chroma automatically persists data
                logger.info("Using automatic persistence mechanism of Chroma")
                
                # Verify persistence actually happened
                if verify_chroma_persistence():
                    logger.info("Confirmed ChromaDB persistence is working")
                else:
                    logger.warning("ChromaDB persistence check failed")
                    
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}")
                failed_files.extend([(doc.metadata["source"], "Failed to add to vector store") for doc in documents])
            
        return processed_count, failed_files

# ... (rest of the code remains the same)

def create_vector_db(documents, update_existing=False):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    # Initialize ChromaDB with robust error handling
    db = initialize_chroma()
    if db is None:
        st.error("Failed to initialize ChromaDB")
        return None
    
    # Create vector store
    with st.spinner("Creating vector database..."):
        texts = []
        metadatas = []
        
        for doc in documents:
            # Extract PMID and title from filename if available
            pmid = None
            title = doc["name"]
            
            # Check if filename contains PMID (typical format: 12345678_Title.pdf)
            if doc["name"].startswith("PMC") and "_" in doc["name"]:
                # Handle PMC ID format
                parts = doc["name"].split("_", 1)
                pmid = parts[0]
                title = parts[1].replace(".pdf", "") if len(parts) > 1 else title
            elif doc["name"][0].isdigit() and "_" in doc["name"]:
                # Handle PMID format
                parts = doc["name"].split("_", 1)
                pmid = parts[0]
                title = parts[1].replace(".pdf", "") if len(parts) > 1 else title
            
            # Generate PubMed URL if PMID is available
            pubmed_url = None
            if pmid:
                if pmid.startswith("PMC"):
                    pubmed_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}/"
                else:
                    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            # Split document into chunks
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                # Add enhanced metadata including chunk number, title, and PubMed URL
                metadatas.append({
                    "source": doc["name"],
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "title": title,
                    "pubmed_url": pubmed_url if pubmed_url else ""
                })
        
        try:
            # Check if we're updating existing DB or creating new one
            if update_existing and len(texts) > 0:
                # Add new documents to existing DB
                db.add_texts(texts=texts, metadatas=metadatas)
            elif len(texts) > 0:
                # Create new documents in the DB
                db.add_texts(texts=texts, metadatas=metadatas)
            
            # Verify persistence actually happened
            if verify_chroma_persistence():
                logger.info("Confirmed ChromaDB persistence is working")
                st.session_state.db_status = "Healthy"
                st.session_state.last_processed_time = time.time()
            else:
                logger.warning("ChromaDB persistence check failed")
                st.session_state.db_status = "Warning: Persistence check failed"
                
        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
            st.error(f"Error creating vector database: {str(e)}")
            st.session_state.db_status = f"Error: {str(e)}"
            return None
    
    return db

# Function to check ChromaDB status and reprocess if needed
def check_db_status():
    if st.session_state.db is None:
        # Try to initialize ChromaDB
        db = initialize_chroma()
        if db is not None:
            st.session_state.processed_docs = True
            return "Healthy (Loaded from disk)"
        return "Not initialized"
    
    try:
        # Try a simple operation to check if the DB is working
        count = st.session_state.db._collection.count()
        logger.info(f"ChromaDB collection has {count} documents")
        return "Healthy"
    except Exception as e:
        logger.error(f"Error checking DB status: {str(e)}")
        # Try to reinitialize
        try:
            db = initialize_chroma()
            if db is not None:
                return "Healthy (Reinitialized)"
            return f"Error: {str(e)}"
        except Exception as reinit_error:
            logger.error(f"Error reinitializing DB: {str(reinit_error)}")
            return f"Error: {str(e)} (Reinitialization failed)"

# Function to process query and generate response
def query_documents(query, db):
    # Initialize LLM with correct parameters
    llm = ChatOpenAI(
        model_name="gpt-4o", 
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    
    # Retrieve relevant documents
    with st.spinner("Searching documents..."):
        docs = db.similarity_search(query, k=4)
        
        # Create context from retrieved documents with enhanced metadata
        context_parts = []
        for doc in docs:
            # Extract metadata
            source = doc.metadata['source']
            title = doc.metadata.get('title', source)
            pubmed_url = doc.metadata.get('pubmed_url', '')
            chunk_info = f"Chunk {doc.metadata.get('chunk', '?')}/{doc.metadata.get('total_chunks', '?')}" if 'chunk' in doc.metadata else ''
            
            # Build context entry with enhanced information
            context_entry = f"Document: {source}\n"
            if title and title != source:
                context_entry += f"Title: {title}\n"
            if pubmed_url:
                context_entry += f"URL: {pubmed_url}\n"
            if chunk_info:
                context_entry += f"Chunk Info: {chunk_info}\n"
            context_entry += f"Content: {doc.page_content}"
            
            context_parts.append(context_entry)
        
        context = "\n\n" + "\n\n".join(context_parts)
        
        # Create prompt template for the main answer
        prompt_template = """
        You are a helpful assistant that answers questions based on the provided document context.
        
        Context:
        {context}
        
        Question: {question}
        
        Please provide a comprehensive answer based only on the information in the context. 
        If the answer cannot be determined from the context, say so.
        Include citations to the source documents in your answer using the format [Document: document_name].
        When citing documents, include the title if available, and mention which chunk of the document you're referencing if that information is provided.
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create and run chain for the main answer
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(context=context, question=query)
        
        # Create a separate prompt for accuracy assessment
        accuracy_prompt_template = """
        You are an expert evaluator assessing the accuracy of answers based on source documents.
        
        Source Documents:
        {context}
        
        Question: {question}
        
        Answer to Evaluate: {answer}
        
        Please analyze the answer and determine its accuracy percentage (0-100%) based on:
        1. Factual correctness compared to the source documents
        2. Completeness of information
        3. Appropriate citations to sources
        4. Lack of hallucinations or made-up information
        
        Return ONLY a number between 0 and 100 representing the accuracy percentage. Do not include any other text, explanation, or symbols.
        """
        
        accuracy_prompt = PromptTemplate(
            template=accuracy_prompt_template,
            input_variables=["context", "question", "answer"]
        )
        
        # Create and run chain for accuracy assessment
        accuracy_chain = LLMChain(llm=llm, prompt=accuracy_prompt)
        try:
            accuracy_score = accuracy_chain.run(context=context, question=query, answer=response)
            # Clean up the accuracy score to ensure it's just a number
            accuracy_score = ''.join(c for c in accuracy_score if c.isdigit() or c == '.')
            accuracy_percentage = float(accuracy_score)
            # Ensure it's within 0-100 range
            accuracy_percentage = max(0, min(100, accuracy_percentage))
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {str(e)}")
            accuracy_percentage = None
        
        # Return response and enhanced sources
        sources = []
        for doc in docs:
            source_info = doc.metadata['source']
            
            # Add title if available and different from source
            if 'title' in doc.metadata and doc.metadata['title'] and doc.metadata['title'] != source_info:
                source_info += f" - {doc.metadata['title']}"
            
            # Add chunk info if available
            if 'chunk' in doc.metadata and 'total_chunks' in doc.metadata:
                source_info += f" (Chunk {doc.metadata['chunk']}/{doc.metadata['total_chunks']})"
            
            # Add PubMed URL if available
            if 'pubmed_url' in doc.metadata and doc.metadata['pubmed_url']:
                source_info += f" | [PubMed Link]({doc.metadata['pubmed_url']})"
            
            sources.append(source_info)
            
        return response, sources, docs, accuracy_percentage

# Main application UI
def main():
    st.title("📚 OptiGuide")
    
    # Display vector database stats in the sidebar
    with st.sidebar:
        st.header("Vector Database Stats")
        
        # Get the Chroma instance
        chroma_instance = initialize_chroma()
        
        if chroma_instance:
            # Get the number of documents in the collection
            try:
                num_documents = chroma_instance._collection.count()
                st.info(f"📊 **Documents in Vector DB:** {num_documents} chunks")
            except Exception as e:
                st.error(f"Error getting document count: {str(e)}")
                num_documents = 0
        else:
            st.warning("Vector database not initialized")
            num_documents = 0
        
        # Add a reset button
        if st.button("🗑️ Reset Vector Database"):
            if reset_chroma():
                # Clear session state variables related to ChromaDB
                if 'chroma_instance' in st.session_state:
                    st.session_state.chroma_instance = None
                if 'db' in st.session_state:
                    st.session_state.db = None
                
                st.success("Vector database reset successfully!")
                st.info("Please refresh the page to initialize a new database.")
                # Force a rerun to refresh the page
                st.rerun()
            else:
                st.error("Failed to reset vector database.")
    
    # Initialize tabs
    tab1, tab2, tab3 = st.tabs(["Document Management", "Search", "PubMed Downloader"])
    
    # Document Management Tab
    with tab1:
        st.header("Document Management")
        
        # Check if ChromaDB exists and load it
        if st.session_state.db is None:
            with st.spinner("Initializing vector database..."):
                db_status = check_db_status()
                if "Healthy" in db_status:
                    # Load document list from disk if DB was loaded
                    docs, _ = process_documents()
                    st.session_state.documents = docs
        
        # Upload new documents
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        if uploaded_file is not None:
            # Save the uploaded file to the results directory
            with open(os.path.join("results", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            # Add to new documents list
            st.session_state.new_documents.append({"name": uploaded_file.name})
        
        # Check for new documents
        if st.button("Check for New Documents"):
            all_docs, new_docs = process_documents(check_for_new=True)
            st.session_state.documents = all_docs
            st.session_state.new_documents = new_docs
            
            if new_docs:
                st.success(f"Found {len(new_docs)} new documents!")
            else:
                st.info("No new documents found.")
        
        # Process documents button - now handles both initial processing and updates
        if st.button("Process All Documents"):
            all_docs, _ = process_documents()
            if all_docs:
                st.session_state.db = create_vector_db(all_docs)
                if st.session_state.db is not None:
                    st.session_state.processed_docs = True
                    st.session_state.documents = all_docs
                    st.session_state.new_documents = []
                    st.success("All documents processed and embedded successfully!")
                    st.session_state.db_status = "Healthy"
                    # Force a rerun to update the sidebar stats
                    st.rerun()
                else:
                    st.error("Failed to create vector database. Please check the logs.")
                    st.session_state.db_status = "Error: Failed to create vector database"
        
        # New documents section
        if st.session_state.new_documents:
            st.subheader(f"New Documents ({len(st.session_state.new_documents)})")
            for doc in st.session_state.new_documents:
                st.write(f"- {doc['name']}")
            
            if st.button("Embed New Documents"):
                if st.session_state.db is not None:
                    # Update existing DB with new documents
                    st.session_state.db = create_vector_db(st.session_state.new_documents, update_existing=True)
                    if st.session_state.db is not None:
                        st.success(f"Successfully embedded {len(st.session_state.new_documents)} new documents!")
                        st.session_state.new_documents = []
                        # Force a rerun to update the sidebar stats
                        st.rerun()
                else:
                    # No existing DB, create new one with all documents
                    all_docs, _ = process_documents()
                    st.session_state.db = create_vector_db(all_docs)
                    if st.session_state.db is not None:
                        st.success("All documents processed and embedded successfully!")
                        st.session_state.new_documents = []
                        st.session_state.documents = all_docs
                        # Force a rerun to update the sidebar stats
                        st.rerun()
        
        # Display available documents
        st.subheader("Available Documents")
        if not st.session_state.processed_docs and not st.session_state.documents:
            # Initial load of documents
            docs, _ = process_documents()
            if docs:
                st.session_state.documents = docs
                for doc in docs:
                    st.write(f"- {doc['name']}")
            else:
                st.info("No documents available. Please upload PDFs to the 'results' folder.")
        else:
            # Display already processed documents
            for doc in st.session_state.documents:
                st.write(f"- {doc['name']}")
        
        # Display ChromaDB status
        st.subheader("Vector Database Status")
        current_status = check_db_status() if st.session_state.db is not None else st.session_state.db_status
        status_color = "green" if "Healthy" in current_status else "red"
        st.markdown(f"<p style='color:{status_color};'>Status: {current_status}</p>", unsafe_allow_html=True)
        
        # Add a button to reprocess documents with enhanced metadata
        if st.button("🔄 Reprocess with Enhanced Metadata"):
            try:
                # First, try to reset the database
                if reset_chroma():
                    st.success("Database reset successfully!")
                    # Clear session state variables related to ChromaDB
                    if 'chroma_instance' in st.session_state:
                        st.session_state.chroma_instance = None
                    if 'db' in st.session_state:
                        st.session_state.db = None
                    
                    # Process all documents with enhanced metadata
                    all_docs, _ = process_documents()
                    if all_docs:
                        # Initialize a new ChromaDB instance with the updated path
                        embedding_function = CustomOpenAIEmbeddings(api_key=OPENAI_API_KEY)
                        
                        # Create a new vector database
                        with st.spinner("Creating vector database with enhanced metadata..."):
                            try:
                                # Create vector store
                                texts = []
                                metadatas = []
                                
                                for doc in all_docs:
                                    # Extract PMID and title from filename if available
                                    pmid = None
                                    title = doc["name"]
                                    
                                    # Check if filename contains PMID (typical format: 12345678_Title.pdf)
                                    if doc["name"].startswith("PMC") and "_" in doc["name"]:
                                        # Handle PMC ID format
                                        parts = doc["name"].split("_", 1)
                                        pmid = parts[0]
                                        title = parts[1].replace(".pdf", "") if len(parts) > 1 else title
                                    elif doc["name"][0].isdigit() and "_" in doc["name"]:
                                        # Handle PMID format
                                        parts = doc["name"].split("_", 1)
                                        pmid = parts[0]
                                        title = parts[1].replace(".pdf", "") if len(parts) > 1 else title
                                    
                                    # Generate PubMed URL if PMID is available
                                    pubmed_url = None
                                    if pmid:
                                        if pmid.startswith("PMC"):
                                            pubmed_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}/"
                                        else:
                                            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                    
                                    # Split document into chunks
                                    text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000,
                                        chunk_overlap=200,
                                        length_function=len,
                                    )
                                    chunks = text_splitter.split_text(doc["content"])
                                    
                                    for i, chunk in enumerate(chunks):
                                        texts.append(chunk)
                                        # Add enhanced metadata
                                        metadatas.append({
                                            "source": doc["name"],
                                            "chunk": i + 1,
                                            "total_chunks": len(chunks),
                                            "title": title,
                                            "pubmed_url": pubmed_url if pubmed_url else ""
                                        })
                                
                                # Initialize fresh ChromaDB instance
                                chroma_instance = Chroma(
                                    persist_directory=CHROMA_PATH,
                                    embedding_function=embedding_function,
                                    collection_name="papers"
                                )
                                
                                # Add texts to the database
                                if len(texts) > 0:
                                    chroma_instance.add_texts(texts=texts, metadatas=metadatas)
                                    st.session_state.db = chroma_instance
                                    st.session_state.chroma_instance = chroma_instance
                                    st.session_state.processed_docs = True
                                    st.session_state.documents = all_docs
                                    st.session_state.new_documents = []
                                    st.success(f"Successfully reprocessed {len(all_docs)} documents with enhanced metadata!")
                                    st.session_state.db_status = "Healthy"
                                    # Force a rerun to update the sidebar stats
                                    st.rerun()
                                else:
                                    st.error("No text content found in documents.")
                            except Exception as e:
                                st.error(f"Error creating vector database: {str(e)}")
                                logger.error(f"Error creating vector database: {str(e)}")
                    else:
                        st.warning("No documents found to process.")
                else:
                    st.error("Failed to reset vector database. Please check the logs.")
            except Exception as e:
                st.error(f"Error during reprocessing: {str(e)}")
                logger.error(f"Error during reprocessing: {str(e)}")
        
        # Show last processed time if available
        if st.session_state.last_processed_time:
            last_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_processed_time))
            st.markdown(f"<p><small>Last processed: {last_time}</small></p>", unsafe_allow_html=True)
    
    # Search Tab
    with tab2:
        st.header("Ask Questions About Your Documents")
        
        # Query input
        query = st.text_input("Enter your question:")
        
        if st.button("Search"):
            if not query:
                st.warning("Please enter a question.")
            elif not st.session_state.db:
                # Try to initialize the database if it's not in session state
                db = initialize_chroma()
                if db is not None and hasattr(db, '_collection') and db._collection.count() > 0:
                    st.session_state.db = db
                    st.session_state.processed_docs = True
                    # Process query and display results
                    response, sources, docs, accuracy_percentage = query_documents(query, db)
                else:
                    st.warning("Please process documents first.")
            else:
                # Process query and display results
                response, sources, docs, accuracy_percentage = query_documents(query, st.session_state.db)
                
                # Display answer with accuracy percentage
                st.subheader("Answer")
                
                # Show accuracy percentage if available
                if accuracy_percentage is not None:
                    # Determine color based on accuracy percentage
                    if accuracy_percentage >= 90:
                        accuracy_color = "green"
                    elif accuracy_percentage >= 70:
                        accuracy_color = "orange"
                    else:
                        accuracy_color = "red"
                    
                    # Display accuracy badge
                    st.markdown(f"<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
                              f"<div style='background-color: {accuracy_color}; color: white; padding: 4px 8px; "
                              f"border-radius: 4px; font-size: 14px; margin-right: 10px;'>"
                              f"Accuracy: {accuracy_percentage:.1f}%</div>"
                              f"<div style='font-size: 12px; color: gray;'>Based on source document analysis</div>"
                              f"</div>", unsafe_allow_html=True)
                
                # Display the answer
                st.write(response)
                
                # Display sources with enhanced information
                st.subheader("Sources")
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(docs):
                        # Create a formatted source heading with all available metadata
                        source_heading = f"**Source {i+1}: {doc.metadata['source']}**"
                        
                        # Add title if available and different from source
                        if 'title' in doc.metadata and doc.metadata['title'] and doc.metadata['title'] != doc.metadata['source']:
                            source_heading = f"**Source {i+1}: {doc.metadata['title']}**"
                        
                        # Display the source heading
                        st.markdown(source_heading)
                        
                        # Display chunk information if available
                        if 'chunk' in doc.metadata and 'total_chunks' in doc.metadata:
                            st.markdown(f"*Chunk {doc.metadata['chunk']}/{doc.metadata['total_chunks']}*")
                        
                        # Display PubMed link if available
                        if 'pubmed_url' in doc.metadata and doc.metadata['pubmed_url']:
                            st.markdown(f"[View on PubMed]({doc.metadata['pubmed_url']})")
                        
                        # Display the content
                        st.text(doc.page_content)
                        st.markdown("---")
    
    # PubMed Downloader Tab
    with tab3:
        st.header("PubMed Downloader")
        pubmed_downloader_ui()
if __name__ == "__main__":
    main()
