# RAG-Powered Document Search System

A Streamlit application that enables users to search through and query PDF documents using OpenAI's GPT-4o model with Retrieval Augmented Generation (RAG).

## Features

- View all available PDFs in the 'results' folder
- Upload new PDFs directly through the interface
- Ask natural language questions about the content in your PDFs
- Receive accurate answers with citations to source documents
- Intuitive interface with minimal training required

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone this repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Application

1. Make sure you have PDF documents in the 'results' folder (or upload them through the interface)
2. Start the Streamlit application:

```bash
streamlit run app.py
```

3. Open your browser and go to the URL displayed in the terminal (typically http://localhost:8501)

## Usage

1. **Process Documents**: Click the "Process Documents" button in the sidebar to scan and process PDFs from the 'results' folder
2. **Upload Documents**: Use the file uploader in the sidebar to add new PDFs
3. **Ask Questions**: Enter your question in the text input field and click "Search"
4. **View Results**: See the answer generated based on your documents, with citations to the source material
5. **Explore Sources**: Expand the "View Source Documents" section to see the exact passages used to generate the answer

## Technical Details

- **PDF Processing**: PyPDF2 for text extraction
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Database**: ChromaDB for storing document chunks
- **LLM Integration**: OpenAI GPT-4o for answer generation
- **Frontend**: Streamlit for the user interface

## Limitations

- Currently only supports text-based PDFs (no OCR for scanned documents)
- Performance may vary with very large document sets
- Requires an internet connection for OpenAI API access
