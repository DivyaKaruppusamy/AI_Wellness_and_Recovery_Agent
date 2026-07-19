from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# STEP 1: Custom Ingestion (Replacing Community DirectoryLoader)
# ==========================================
def load_raw_text_files(directory_path: str, glob_pattern: str = "*.txt") -> list[Document]:
    documents = []
    base_path = Path(directory_path)
    
    for file_path in base_path.glob(glob_pattern):
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                # Wrap directly in LangChain's core Document object
                documents.append(Document(
                    page_content=content,
                    metadata={"source": str(file_path)}
                ))
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
    return documents

raw_docs = load_raw_text_files("data/raw")

# ==========================================
# STEP 2: Document Chunking
# ==========================================
# Break text into dense 1000-character blocks with overlap to keep context split-points healthy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(raw_docs)
print(f"Split {len(raw_docs)} files into {len(chunked_docs)} semantic chunks.")


# embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# vector_store = Chroma.from_documents(
#     documents=chunked_docs,
#     embedding=embedding_model,
#     collection_name="my__wellness_collecttion",
#     persist_directory="D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db"  # Automatically saves to your local disk
# )

# ==========================================
# STEP 3: Persist into Chroma DB using Standalone Package
# ==========================================
# Initialise your standalone embedding wrapper
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist the database locally
vector_store = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embedding_model,
    collection_name="my_wellness_collection",
    persist_directory="D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db"
)

print("ChromaDB vector store successfully created and saved!")
