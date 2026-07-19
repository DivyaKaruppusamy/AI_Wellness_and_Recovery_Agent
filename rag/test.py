from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# This downloads and runs the embedding model automatically inside Python
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db_connection = Chroma(
    collection_name="my_wellness_collection",
    persist_directory="D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db",
    embedding_function=embedding_model
)

# ==========================================
# VERIFICATION TEST 1: Check Document Count
# ==========================================
try:
    # Fetch the underlying client collection to count vectors
    collection_data = db_connection._collection.get()
    vector_count = len(collection_data['ids'])
    
    print("=" * 40)
    print(" DATABASE STATUS: SUCCESS")
    print(f" Total text chunks stored: {vector_count}")
    print("=" * 40)
    
    if vector_count == 0:
        print("⚠️ Warning: The database exists, but it contains 0 documents.")
        
except Exception as e:
    print(f"❌ Error connecting to the database: {e}")
    exit()

# ==========================================
# VERIFICATION TEST 2: Semantic Search Query
# ==========================================
# Pick a topic or keyword you know exists inside your raw text data
test_query = "What is the primary conclusion of the document?" 

print(f"\nRunning test query: '{test_query}'...")
results = db_connection.similarity_search(test_query, k=2)

print(f"\n Returned Matches: {len(results)}")
for index, doc in enumerate(results):
    print(f"\n--- Match #{index + 1} ---")
    print(f"Source file: {doc.metadata.get('source', 'Unknown')}")
    print(f"Snippet: {doc.page_content[:200]}...")