from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory="D:\\AI Course\\Capstone_Project\\AI_Wellness_and_Recovery_Agent\\chroma_db",
    embedding_function=embedding
)


def retrieve_context(question):

    docs = db.similarity_search(question, k=3)

    return "\n\n".join([doc.page_content for doc in docs])