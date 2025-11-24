# db_operations.py (local sentence-transformers embeddings + InMemoryVectorStore)
from langchain_core.vectorstores import InMemoryVectorStore
from sentence_transformers import SentenceTransformer

HF_MODEL_NAME = "all-MiniLM-L6-v2"

class LocalEmbeddings:
    def __init__(self, model_name: str = HF_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(self, text):
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(emb)

def get_embedding_function(model_name: str = HF_MODEL_NAME):
    return LocalEmbeddings(model_name)

def add_to_db(chunks, embedding_function) -> InMemoryVectorStore:
    db = InMemoryVectorStore(embedding=embedding_function)
    db.add_documents(chunks)
    return db
