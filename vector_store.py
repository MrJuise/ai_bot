import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class VectorStore:
    def __init__(self, path: str = "./vectorstore/chroma_db"):
        # PersistentClient создаёт хранилище на диске
        self.client = chromadb.PersistentClient(path=path)
        # Создаём embedding function для ChromaDB
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Создаём коллекцию для хранения знаний
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base", 
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function
        )




    def add(self, ids: list, documents: list):
        # ChromaDB автоматически создаст эмбеддинги через embedding_function
        self.collection.add(ids=ids, documents=documents)

    def query(self, text: str, n_results: int = 3):
        # ChromaDB автоматически создаст эмбеддинг для запроса через embedding_function
        return self.collection.query(
            query_texts=[text], n_results=n_results
        )
