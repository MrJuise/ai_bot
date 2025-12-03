import chromadb
from langchain_ollama import OllamaEmbeddings
import asyncio

class OllamaEmbeddingFunction:
    """Обёртка для OllamaEmbeddings для использования с ChromaDB"""
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.embeddings = OllamaEmbeddings(model=model_name, base_url="http://ollama:11434")
        self._model_name = model_name
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Преобразует список текстов в список эмбеддингов"""
        return self.embeddings.embed_documents(input)
    
    def embed_query(self, input=None, **kwargs) -> list[list[float]]:
        """Преобразует текст(и) в эмбеддинг(и) для запросов
        
        ChromaDB вызывает этот метод с keyword argument 'input'
        Может передавать как один текст, так и список текстов
        Всегда возвращает список списков эмбеддингов
        """
        # Обрабатываем случай, когда input передается как keyword argument
        if input is None:
            if 'input' in kwargs:
                input = kwargs['input']
            else:
                raise ValueError("No input provided to embed_query")
        
        # Если input - список, обрабатываем каждый элемент
        if isinstance(input, list):
            if len(input) == 0:
                raise ValueError("Empty input list")
            # Для списка текстов создаем эмбеддинги для каждого
            embeddings = []
            for text in input:
                if not text:
                    raise ValueError("Empty text in input list")
                emb = self.embeddings.embed_query(text)
                embeddings.append(emb)
            return embeddings
        else:
            # Для одного текста возвращаем список с одним эмбеддингом
            if not input:
                raise ValueError("Empty text input")
            emb = self.embeddings.embed_query(input)
            return [emb]  # Возвращаем список списков
    
    def name(self) -> str:
        """Возвращает имя embedding function для ChromaDB"""
        return f"ollama-{self._model_name}"

class VectorStore:
    def __init__(self, path: str = "/app/data/vectorstore/chroma_db"):
        # PersistentClient создаёт хранилище на диске
        self.client = chromadb.PersistentClient(path=path)
        # Создаём embedding function для ChromaDB с Ollama
        embedding_function = OllamaEmbeddingFunction(
            model_name="nomic-embed-text"
        )
        
        # Удаляем существующую коллекцию, если она есть (для пересоздания с новой embedding function)
        try:
            self.client.delete_collection(name="knowledge_base")
        except Exception:
            # Коллекция не существует, это нормально
            pass
        
        # Создаём коллекцию для хранения знаний
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base", 
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedding_function
        )




    def add(self, ids: list, documents: list):
        # ChromaDB автоматически создаст эмбеддинги через embedding_function
        self.collection.add(ids=ids, documents=documents)
    
    async def add_async(self, ids: list, documents: list):
        """Асинхронная версия add - не блокирует event loop"""
        await asyncio.to_thread(self.add, ids, documents)

    def query(self, text: str, n_results: int = 3):
        # ChromaDB автоматически создаст эмбеддинг для запроса через embedding_function
        return self.collection.query(
            query_texts=[text], n_results=n_results
        )
    
    async def query_async(self, text: str, n_results: int = 3):
        """Асинхронная версия query - не блокирует event loop"""
        return await asyncio.to_thread(self.query, text, n_results)
