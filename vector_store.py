from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    def __init__(self, embedding_model: Embeddings):
        """Initialize the vector store manager.
        
        Args:
            embedding_model: The embedding model to use for vectorization
        """
        self._vector_store: Optional[FAISS] = None
        self.embedding_model = embedding_model

    def init_store(self, docs: List[Document]) -> None:
        """Initialize a new vector store with documents.
        
        Args:
            docs: List of documents to initialize the store with
        """
        self._vector_store = FAISS.from_documents(docs, self.embedding_model)

    def save_store(self, path: str, force: bool = False) -> None:
        """Save the vector store to disk.
        
        Args:
            path: Path to save the vector store to
            force: Whether to overwrite existing store
        
        Raises:
            FileExistsError: If path exists and force=False
            RuntimeError: If no vector store is loaded
        """
        if Path(path).exists() and not force:
            raise FileExistsError(
                f"Path '{path}' already exists. Use a different name or set force=True."
            )
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory to save.")
        
        self._vector_store.save_local(path)
        print(f"Vector store saved to '{path}'.")

    def load_store(
        self, 
        path: str, 
        force: bool = False, 
        allow_dangerous_deserialization: bool = False
    ) -> None:
        """Load a vector store from disk.
        
        Args:
            path: Path to load the vector store from
            force: Whether to override existing loaded store
            allow_dangerous_deserialization: Whether to allow pickle deserialization
            
        Raises:
            RuntimeError: If store already loaded and force=False
            FileNotFoundError: If path doesn't exist
            ValueError: If allow_dangerous_deserialization not explicitly set
        """
        if self._vector_store is not None and not force:
            raise RuntimeError(
                "A vector store is already loaded. Use force=True to override."
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"Vector store path '{path}' does not exist.")
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization relies on loading a pickle file. "
                "Pickle files can be modified to deliver malicious payloads. "
                "Set allow_dangerous_deserialization=True if you trust the source."
            )

        self._vector_store = FAISS.load_local(
            path, 
            self.embedding_model,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        print(f"Vector store loaded from '{path}'.")

    def update_store(
        self, 
        new_docs: List[Document], 
        save_path: Optional[str] = None,
        force: bool = False
    ) -> None:
        """Update the vector store with new documents.
        
        Args:
            new_docs: New documents to add
            save_path: Optional path to save updated store
            force: Whether to force save if path exists
            
        Raises:
            RuntimeError: If no store is loaded
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory. Load or initialize first.")
            
        self._vector_store.add_documents(new_docs)
        print(f"Vector store updated with {len(new_docs)} new documents.")
        
        if save_path:
            self.save_store(save_path, force)

    def get_store(self) -> FAISS:
        """Get the underlying vector store.
        
        Returns:
            The FAISS vector store instance
            
        Raises:
            RuntimeError: If no store is loaded
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory.")
        return self._vector_store

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on the vector store.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of similar documents
            
        Raises:
            RuntimeError: If no store is loaded
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory.")
        return self._vector_store.similarity_search(query, k=k)

    @property
    def document_count(self) -> int:
        """Get the number of documents in the store.
        
        Returns:
            Number of documents
            
        Raises:
            RuntimeError: If no store is loaded
        """
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory.")
        return self._vector_store.index.ntotal 