from pathlib import Path
from typing import List, Optional
import hashlib
import json

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
        self._checksum = None
        self._doc_contents = []

    def _calculate_docs_checksum(self, docs: List[Document]) -> str:
        """Calculate checksum of document contents.
        
        Args:
            docs: List of documents to calculate checksum for
            
        Returns:
            str: Hexadecimal checksum of document contents
        """
        # Sort documents by content to ensure consistent ordering
        sorted_contents = sorted([doc.page_content for doc in docs])
        combined_content = "\n".join(sorted_contents)
        hasher = hashlib.sha256()
        hasher.update(combined_content.encode())
        return hasher.hexdigest()

    def init_store(self, docs: List[Document]) -> None:
        """Initialize a new vector store with documents.
        
        Args:
            docs: List of documents to initialize the store with
        """
        self._vector_store = FAISS.from_documents(docs, self.embedding_model)
        self._checksum = self._calculate_docs_checksum(docs)
        self._doc_contents = [doc.page_content for doc in docs]

    def _get_checksum_path(self, store_path: str) -> Path:
        """Get the path for the checksum file.
        
        Args:
            store_path: Path to the vector store
            
        Returns:
            Path to the checksum file
        """
        # Convert store path to Path object
        store_path = Path(store_path)
        # Place checksum file next to the store directory
        return store_path.parent / f"{store_path.name}.checksum"

    def save_store(self, path: str, force: bool = False) -> None:
        """Save the vector store to disk with integrity check.
        
        Args:
            path: Path to save the vector store to
            force: Whether to overwrite existing store
        
        Raises:
            FileExistsError: If path exists and force=False
            RuntimeError: If no vector store is loaded
        """
        store_path = Path(path)
        
        # Create directory if it doesn't exist
        store_path.parent.mkdir(parents=True, exist_ok=True)
        
        if store_path.exists() and not force:
            raise FileExistsError(
                f"Path '{path}' already exists. Use a different name or set force=True."
            )
        if self._vector_store is None:
            raise RuntimeError("No vector store in memory to save.")
        
        # Save the store
        self._vector_store.save_local(str(store_path))
        
        # Save the document checksum and contents
        checksum_path = self._get_checksum_path(store_path)
        with open(checksum_path, 'w') as f:
            json.dump({
                "checksum": self._checksum,
                "doc_contents": self._doc_contents
            }, f)
            
        print(f"Vector store saved to '{store_path}'.")

    def load_store(self, path: str, force: bool = False) -> None:
        """Load a vector store from disk with integrity verification.
        
        Args:
            path: Path to load the vector store from
            force: Whether to override existing loaded store
            
        Raises:
            RuntimeError: If store already loaded and force=False
            FileNotFoundError: If path doesn't exist
            SecurityError: If integrity check fails
        """
        if self._vector_store is not None and not force:
            raise RuntimeError(
                "A vector store is already loaded. Use force=True to override."
            )
            
        store_path = Path(path)
        
        # Check if the index file exists
        index_path = store_path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Vector store index not found at '{index_path}'")
            
        # Check for existing checksum and doc contents
        checksum_path = self._get_checksum_path(path)
        if not checksum_path.exists():
            raise FileNotFoundError(f"Checksum file not found at '{checksum_path}'")
            
        with open(checksum_path, 'r') as f:
            stored_data = json.load(f)
            stored_checksum = stored_data["checksum"]
            if "doc_contents" not in stored_data:
                print("Warning: Loading vector store saved in old format. Please recreate the store to enable content verification.")
                # For backward compatibility, we'll load without content verification
                self._vector_store = FAISS.load_local(
                    str(store_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                self._checksum = stored_checksum
                return
            stored_contents = stored_data["doc_contents"]
        
        # Load the store
        self._vector_store = FAISS.load_local(
            str(store_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Verify integrity by reconstructing documents and checking checksum
        docs = [Document(page_content=content) for content in stored_contents]
        current_checksum = self._calculate_docs_checksum(docs)
        
        print(f"Debug - Stored checksum: {stored_checksum}")
        print(f"Debug - Current checksum: {current_checksum}")
        
        if current_checksum != stored_checksum:
            self._vector_store = None
            raise SecurityError(
                f"Vector store integrity check failed. Document contents have changed."
            )
            
        self._checksum = stored_checksum
        self._doc_contents = stored_contents
        
        print(f"Vector store loaded from '{store_path}'.")

    def update_store(self, new_docs: List[Document], save_path: Optional[str] = None, force: bool = False) -> None:
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
        
        # Update document contents and checksum
        self._doc_contents.extend([doc.page_content for doc in new_docs])
        docs = [Document(page_content=content) for content in self._doc_contents]
        self._checksum = self._calculate_docs_checksum(docs)
        
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


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass 