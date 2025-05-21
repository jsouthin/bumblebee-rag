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

    def init_store(self, docs: List[Document]) -> None:
        """Initialize a new vector store with documents.
        
        Args:
            docs: List of documents to initialize the store with
        """
        self._vector_store = FAISS.from_documents(docs, self.embedding_model)
        self._update_checksum()

    def _update_checksum(self) -> None:
        """Update the store's checksum."""
        if self._vector_store:
            # Create a hash of the vector store's content
            hasher = hashlib.sha256()
            store_data = self._vector_store.serialize_to_bytes()
            hasher.update(store_data)
            self._checksum = hasher.hexdigest()

    def _verify_checksum(self, store_data: bytes) -> bool:
        """Verify the integrity of loaded data.
        
        Args:
            store_data: The data to verify
            
        Returns:
            bool: Whether the data is valid
        """
        hasher = hashlib.sha256()
        hasher.update(store_data)
        computed_hash = hasher.hexdigest()
        
        # If no previous checksum exists, this is the first load
        if not self._checksum:
            self._checksum = computed_hash
            return True
            
        return computed_hash == self._checksum

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
        
        # Update and save the checksum
        self._update_checksum()
        checksum_path = self._get_checksum_path(store_path)
        with open(checksum_path, 'w') as f:
            json.dump({"checksum": self._checksum}, f)
            
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
            
        # Check for existing checksum
        checksum_path = self._get_checksum_path(path)
        has_existing_checksum = checksum_path.exists()
        
        if has_existing_checksum:
            with open(checksum_path, 'r') as f:
                stored_checksum = json.load(f)["checksum"]
                self._checksum = stored_checksum
        
        # Load the store with dangerous deserialization allowed
        self._vector_store = FAISS.load_local(
            str(store_path),
            self.embedding_model,
            allow_dangerous_deserialization=True  # Required by FAISS
        )
        
        # For pre-existing stores without checksums, create one
        if not has_existing_checksum:
            self._update_checksum()
            # Save the new checksum
            with open(checksum_path, 'w') as f:
                json.dump({"checksum": self._checksum}, f)
            print(f"Created new checksum for existing vector store at '{store_path}'")
        else:
            # Verify integrity for stores with existing checksums
            store_data = self._vector_store.serialize_to_bytes()
            if not self._verify_checksum(store_data):
                self._vector_store = None
                raise SecurityError("Vector store integrity check failed. Possible tampering detected.")
            
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
        self._update_checksum()
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