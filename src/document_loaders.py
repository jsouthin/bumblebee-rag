# document_loaders.py

from pathlib import Path
from typing import List, Union, Optional
from datetime import datetime
from urllib.parse import urlparse
import logging
import bs4
import time
from abc import ABC, abstractmethod
import backoff
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from langchain.schema import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    CSVLoader,
    GoogleDriveLoader,
    NotionDBLoader,
    SeleniumURLLoader,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoadError(Exception):
    """Custom exception for document loading errors."""
    pass

class BaseDocumentLoader(ABC):
    """Abstract base class for all document loaders."""
    
    def __init__(self, project_id: str):
        """Initialize the document loader.
        
        Args:
            project_id: Identifier for the project
        """
        self.project_id = project_id
        
    @abstractmethod
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        """Load documents from the source.
        
        Args:
            source: The source to load documents from (URL, file path, etc.)
            **kwargs: Additional arguments specific to the loader
            
        Returns:
            List of loaded documents
            
        Raises:
            DocumentLoadError: If loading fails
        """
        pass
    
    def _add_metadata(self, docs: List[Document], source_id: str) -> List[Document]:
        """Add common metadata to documents.
        
        Args:
            docs: List of documents to add metadata to
            source_id: Source identifier
            
        Returns:
            List of documents with added metadata
        """
        timestamp = int(time.time())
        for doc in docs:
            doc.metadata.update({
                "project_id": self.project_id,
                "source_id": source_id,
                "timestamp_added": timestamp
            })
        return docs
    
    @staticmethod
    def _ensure_list(items: Union[str, List[str]]) -> List[str]:
        """Convert single string to list if necessary."""
        return [items] if isinstance(items, str) else items


class WebDocumentLoader(BaseDocumentLoader):
    """Loader for static web pages."""
    
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        urls = self._ensure_list(source)
        all_docs = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(
                    web_paths=(url,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(
                            class_=("post-content", "post-title", "post-header")
                        )
                    ),
                )
                docs = loader.load()
                source_id = Path(urlparse(url).path).stem
                all_docs.extend(self._add_metadata(docs, source_id))
                logger.info(f"Successfully loaded document from {url}")
            except Exception as e:
                logger.error(f"Failed to load document from {url}: {str(e)}")
                raise DocumentLoadError(f"Failed to load web document: {str(e)}")
        
        return all_docs


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents."""
    
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        file_paths = self._ensure_list(source)
        all_docs = []
        
        for path in file_paths:
            try:
                loader = PyPDFLoader(path)
                docs = loader.load()
                source_id = Path(path).stem
                all_docs.extend(self._add_metadata(docs, source_id))
                logger.info(f"Successfully loaded PDF from {path}")
            except Exception as e:
                logger.error(f"Failed to load PDF from {path}: {str(e)}")
                raise DocumentLoadError(f"Failed to load PDF document: {str(e)}")
        
        return all_docs


class GoogleDriveDocumentLoader(BaseDocumentLoader):
    """Loader for Google Drive documents."""
    
    def load(self, source: Union[str, List[str]], *, service_account_json: str, **kwargs) -> List[Document]:
        doc_ids = self._ensure_list(source)
        all_docs = []
        
        for doc_id in doc_ids:
            try:
                loader = GoogleDriveLoader(
                    document_ids=[doc_id],
                    service_account_key=Path(service_account_json)
                )
                docs = loader.load()
                all_docs.extend(self._add_metadata(docs, doc_id))
                logger.info(f"Successfully loaded Google Drive document {doc_id}")
            except Exception as e:
                logger.error(f"Failed to load Google Drive document {doc_id}: {str(e)}")
                raise DocumentLoadError(f"Failed to load Google Drive document: {str(e)}")
        
        return all_docs


class CSVDocumentLoader(BaseDocumentLoader):
    """Loader for CSV documents."""
    
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        file_paths = self._ensure_list(source)
        all_docs = []
        
        for path in file_paths:
            try:
                loader = CSVLoader(path)
                docs = loader.load()
                source_id = Path(path).stem
                all_docs.extend(self._add_metadata(docs, source_id))
                logger.info(f"Successfully loaded CSV from {path}")
            except Exception as e:
                logger.error(f"Failed to load CSV from {path}: {str(e)}")
                raise DocumentLoadError(f"Failed to load CSV document: {str(e)}")
        
        return all_docs


class NotionDocumentLoader(BaseDocumentLoader):
    """Loader for Notion documents."""
    
    def load(self, source: Union[str, List[str]], *, token: str, **kwargs) -> List[Document]:
        page_ids = self._ensure_list(source)
        
        try:
            loader = NotionDBLoader(
                integration_token=token,
                page_ids=page_ids
            )
            docs = loader.load()
            # Add metadata to each document
            for doc in docs:
                self._add_metadata([doc], doc.metadata.get("page_id", "unknown"))
            logger.info(f"Successfully loaded {len(docs)} Notion documents")
            return docs
        except Exception as e:
            logger.error(f"Failed to load Notion documents: {str(e)}")
            raise DocumentLoadError(f"Failed to load Notion documents: {str(e)}")


class DynamicWebDocumentLoader(BaseDocumentLoader):
    """Loader for dynamic web pages using Selenium with enhanced reliability."""
    
    def __init__(self, project_id: str):
        """Initialize the document loader with custom settings."""
        super().__init__(project_id)
        self.timeout = 10  # seconds to wait for elements
        self.max_retries = 3
        
    @backoff.on_exception(backoff.expo, 
                         (TimeoutException, WebDriverException),
                         max_tries=3)
    def _load_with_retry(self, url: str) -> Document:
        """Load a single URL with retry logic and better waiting strategies."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            
            # Wait for the main content to be present
            wait = WebDriverWait(driver, self.timeout)
            
            # Wait for essential elements that indicate page load
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            
            # Get the main content - prefer main tag if it exists, otherwise use body
            try:
                main_content = driver.find_element(By.TAG_NAME, "main")
                content = main_content.text
            except:
                content = driver.find_element(By.TAG_NAME, "body").text
            
            return Document(
                page_content=content,
                metadata={"source": url, "title": driver.title}
            )
            
        finally:
            driver.quit()
    
    def load(self, source: Union[str, List[str]], **kwargs) -> List[Document]:
        """Load documents from the source URLs with enhanced error handling."""
        urls = self._ensure_list(source)
        all_docs = []
        
        for url in urls:
            try:
                logger.info(f"Loading dynamic content from {url}")
                doc = self._load_with_retry(url)
                source_id = Path(urlparse(url).path).stem or "root"
                all_docs.extend(self._add_metadata([doc], source_id))
                logger.info(f"Successfully loaded dynamic content from {url}")
            except Exception as e:
                logger.error(f"Failed to load dynamic content from {url}: {str(e)}")
                raise DocumentLoadError(f"Failed to load dynamic web document: {str(e)}")
        
        return all_docs


# Factory function to create the appropriate loader
def create_document_loader(loader_type: str, project_id: str) -> BaseDocumentLoader:
    """Create a document loader instance based on the type.
    
    Args:
        loader_type: Type of loader to create ('web', 'pdf', 'gdrive', 'csv', 'notion', 'dynamic_web')
        project_id: Project identifier
        
    Returns:
        An instance of the appropriate document loader
        
    Raises:
        ValueError: If loader_type is not recognized
    """
    loaders = {
        'web': WebDocumentLoader,
        'pdf': PDFDocumentLoader,
        'gdrive': GoogleDriveDocumentLoader,
        'csv': CSVDocumentLoader,
        'notion': NotionDocumentLoader,
        'dynamic_web': DynamicWebDocumentLoader
    }
    
    loader_class = loaders.get(loader_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported loader type: {loader_type}")
    
    return loader_class(project_id)


# Example usage functions that maintain backward compatibility
def load_web_document(urls: Union[str, List[str]], project_id: str) -> List[Document]:
    """Backward compatible function for loading web documents."""
    loader = create_document_loader('web', project_id)
    return loader.load(urls)

def load_pdf_document(file_paths: Union[str, List[str]], project_id: str) -> List[Document]:
    """Backward compatible function for loading PDF documents."""
    loader = create_document_loader('pdf', project_id)
    return loader.load(file_paths)

def load_google_drive_document(doc_ids: Union[str, List[str]], service_account_json: str, project_id: str) -> List[Document]:
    """Backward compatible function for loading Google Drive documents."""
    loader = create_document_loader('gdrive', project_id)
    return loader.load(doc_ids, service_account_json=service_account_json)

def load_csv_document(file_paths: Union[str, List[str]], project_id: str) -> List[Document]:
    """Backward compatible function for loading CSV documents."""
    loader = create_document_loader('csv', project_id)
    return loader.load(file_paths)

def load_notion_document(token: str, page_ids: Union[str, List[str]], project_id: str) -> List[Document]:
    """Backward compatible function for loading Notion documents."""
    loader = create_document_loader('notion', project_id)
    return loader.load(page_ids, token=token)

def load_dynamic_web_document(urls: Union[str, List[str]], project_id: str) -> List[Document]:
    """Backward compatible function for loading dynamic web documents."""
    loader = create_document_loader('dynamic_web', project_id)
    return loader.load(urls)
    