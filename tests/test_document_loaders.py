import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.document_loaders import (
    WebDocumentLoader,
    PDFDocumentLoader,
    GoogleDriveDocumentLoader,
    CSVDocumentLoader,
    NotionDocumentLoader,
    DynamicWebDocumentLoader,
    DocumentLoadError,
    create_document_loader
)
from langchain.schema import Document

# Common test fixtures
@pytest.fixture
def project_id():
    return "test_project"

@pytest.fixture
def sample_document():
    return Document(
        page_content="Test content",
        metadata={"source": "test"}
    )

@pytest.fixture
def mock_time():
    with patch('time.time') as mock_time:
        mock_time.return_value = 1234567890
        yield mock_time

# Test base functionality across all loaders
@pytest.mark.parametrize("loader_class", [
    WebDocumentLoader,
    PDFDocumentLoader,
    GoogleDriveDocumentLoader,
    CSVDocumentLoader,
    NotionDocumentLoader,
    DynamicWebDocumentLoader
])
def test_loader_initialization(loader_class, project_id):
    loader = loader_class(project_id)
    assert loader.project_id == project_id

def test_ensure_list_conversion():
    loader = WebDocumentLoader("test_project")
    # Test single string
    assert loader._ensure_list("test") == ["test"]
    # Test list
    test_list = ["test1", "test2"]
    assert loader._ensure_list(test_list) == test_list

# WebDocumentLoader tests
class TestWebDocumentLoader:
    @pytest.fixture
    def web_loader(self, project_id):
        return WebDocumentLoader(project_id)

    def test_successful_load(self, web_loader, sample_document, mock_time):
        with patch('src.document_loaders.WebBaseLoader') as mock_web_loader:
            # Setup mock
            mock_instance = Mock()
            mock_instance.load.return_value = [sample_document]
            mock_web_loader.return_value = mock_instance

            # Test
            url = "https://example.com"
            docs = web_loader.load(url)

            # Verify
            assert len(docs) == 1
            assert docs[0].page_content == "Test content"
            assert docs[0].metadata["project_id"] == "test_project"
            assert docs[0].metadata["timestamp_added"] == 1234567890

    def test_load_multiple_urls(self, web_loader, sample_document):
        with patch('src.document_loaders.WebBaseLoader') as mock_web_loader:
            # Setup mock
            mock_instance = Mock()
            mock_instance.load.return_value = [sample_document]
            mock_web_loader.return_value = mock_instance

            # Test
            urls = ["https://example1.com", "https://example2.com"]
            docs = web_loader.load(urls)

            # Verify
            assert len(docs) == 2
            assert mock_web_loader.call_count == 2

    def test_load_failure(self, web_loader):
        with patch('src.document_loaders.WebBaseLoader') as mock_web_loader:
            # Setup mock to raise exception
            mock_instance = Mock()
            mock_instance.load.side_effect = Exception("Failed to load")
            mock_web_loader.return_value = mock_instance

            # Test
            with pytest.raises(DocumentLoadError) as exc_info:
                web_loader.load("https://example.com")
            assert "Failed to load web document" in str(exc_info.value)

# PDFDocumentLoader tests
class TestPDFDocumentLoader:
    @pytest.fixture
    def pdf_loader(self, project_id):
        return PDFDocumentLoader(project_id)

    def test_successful_load(self, pdf_loader, sample_document, mock_time):
        with patch('src.document_loaders.PyPDFLoader') as mock_pdf_loader, \
             patch('src.document_loaders.sanitize_filepath') as mock_sanitize, \
             patch('src.document_loaders.validate_file_type') as mock_validate_type:
            # Setup mocks
            mock_instance = Mock()
            mock_instance.load.return_value = [sample_document]
            mock_pdf_loader.return_value = mock_instance
            mock_sanitize.return_value = Path("test.pdf")
            mock_validate_type.return_value = True

            # Test
            docs = pdf_loader.load("test.pdf")

            # Verify
            assert len(docs) == 1
            assert docs[0].metadata["project_id"] == "test_project"
            assert docs[0].metadata["source_id"] == "test"
            assert docs[0].metadata["timestamp_added"] == 1234567890

    def test_load_failure(self, pdf_loader):
        with patch('src.document_loaders.PyPDFLoader') as mock_pdf_loader, \
             patch('src.document_loaders.sanitize_filepath') as mock_sanitize, \
             patch('src.document_loaders.validate_file_type') as mock_validate_type:
            # Setup mock to raise exception
            mock_instance = Mock()
            mock_instance.load.side_effect = Exception("Failed to load PDF")
            mock_pdf_loader.return_value = mock_instance
            mock_sanitize.return_value = Path("test.pdf")
            mock_validate_type.return_value = True

            # Test
            with pytest.raises(DocumentLoadError) as exc_info:
                pdf_loader.load("test.pdf")
            assert "Failed to load PDF document" in str(exc_info.value)

# GoogleDriveDocumentLoader tests
class TestGoogleDriveDocumentLoader:
    @pytest.fixture
    def gdrive_loader(self, project_id):
        return GoogleDriveDocumentLoader(project_id)

    def test_successful_load(self, gdrive_loader, sample_document, mock_time):
        with patch('src.document_loaders.GoogleDriveLoader') as mock_gdrive_loader:
            # Setup mock
            mock_instance = Mock()
            mock_instance.load.return_value = [sample_document]
            mock_gdrive_loader.return_value = mock_instance

            # Test
            docs = gdrive_loader.load(
                "doc_id",
                service_account_json="credentials.json"
            )

            # Verify
            assert len(docs) == 1
            assert docs[0].metadata["project_id"] == "test_project"
            mock_gdrive_loader.assert_called_once()

# Factory function tests
def test_create_document_loader(project_id):
    # Test valid loader types
    assert isinstance(create_document_loader("web", project_id), WebDocumentLoader)
    assert isinstance(create_document_loader("pdf", project_id), PDFDocumentLoader)
    
    # Test invalid loader type
    with pytest.raises(ValueError) as exc_info:
        create_document_loader("invalid_type", project_id)
    assert "Unsupported loader type" in str(exc_info.value)

# Test metadata consistency
def test_metadata_consistency(mock_time):
    # Create test documents with different loaders
    web_loader = WebDocumentLoader("test_project")
    pdf_loader = PDFDocumentLoader("test_project")
    
    # Mock document loading
    with patch('src.document_loaders.WebBaseLoader') as mock_web_loader, \
         patch('src.document_loaders.PyPDFLoader') as mock_pdf_loader, \
         patch('src.document_loaders.sanitize_filepath') as mock_sanitize, \
         patch('src.document_loaders.validate_file_type') as mock_validate_type:
        
        # Setup mocks
        doc = Document(page_content="Test", metadata={})
        mock_web_instance = Mock()
        mock_pdf_instance = Mock()
        mock_web_instance.load.return_value = [doc]
        mock_pdf_instance.load.return_value = [doc]
        mock_web_loader.return_value = mock_web_instance
        mock_pdf_loader.return_value = mock_pdf_instance
        mock_sanitize.return_value = Path("test.pdf")
        mock_validate_type.return_value = True
        
        # Load documents
        web_doc = web_loader.load("https://example.com")[0]
        pdf_doc = pdf_loader.load("test.pdf")[0]
        
        # Verify metadata consistency
        assert web_doc.metadata["project_id"] == pdf_doc.metadata["project_id"]
        assert web_doc.metadata["timestamp_added"] == pdf_doc.metadata["timestamp_added"]
        assert "source_id" in web_doc.metadata
        assert "source_id" in pdf_doc.metadata 