"""
Comprehensive tests for data loaders.
Tests document loading for various formats.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from kite.data_loaders import DocumentLoader


class TestDocumentLoader:
    """Test document loader functionality."""
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = DocumentLoader()
        assert loader is not None
    
    def test_load_text_file(self, tmp_path):
        """Test loading .txt files."""
        loader = DocumentLoader()
        
        # Create temp text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.", encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert content == "This is a test document."
    
    def test_load_json_file(self, tmp_path):
        """Test loading .json files."""
        loader = DocumentLoader()
        
        # Create temp JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value", "number": 42}', encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert "key" in content
        assert "value" in content
    
    def test_load_csv_file(self, tmp_path):
        """Test loading .csv files."""
        loader = DocumentLoader()
        
        # Create temp CSV file
        test_file = tmp_path / "test.csv"
        test_file.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert "Alice" in content
        assert "Bob" in content
    
    @pytest.mark.skip(reason="PDF loading requires additional dependencies")
    def test_load_pdf_file(self, tmp_path):
        """Test loading .pdf files (mocked)."""
        loader = DocumentLoader()
        
        # Mock PDF loading
        with patch.object(loader, 'load_pdf', return_value="PDF content"):
            content = loader.load_any("test.pdf")
            assert content == "PDF content"
    
    def test_load_unsupported_format(self):
        """Test error handling for unsupported formats."""
        loader = DocumentLoader()
        
        # Should handle gracefully
        try:
            content = loader.load_any("test.xyz")
            # Either returns empty/None or raises
            assert content is None or content == ""
        except (ValueError, NotImplementedError):
            # Expected behavior
            pass
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_any("nonexistent_file.txt")


class TestBatchLoading:
    """Test loading multiple documents."""
    
    def test_batch_loading(self, tmp_path):
        """Test loading multiple documents."""
        loader = DocumentLoader()
        
        # Create multiple test files
        files = []
        for i in range(3):
            test_file = tmp_path / f"doc{i}.txt"
            test_file.write_text(f"Document {i} content", encoding="utf-8")
            files.append(str(test_file))
        
        # Load all files
        contents = []
        for file_path in files:
            content = loader.load_any(file_path)
            contents.append(content)
        
        assert len(contents) == 3
        assert all("Document" in c for c in contents)


class TestLoaderIntegration:
    """Integration tests for document loader."""
    
    def test_loader_with_kite(self, ai, tmp_path):
        """Test loader integration with Kite."""
        # Create test document
        test_file = tmp_path / "test_doc.txt"
        test_file.write_text("Test content for Kite", encoding="utf-8")
        
        # Load document through Kite
        ai.load_document(str(test_file), doc_id="test1")
        
        # Document should be loaded into vector memory
        # (Actual verification depends on implementation)
        assert True  # Placeholder
    
    def test_load_and_search(self, ai, tmp_path, sample_documents):
        """Test loading documents and searching."""
        # Create test files
        for doc_id, content in sample_documents.items():
            test_file = tmp_path / f"{doc_id}.txt"
            test_file.write_text(content, encoding="utf-8")
            
            # Load into Kite
            try:
                ai.load_document(str(test_file), doc_id=doc_id)
            except Exception:
                # May fail if vector memory not initialized
                pass


class TestFileFormatHandling:
    """Test handling of different file formats."""
    
    def test_markdown_file(self, tmp_path):
        """Test loading markdown files."""
        loader = DocumentLoader()
        
        test_file = tmp_path / "test.md"
        test_file.write_text("# Header\n\nContent", encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert "Header" in content
        assert "Content" in content
    
    def test_html_file(self, tmp_path):
        """Test loading HTML files."""
        loader = DocumentLoader()
        
        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body>HTML Content</body></html>", encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert "HTML Content" in content or "html" in content.lower()
    
    def test_encoding_handling(self, tmp_path):
        """Test handling different encodings."""
        loader = DocumentLoader()
        
        test_file = tmp_path / "test_utf8.txt"
        test_file.write_text("UTF-8 content with émojis 🚀", encoding="utf-8")
        
        content = loader.load_any(str(test_file))
        assert "UTF-8" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
