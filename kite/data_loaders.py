"""
Document Loaders Module
Provides loaders for various document types: PDF, DOCX, CSV, HTML.
"""

import os
from typing import List, Dict, Optional
import logging

class DocumentLoader:
    """Base class for document loaders."""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Load text from PDF."""
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            return "Error: PyPDF2 not installed. Run 'pip install PyPDF2'"
        except Exception as e:
            return f"Error loading PDF: {str(e)}"

    @staticmethod
    def load_docx(file_path: str) -> str:
        """Load text from DOCX."""
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            return "Error: python-docx not installed. Run 'pip install python-docx'"
        except Exception as e:
            return f"Error loading DOCX: {str(e)}"

    @staticmethod
    def load_csv(file_path: str) -> str:
        """Load text from CSV (summary or specific columns)."""
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        except ImportError:
            import csv
            text = ""
            with open(file_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += ", ".join(row) + "\n"
            return text
        except Exception as e:
            return f"Error loading CSV: {str(e)}"

    @staticmethod
    def load_html(file_path: str) -> str:
        """Load text from HTML."""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator=' ')
        except ImportError:
            return "Error: beautifulsoup4 not installed. Run 'pip install beautifulsoup4'"
        except Exception as e:
            return f"Error loading HTML: {str(e)}"

    @classmethod
    def load_any(cls, file_path: str) -> str:
        """Auto-detect and load any supported file type."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return cls.load_pdf(file_path)
        elif ext == '.docx':
            return cls.load_docx(file_path)
        elif ext == '.csv':
            return cls.load_csv(file_path)
        elif ext == '.json':
            # Add JSON support
            try:
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            except Exception as e:
                return f"Error loading JSON: {str(e)}"
        elif ext in ['.html', '.htm']:
            return cls.load_html(file_path)
        elif ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return None for unsupported formats instead of error message
            return None

    @classmethod
    def load_directory(cls, directory_path: str) -> Dict[str, str]:
        """Load all supported files from a directory."""
        results = {}
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                results[filename] = cls.load_any(file_path)
        return results
