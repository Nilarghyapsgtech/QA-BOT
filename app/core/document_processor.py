"""Document processing module for loading and chunking documents"""
import tempfile
from typing import BinaryIO
from pathlib import Path

from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings
from app.utils.logger import get_logger


logger=get_logger(__name__)

class DocumentProcessor:
    """Process Documents For RAG Pipeline"""
    SUPPORTED_EXTENSION={".pdf",".txt",".csv"}
    def __init__(
        self,
        chunk_size:int|None=None,
        chunk_overlap:int|None=None
    ):
        """Initialize document processor.
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
        """
        settings=get_settings()
        self.chunk_size=chunk_size or settings.chunk_size
        self.chunk_overlap=chunk_overlap or settings.chunk_overlap

        self.text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n","\n"," ",".",",",]
        )

        logger.info(
            f"DocumentProcessor initialized with chunk_size={self.chunk_size} ,chunk_overlap={self.chunk_overlap}"
        )

    def load_pdf(self,file_path:str|Path)->list[Document]:
            """Load a PDF file.
            Args:
                file_path: Path to PDF file
            
            Returns :
                List of Document objects
            """
            path=Path(file_path)
            logger.info(f"Loading PDF: {file_path.name}")
            
            loader=PyPDFLoader(str(path))
            documents=loader.load()
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
        
        
    def load_text(self,file_path:str|Path)->list[Document]:
            """Load a Text file.
            Args:
                file_path: Path to Text file
            
            Returns :
                List of Document objects
            """
            path=Path(file_path)
            logger.info(f"Loading Text: {file_path.name}")
            
            loader=TextLoader(str(path))
            documents=loader.load()
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
        
        
    def load_csv(self,file_path:str|Path)->list[Document]:
            """Load a CSV file.
            Args:
                file_path: Path to CSV file
            Returns:
                List of Document objects (one per row)
            """
            path=Path(file_path)
            logger.info(f"Loading CSV: {file_path.name}")

            loader=CSVLoader(str(path))
            documents=loader.load()

            logger.info(f"Loaded {len(documents)} page from {file_path.name}")
            return documents
        
    def load_file(self,file_path:str|Path)->list[Document]:
            """Load a file based on its extension.
            Args:
                file_path: Path to file
            
            Returns:
                List of Document objects

            Raises:
                ValueError: If file extension is not supported
            """
            file_path=Path(file_path)
            extension=file_path.suffix.lower()

            if extension not in self.SUPPORTED_EXTENSION:
                raise ValueError(
                    f"Unsupported file extension: {extension}."
                    f"Supported: {self.SUPPORTED_EXTENSION}"
                )
            loaders ={
                ".pdf": self.load_pdf,
                ".txt": self.load_text,
                ".csv": self.load_csv
            }

            return loaders[extension](file_path)
        
    def load_from_upload(self,file: BinaryIO, filename: str)->list[Document]:
            """Load Document from uploaded file.

            Args:
                file: File-like object
                filename: Original filename

            Returns:
                List of Document objects
            """
            extension=Path(filename).suffix.lower()
            if extension not in self.SUPPORTED_EXTENSION:
                raise ValueError(
                    f"Unsupported file extension: {extension}."
                    f"Supported: {self.SUPPORTED_EXTENSION}"
                )
            
            # Saving to temporary file for processing
            with tempfile.NamedTemporaryFile(delete=False,suffix=extension) as tmp_file:
                tmp_file.write(file.read())
                tmp_path= tmp_file.name
            
            try:
                documents=self.load_file(tmp_path)
                for doc in documents:
                    doc.metadata["source"]=filename

                return documents
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        
    def split_documents(self,documents:list[Document])->list[Document]:
            """Split Documents into chunks.
            
            Args:
                documents: List of Document objects
            
            Returns:
                List of chunked Document objects
            """

            logger.info(f"Splitting {len(documents)} documents into chunks")
            chunks=self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")

            return chunks
        
    def process_file(self,file_path:str|Path)->list[Document]:
            """Load and plit a file in one step.
            Args:
                file_path: Path to file
            Returns:
                List of chunked Document objects
            """
            documents= self.load_file(file_path)
            return self.split_documents(documents)
        
    def process_upload(self,file:BinaryIO,filename:str)->list[Document]:
        """Load and split an uploaded file.

        Args:
            file: File-like object
            filename: Original filename

        Returns:
            List of chunked Document objects
        """
        documents=self.load_from_upload(file,filename)
        return self.split_documents(documents)









        
        
        

