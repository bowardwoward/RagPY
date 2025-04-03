# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import MarkdownTextSplitter

from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import os
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """A class for loading different document types."""

    @staticmethod
    def load(file_path: str):
        """
        Load a document based on its file extension.

        Args:
            file_path: Path to the document file

        Returns:
            List of Document objects
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return PyPDFLoader(file_path=file_path).load()
        elif file_extension in ['.md', '.markdown']:
            return TextLoader(file_path=file_path).load()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


class ChatPDF:
    """A class for handling document ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large", vector_store_type: str = "faiss"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        
        Args:
            llm_model: The Ollama LLM model to use
            embedding_model: The embedding model to use
            vector_store_type: Type of vector store to use ('faiss' or 'chroma')
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store_type = vector_store_type.lower()
        
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
            Answer the user's question based on the following context information. For technical implementation questions, provide guidance using the approved tech stack: NestJS, ReactJS, React Native/Expo, and PostgreSQL.
      
            Context:
            {context}
            
            Question:
            {question}
             """
        )
        self.vector_store = None
        self.retriever = None
        
        # Directory paths for different vector stores
        self.faiss_index_path = "faiss_index"
        self.chroma_db_path = "chroma_db"

    def ingest(self, file_path: str):
        """
        Ingest a document file, split its contents, and store the embeddings in the vector store.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        logger.info(
            f"Starting ingestion for {file_extension} file: {file_path}")

        try:
            docs = DocumentLoader.load(file_path)
            chunks = self.text_splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)

            if self.vector_store is None:
                if self.vector_store_type == "faiss":
                    self.vector_store = FAISS.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                    )
                    # Save the FAISS index to disk
                    os.makedirs(self.faiss_index_path, exist_ok=True)
                    self.vector_store.save_local(self.faiss_index_path)
                else:  # default to Chroma
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory=self.chroma_db_path,
                    )
            else:
                if self.vector_store_type == "faiss":
                    # Add documents to existing FAISS store
                    self.vector_store.add_documents(chunks)
                    # Save updated index
                    self.vector_store.save_local(self.faiss_index_path)
                else:
                    # Add documents to existing Chroma store
                    self.vector_store.add_documents(chunks)

            logger.info(
                f"Ingestion completed for {file_path}. Document embeddings stored successfully in {self.vector_store_type}.")
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            raise

    def load_vector_store(self):
        """
        Load an existing vector store from disk if available.
        """
        try:
            if self.vector_store_type == "faiss" and os.path.exists(self.faiss_index_path):
                logger.info(f"Loading existing FAISS index from {self.faiss_index_path}")
                self.vector_store = FAISS.load_local(
                    self.faiss_index_path, 
                    self.embeddings
                )
                return True
            elif self.vector_store_type == "chroma" and os.path.exists(self.chroma_db_path):
                logger.info(f"Loading existing Chroma DB from {self.chroma_db_path}")
                self.vector_store = Chroma(
                    persist_directory=self.chroma_db_path,
                    embedding_function=self.embeddings
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            # Try to load existing vector store
            if not self.load_vector_store():
                raise ValueError(
                    "No vector store found. Please ingest a document first.")

        if not self.retriever:
            if self.vector_store_type == "faiss":
                # FAISS doesn't support similarity_score_threshold directly like Chroma
                # We'll retrieve more documents and manually filter if needed
                self.retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": k}
                )
            else:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": k, "score_threshold": score_threshold},
                )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        # For FAISS, manually filter by score if score_threshold is specified
        if self.vector_store_type == "faiss" and hasattr(retrieved_docs[0], "metadata") and "score" in retrieved_docs[0].metadata:
            retrieved_docs = [doc for doc in retrieved_docs if doc.metadata.get("score", 1.0) >= score_threshold]
            if not retrieved_docs:
                return "No sufficiently relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
        
    def set_vector_store_type(self, vector_store_type: str):
        """
        Change the vector store type.
        
        Args:
            vector_store_type: The vector store type ('faiss' or 'chroma')
        """
        if vector_store_type.lower() not in ["faiss", "chroma"]:
            raise ValueError("Vector store type must be 'faiss' or 'chroma'")
            
        if self.vector_store_type != vector_store_type.lower():
            logger.info(f"Changing vector store type from {self.vector_store_type} to {vector_store_type.lower()}")
            self.vector_store_type = vector_store_type.lower()
            self.vector_store = None
            self.retriever = None