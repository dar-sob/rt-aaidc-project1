import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import hashlib, unicodedata, re


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """


    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")


    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size = chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_text(text)             

        return chunks


    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of dictionary of documents
        """

        if not documents:
            logging.warning('No documents to process.')
            return

        print(f"Processing {len(documents)} documents...")

        try:
            # Loop over the documents
            for nr_doc, doc in enumerate(documents):     

                if not isinstance(doc, dict) or 'content' not in doc:
                    logging.error(f'Info add_dokuments: Skipping invalid document format {doc} in add_documents.', exc_info=True)
                    continue
                
                # Dividing the text to chunks
                content = doc.get('content', '')            
                chunks = self.chunk_text(content)

                # Embeddings chunks
                embeddings = self.embedding_model.encode(chunks, batch_size=16, show_progress_bar=True).tolist()

                # Lists of metadatas and ids of length like chunks
                metadatas = [doc.get('metadata', {}) for _ in chunks]
                ids = list(f'id_doc_nr_{nr_doc}_chunk_nr_{i}' for i in range(len(chunks)))

                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )

        except Exception as e:
            logging.exception(f"Failed to add chunks for document {nr_doc}: {e}")

        print("Documents added to vector database")


    def search(self, query: str, n_results: int = 5, similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            similarity_threshold: 0.0 the same, 1.0 completely different 

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        
        # Check if query is str
        if not query or not isinstance(query, str):
            logging.warning('Invalid query provided: %s', query)
            return{"documents": [], "distances": [], "metadatas": [], "ids": []}

        try:
            # Create an embedding for the query
            query_embeddings = self.embedding_model.encode([query])[0]

            # Search the ChromaDB collection
            results = self.collection.query(
                query_embeddings=[query_embeddings],
                n_results=n_results,
                include=["documents", "distances", "metadatas"]
            )
            # Documents with another atributs
            docs    = results.get("documents", [[]])[0]
            dists   = results.get("distances", [[]])[0]
            metas   = results.get("metadatas", [[]])[0]
            ids     = results.get("ids", [[]])[0]

            # Select docs with the lowest similarity threshold
            filtered_doc = {key:[] for key in ["documents", "distances", "metadatas", "ids"]}

            for doc, dist, meta, id_ in zip(docs, dists, metas, ids):                
                if dist <= similarity_threshold:
                    for key, val in zip(
                        ["documents", "distances", "metadatas", "ids"],
                        [doc, dist, meta, id_]
                    ):
                        filtered_doc[key].append(val)

            logging.info(
                'Search complited for query=%s, total results = %d, returned=%d', 
                query, len(docs), len(filtered_doc["documents"])
            ) 
            
            return filtered_doc            
        
        except Exception as e:
            logging.error("Search failed for query='%s': %s", query, str(e), exc_info=True)
            return {"documents": [], "distances": [], "metadatas": [], "ids": []}
