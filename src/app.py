import os
from pathlib import Path
import PyPDF2
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Load environment variables
load_dotenv()

# Load logging
logging.basicConfig(level=logging.INFO)


def load_documents(directory: Path | None = None) -> List[Dict[str, Any]]:
    """
    Load documents (format(txt, pdf)) from directory, for demonstration.
    
    Args: 
        Path to dictionary

    Returns:
        List of dictionaries of sample documents and metadata.

        Lists of dicts:        
        dokument_dict ={
            "content": Content of one document,
            "metadata": {
                "title": Title of document from name of file,
                "file_type": Based on suffix of name of file (txt, pdf)
    """

    # If directory path is not provided
    if directory is None:
        directory = Path(__file__).resolve().parent.parent / "data"

    # Info - Error - Not exists, Not dir
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory} with documents, not exist.") 
    if not directory.is_dir():
        raise NotADirectoryError(f"Directory {directory} is not a dir.")

    results = []

    # Iterate over documents
    for doc in directory.iterdir():
        try:
            # TXT
            if doc.suffix.lower() == ".txt":
                content = doc.read_text(encoding="utf-8")            
            # PDF
            elif doc.suffix.lower() == ".pdf":
                with open(doc, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    content = "\n".join(page.extract_text() or "" for page in reader.pages)

            # Shape of document dictionary        
            dokument_dict ={
                "content": content,
                "metadata": {
                    "title": doc.stem,
                    "file_type": doc.suffix.lower()
                }
            }  

            results.append(dokument_dict)

        except Exception as e:
            # Make info bat not stop def
            logging.warning(f'Problem with load dokument {doc.name}: {e}', exc_info=True)    

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""

        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # System prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
        
            """
            You are a helpful assistant that can answer the users questions given some relevant documents..

            Your task is as follows:
            Given the some documents that should be relevant to the user's question, answer the user's question.

            Ensure your response follows these rules:
            - Only answer questions based on the provided documents.
            - If the user's question is not related to the documents, then you SHOULD NOT answer the question. Say "The question is not answerable given the documents".
            - Never answer a question from your own knowledge.

            Follow these style and tone guidelines in your response:
            - Use clear, concise language with bullet points where appropriate.

            Structure your response as follows:
            - Provide answers in markdown format.
            - Provide concise answers in bullet points when relevant.

            Here is the content you need to work with:
            <<<BEGIN CONTENT>>>
            ```
            Relevant documents:

            {context}

            User's question:

            {question}
            ```
            <<<END CONTENT>>>

            Now perform the task as instructed above.
            """
        )  

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")


    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """

        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )


    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: Lists of dicts:
            {
            "content": Content of one document,
            "metadata": {
                "title": Title of document from name of file,
                "file_type": Based on suffix of name of file (txt, pdf)}
            }
        
        """
        self.vector_db.add_documents(documents)


    def query(self, query: str, n_results: int = 3, similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Query the RAG assistant.
        Finds documents using vectordb.search: 
        {key:
            "documents", - context of documents
            "distances", - cosine distance
            "metadatas", - metadata like title, file_type, 
            "ids"
        }
        Join context (documents). 
        Invoke llm with context and query and grab aswer.   

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve
            similarity_threshold: 0.0 the same, 1.0 completely different 

        Returns:
            Dictionary containing the answer and retrieved context and metadata.
            {"answer": llm_answer , 
            "context": context, 
            "metadata": {"retrieved_docs"}
            }
        """ 
        try:
            # Find matching documents (chunks)
            retrieved_chunks = self.vector_db.search(
                                    query=query, 
                                    n_results=n_results, 
                                    similarity_threshold=similarity_threshold
            )

            documents = retrieved_chunks.get("documents", [[]])
            documents = [str(doc) for doc in documents if doc]

            # Empty documents
            if not documents:
                return {
                "answer": "Info (query): There is no documents in base.",
                "context": "",
                "metadata": {"retrieved_docs": 0}
            }
            
            context = "\n\n".join(documents)

            llm_answer = self.chain.invoke({
                "context": context,
                "question": query
            })                      

            return {"answer": llm_answer if llm_answer else '', "context": context, "metadata": {"retrieved_docs": len(documents)}}
        
        except Exception as e:
            return{
                "answer": f"Info (query): An error occured while processing your query: {str(e)}",
                "context": "",
                "metadata": {"error": True}
            }


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} sample documents")
        
        # Add documents to data base 
        print("\nAdd documents to db...")
        assistant.add_documents(sample_docs)
        print("\nDB done...")

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.query(query=question)
                print(result['answer'].strip())

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()