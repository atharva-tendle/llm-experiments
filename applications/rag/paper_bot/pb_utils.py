import cohere
from langchain.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
import urllib.request
import uuid
import streamlit as st

class Vectorstore:
    def __init__(self, co: cohere.Client, paper_url: str) -> None:
        """Initialize the Vectorstore.

        Populates the VectorStore with embeddings and documents.

        Args:
            paper_url (str): URL to the paper's pdf.

        Returns:
            None
        """
        self.co = co
        self.docs = self.load_docs(paper_url) # can be extended for multiple docs (creating a paper db)
        self.retrieve_top_k = 10 # can be tuned for optimal results
        self.rerank_top_k = 3 # can be tuned for optimal results
        self.embed()

    def load_docs(self, paper_url: str) -> list[Document]:
        """ Utilizes the PyPDFLoader to load the paper (pdf) and split it into pages.

        Args:
            paper_url (str): URL to the paper's pdf.

        Returns:
            pages (list[Document]): List of pages represented as langchain docs.

        """
        local_path = "./paper.pdf"
        urllib.request.urlretrieve(paper_url, local_path)
        pdf_loader = PyPDFLoader(local_path)
        # split pages from pdf
        return pdf_loader.load_and_split()

    def embed(self) -> None:
        """Embeds the document chunks using the Cohere Embed API."""

        # Add to vectorstore
        self.vs = FAISS.from_documents(
            documents=self.docs,
            embedding=CohereEmbeddings(),
            )

        print("Embeddings Generated for documents.")

    def retrieve(self, query: str) -> list[dict[str, str]]:
        """ Retrieves document chunks based on the given query.

        Args:
            query (str): The query to retrieve document chunks for.

        Returns:
            retrieved_docs (list[dict[str, str]]): List of dicts representing the retrieved document chunks, with 'text' and 'metadata'.
        """

        # Dense retrieval
        # NOTE: we can do this for other languages as well.
        retrieved_docs = self.vs.similarity_search(query, k=self.retrieve_top_k)

        # Reranking
        rd_page_content = [doc.page_content for doc in retrieved_docs]

        rerank_results = self.co.rerank(
            query=query,
            documents=rd_page_content,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0",
        ).results


        reranked_docs = [
            {
                "text": retrieved_docs[result.index].page_content,
                "metadata": retrieved_docs[result.index].metadata,
            }
            for result in rerank_results
        ]

        return reranked_docs

class PaperBot:
    def __init__(self, co: cohere.Client, vectorstore) -> None:
        """Initializes an instance of the PaperBot class.

        This acts as a chatbot that the user can interact with to discuss research papers (can be extended to any text based pdfs)

        Parameters:
        vectorstore (Vectorstore): An instance of the Vectorstore class.
        """
        self.co = co
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4()) # required to keep context within the cohere chat.

    def chat_completion(self, message):
        """ Runs the chatbot application.

        Opens up an interactive chat window for the user.
        Can exit with the input: quit.
        """
    
        # Generate search queries, if any
        response = self.co.chat(message=message, search_queries_only=True)
        
        stream_responses = []
        # If there are search queries, retrieve document chunks and respond
        if response.search_queries:

            # Retrieve document chunks for each query
            documents = []
            for query in response.search_queries:

                documents.extend(self.vectorstore.retrieve(query.text))

            document_contents = [{"text": doc["text"]} for doc in documents]

            # Use document chunks to respond
            response = self.co.chat_stream(
                message=message,
                model="command-r",
                documents=document_contents,
                conversation_id=self.conversation_id,
            )
        else:
            response = self.co.chat_stream(
                message=message,
                model="command-r",
                conversation_id=self.conversation_id,
            )

        # Build chatbot output
        citations = []
        cited_documents = []

        # Display response
        for event in response:
            if event.event_type == "text-generation":
                stream_responses.append(event.text)
            elif event.event_type == "citation-generation":
                citations.extend(event.citations)
            elif event.event_type == "search-results":
                cited_documents = event.documents

        # Display citations and source documents
        if citations:
            stream_responses.append("\n\nCITATIONS:")
            for citation in citations:
                stream_responses.append(citation)

            stream_responses.append("\nDOCUMENTS:")
            for document in cited_documents:
                stream_responses.append(f"{document['id']}, ")

        for response in stream_responses:
            yield response