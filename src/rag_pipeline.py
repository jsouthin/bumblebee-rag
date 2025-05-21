from typing import List, TypedDict, Optional

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from .vector_store import VectorStoreManager


class State(TypedDict):
    """Type definition for the RAG pipeline state."""
    question: str
    project_id: str
    context: List[Document]
    answer: str


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm: BaseLanguageModel,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        prompt_template: Optional[str] = None
    ):
        """Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector store manager instance
            llm: Language model for generating answers
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            prompt_template: Optional custom prompt template
        """
        self.vector_store = vector_store
        self.llm = llm
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Set up prompt template
        if prompt_template is None:
            prompt_template = '''
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Question: {question} 
            Context: {context} 
            Answer:
            '''
        self.prompt = PromptTemplate.from_template(prompt_template)
        
        # Initialize the graph
        self.graph = self._build_graph()

    def _retrieve(self, state: State) -> dict:
        """Retrieve relevant documents from vector store.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Dict with retrieved context
        """
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        filtered_docs = [
            doc for doc in retrieved_docs 
            if doc.metadata.get("project_id") == state["project_id"]
        ]
        return {"context": filtered_docs}

    def _generate(self, state: State) -> dict:
        """Generate answer using retrieved context.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Dict with generated answer
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({
            "question": state["question"], 
            "context": docs_content
        })
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _build_graph(self) -> StateGraph:
        """Build the RAG pipeline graph.
        
        Returns:
            Compiled state graph
        """
        graph_builder = StateGraph(State)
        graph_builder.add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store.
        
        Args:
            documents: List of documents to add
        """
        splits = self.text_splitter.split_documents(documents)
        self.vector_store.update_store(splits)

    def query(self, question: str, project_id: str) -> str:
        """Run a query through the RAG pipeline.
        
        Args:
            question: The question to answer
            project_id: Project ID for filtering documents
            
        Returns:
            Generated answer
        """
        response = self.graph.invoke({
            "question": question,
            "project_id": project_id
        })
        return response["answer"] 