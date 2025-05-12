from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import SystemMessage, trim_messages
from typing import Sequence
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore



# define dictionary for state  
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

class AdvanceChatBot:
    def __init__(self):
        self.vector_store = self.__init_vector_store()
        self.retriever = self.__init_retriever()
        self.llm = self.__init_llm_model()
        self.trimmer = self.__init_trimmer()
        self.prompt_template = self.__init_prompt_template()
        self.prompt_template_with_retrieval = self.__init_prompt_template_with_retrieval()

    def __init_vector_store(self):
        """
        Melakukan inisialisasi vector store untuk menyimpan hasil indexing
        document yang nantinya digunakan dalam proses RAG.
        """
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(), # vector db hanya akan disimpan di dalam memory selama runtime
            index_to_docstore_id={},
        )
        return vector_store
    
    def __init_retriever(self):
        """
        Membuat retriever dari dokumen yang telah diindexing ke dalam sebuah 
        vector store
        """
        # load document
        loader = PyPDFLoader('resources/ISP Company FAQ.pdf')
        docs = loader.load()

        # split docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)

        # Store splitted document into vector_store
        self.vector_store.add_documents(documents=all_splits)

        # Create retriever
        return self.vector_store.as_retriever()

    def __init_llm_model(self):
        # load the LLM model
        return ChatOllama(
            temperature=0.5,
            model='gemma3:1b'
        )
    
    def __init_trimmer(self):
        # create trimmer
        return trim_messages(
            max_tokens=1000,
            strategy="last",
            token_counter=self.llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
    
    def __init_prompt_template(self):
        # create prompt template
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant."
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
    
    def __init_prompt_template_with_retrieval(self):
        # create prompt template
        return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are an assistant for question-answering tasks. 
                    Use the following pieces of retrieved context to answer the question. 
                    If you don't know the answer, say that you don't know. 
                    Use three sentences maximum and keep the answer concise.
                    
                    retrieved context:
                    {context}
                """
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
        
    def load_model(self, is_use_rag:bool=False):
        """
        membuat workflow yang akan digunakan untuk menentukan alur kerja dari
        LLM dalam memberikan response dari pertanyaan yang diberikan.
        """
        
        # buat workflow
        workflow = StateGraph(state_schema=State)

        if is_use_rag:
            workflow.add_node("model", self.__generate_rag)
        else:
            workflow.add_node("model", self.__generate)
        
        workflow.add_edge(START, "model")

        # compile workflow dengan memory
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app

    def __generate(self, state: State):
        # lakukan trimmer pada history message
        trimmed_messages = self.trimmer.invoke(state["messages"])

        # masukan trimmed message dan language ke prompt template 
        prompt = self.prompt_template.invoke(
            {"messages": trimmed_messages}
        )

        # generate jawaban dengan model LLM
        response = self.llm.invoke(prompt)

        return {"messages": [response]}
    
    def __generate_rag(self, state: State):
        # lakukan trimmer pada history message
        trimmed_messages = self.trimmer.invoke(state["messages"])

        last_message = state["messages"][-1]
        context = self.retriever.invoke(last_message.content)
        
        # masukan trimmed message dan language ke prompt template 
        prompt = self.prompt_template_with_retrieval.invoke({
            "context": context,
            "messages": trimmed_messages,
        })

        # generate jawaban dengan model LLM
        response = self.llm.invoke(prompt)

        return {"messages": [response]}