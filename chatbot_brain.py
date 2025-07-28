from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
from termcolor import colored

# Load environment variables from a .env file
load_dotenv()

# Function to create a knowledge base from text documents
def create_knowledge_base():
    print(colored("Starting knowledge base creation...", "green"))
    try:
        # Load all text files from the 'docs' folder
        loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
        documents = loader.load()
        if not documents:
            print(colored("No documents found in the 'docs' folder!", "red"))
            return None
        print(colored(f"Loaded {len(documents)} documents.", "yellow"))

        # Split documents into manageable chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(documents)
        print(colored(f"Created {len(chunks)} chunks from the documents.", "yellow"))

        # Generate embeddings and create a FAISS index
        embeddings = OpenAIEmbeddings()
        print(colored("Generating embeddings...", "cyan"))
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("faiss_index")
        print(colored("Database creation complete! Launch the UI to dive into the RMIT Chatbot adventure!", "green", attrs=["bold"]))
        return db
    except Exception as e:
        print(colored(f"Error creating knowledge base: {e}", "red"))
        return None

# Function to query the chatbot with a given question
def query_chatbot(db, question):
    try:
        if db is None:
            return colored("Error: Knowledge base not loaded.", "red")
        print(colored(f"Processing question: {question}", "blue"))
        llm = OpenAI()
        retriever = db.as_retriever(search_kwargs={"k": 5})
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        response = chain.invoke({"query": question})["result"]
        return response
    except Exception as e:
        return colored(f"Error querying chatbot: {e}", "red")

# Main execution block to create the knowledge base
if __name__ == "__main__":
    print(colored("Initializing RMIT Chatbot Training...", "green", attrs=["bold"]))
    db = create_knowledge_base()