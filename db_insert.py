from langchain_community.document_loaders import DirectoryLoader,TextLoader,PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

embeddings= OpenAIEmbeddings(api_key = os.getenv("OPENAI_API_KEY"))
index_name = "venom"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# load/read the document
directory= "knowledge_base/"
# loader = PyPDFLoader("knowledge_base/EEE HANDBOOK 2014 edited-2.pdf")
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
# loaded_documents = loader.load_and_split(text_splitter)
loader = DirectoryLoader(directory,glob="*.txt", loader_cls=TextLoader)
loaded_documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=10)

print("Document has been loaded.")
# split it into chunks
print("Splitting documents")
splitted_docs = text_splitter.split_documents(loaded_documents)
print("Document splitted,Generating embeddings..")
# create the open-source embedding function
embeddings= OpenAIEmbeddings()
#load into pinecone
index_name = "venom"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings,pinecone_api_key='efac0c6f-dabf-4cec-9c74-29ffa4c639ce')
#vectorstore.delete(delete_all=True)
vectorstore.add_documents(splitted_docs)
print("Embeddings generated and loaded into vectorstore...")

