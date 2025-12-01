from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings= OpenAIEmbeddings()
#load into pinecone
index_name = "chatbot-storage"
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
query=input('Type in your query:')
docs = vectorstore.similarity_search(query,k=2)
print (docs)