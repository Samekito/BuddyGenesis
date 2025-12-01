from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferWindowMemory
from langchain_pinecone import PineconeVectorStore

import chainlit as cl
from chainlit.types import ThreadDict
from typing import Optional
from operator import itemgetter
from langchain_openai import OpenAIEmbeddings


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferWindowMemory
    model = ChatOpenAI(temperature=0,streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful virtual assistant ,named 'SOC Buddy', assigned to the School of Computing in the Federal University of Technology Akure,Nigeria.You are to use information provided in 'Reference' as a guide when responding to messages.Absolutely do not mention this guide to the user.If it lacks needed information,explicitly say you do not know."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    
    cl.user_session.set("runnable", runnable)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: dict[str, str],    
  default_user: cl.User,
) -> Optional[cl.User]: 
    default_user.identifier=raw_user_data['name']
    return default_user


@cl.on_chat_start
async def on_chat_start():
    app_user = cl.user_session.get("user")
    await cl.Message(f"Hello {app_user.identifier}.\n\nI am SOC Buddy,an AI Virtual Assistant designated to the School of Computing,FUTA.\nHow may I be of service to you today?").send()
    cl.user_session.set("memory", ConversationBufferWindowMemory(return_messages=True,k=2))
    setup_runnable()
    

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferWindowMemory(return_messages=True,k=2)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    embeddings= OpenAIEmbeddings()
    index_name = "venom"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(
     search_kwargs={"k":3}
    )
    vectorstore_response=retriever.invoke(message.content)
    print(vectorstore_response) 
    A=str(vectorstore_response[0])
    B=str(vectorstore_response[1])
    def string_stripper(string1,string2):
        x=string1.find('metadata')
        y=string1.find('=')
        string1_stripped=string1[y+1:x-2]
        x=string2.find('metadata')
        y=string2.find('=')
        string2_stripped=string2[y+1:x-2]
        return string1_stripped,string2_stripped
    vectorstore_response=string_stripper(A,B)
    print(f"\n {vectorstore_response}")
    memory = cl.user_session.get("memory")  # type: ConversationBufferWindowMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable
    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question":f"Reference:\n{vectorstore_response}\nQuestion:\n{message.content}"},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)
    
    await res.send()
    memory.chat_memory.add_user_message(message.content)
    print(message.content)
    memory.chat_memory.add_ai_message(res.content)
    print(res.content)
    print(memory.load_memory_variables({}))
