# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

import chainlit as cl  # importing chainlit for our app
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
#from utils.custom_retriver import CustomQDrant
#from starters import set_starters


load_dotenv()




RAG_PROMPT = """
CONTEXT:
{context}
QUERY:
{question}
You are an expert in financial statements. Use the provided context to answer the user's query. 
Answer questions only using provided context not your prior knowladge. 
If you do not know the answer, or cannot answer, please respond with I don't know.
In your answer never use term "based on provided context"
"""

data_path = "data/United states securities and exchange commission 10-Q.pdf"
docs = PyMuPDFLoader(data_path).load()
openai_chat_model = ChatOpenAI(model="gpt-4o", streaming=True) #gpt-4o

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="Airbnb",
    score_threshold=0.5
    
)

qdrant_retriever = qdrant_vectorstore.as_retriever()

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)




@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():



    cl.user_session.set("chain", retrieval_augmented_qa_chain, )





@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    resp = await chain.ainvoke({"question" : message.content})
    source_documents = resp["context"] 

    text_elements = [] 

    resp_msg = resp["response"].content


    msg = cl.Message(content=resp_msg)

    #print(msg.content)
    await msg.send()


    """async for chunk in msg.content:
    
        if token := chunk.choices[0].delta.content or "":
            await msg.stream_token(token)
    await msg.update()"""

    #async for chunk in chain:
    #    if token:=