from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import bs4
from google.colab import userdata
import getpass
import os

#Your token should be here
#os.environ["OPENAI_API_KEY"] = userdata.get('OPEN_AI_API')
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
os.environ["GROQ_API_KEY"]= userdata.get('groq_api_key')

#This is my llm model [GROQ]
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Load docs
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
web_paths = ("https://docs.llamaindex.ai/en/stable/",),
#bs_kwargs = {"parse_only": bs4_strainer}

)
data = loader.load()

#Huggingface embedding
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


#Splitting my documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(data)

# put my chunk documents into vector database
vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

# chaining it

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

output=rag_chain.invoke("Give me basic llamaindex code ?")
print(output)
