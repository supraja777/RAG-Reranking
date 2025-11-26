import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
from langchain_core.retrievers import BaseRetriever

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

path = "data/Understanding_Climate_Change.pdf"

def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def encode_pdf(path, chunk_size = 1000, chunk_overlap = 200):
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len
    )

    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLm-L6-v2")

    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore

vectorstore = encode_pdf(path)
llm  = ChatGroq(model="llama-3.3-70b-versatile")

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description = "The relevance score of a document to a query")

def rerank_documents(query : str, docs : List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables = ["query", "doc"],
        template = """On a scale of 1-10, rate the relevance of the following document to the query.append
        Consider the specific context and intent of the query not the specific keyword

        Query : {query}
        Document : {doc}
        Relevance Score: 
        """
    )

    
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)

    scored_docs = []
    for doc in docs:
        input_data = {"query" : query, "doc" : doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score

        try: 
            score = float(score)
        except ValueError:
            score = 0
        scored_docs.append((doc, score))
    
    reranked_docs = sorted(scored_docs, key = lambda x : x[1], reverse = True)
    return [doc for doc, _ in reranked_docs[:top_n]]

query = "What are the impacts of climate change on biodiversity?"
initial_docs = vectorstore.similarity_search(query, k = 15)
reranked_documents = rerank_documents(query, initial_docs)

# print first 3 initial documents
print("Top initial documents:")
for i, doc in enumerate(initial_docs[:3]):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document


# Print results
print(f"Query: {query}\n")
print("Top reranked documents:")
for i, doc in enumerate(reranked_documents):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:200] + "...")  # Print first 200 characters of each document










