# RAG Reranking

**RAG Reranking** is a high-performance **Retrieval-Augmented Generation (RAG)** framework for **semantic document retrieval** and **context-aware reranking**. It combines **vector-based similarity search** with **large language model reasoning** to deliver precise, knowledge-grounded results from PDF documents and other text corpora.

RAG Reranking addresses key challenges in information retrieval:  
- Efficiently retrieving relevant documents from large-scale datasets.  
- Leveraging **LLM-powered scoring** to reorder results for higher relevance.  
- Minimizing hallucinations and improving answer accuracy in downstream NLP applications.  

This pipeline is ideal for **climate research, academic literature analysis, enterprise knowledge bases**, and any domain requiring high-precision document retrieval and question answering.

---

## 🔹 Key Features

- **PDF & Document Processing:** Load, clean, and split multi-page PDFs into manageable chunks.  
- **Semantic Embeddings:** Transform textual data into dense vector representations using HuggingFace embeddings.  
- **Vector Search & Indexing:** Fast similarity search using **FAISS**, scalable for large document collections.  
- **LLM-Based Reranking:** Context-aware reranking of documents with structured output from modern LLMs.  
- **Customizable Retrieval:** Fine-tune chunk sizes, overlaps, and top-N document retrieval.  
- **RAG Pipeline Ready:** Integrates seamlessly into downstream NLP tasks like **question answering**, **summarization**, and **knowledge extraction**.

---

## 🛠 Technology Stack
-  Python 3.10+ – Modern, efficient development environment.

- LangChain – Orchestrates document processing and LLM interactions.

-  FAISS – High-speed vector search and indexing.

-  HuggingFace Sentence Transformers – Semantic embeddings for high-quality retrieval.

-  ChatGroq LLM – Structured LLM scoring for reranking.

---

## 🛠 Installation

```bash
git clone https://github.com/yourusername/RAG-Reranking.git
cd RAG-Reranking
pip install -r requirements.txt
```
---
## 🚀 Usage

Add your PDF documents to the data/ folder.

Update the path variable in app.py to your target document.

Run the main script:

```bash
python app.py
```

Input a query to retrieve and rerank relevant documents.


## 🚀 STEPS


Step 1: FAISS retrieves top relevant documents based on vector similarity.

Step 2: The LLM reranks documents based on contextual relevance and intent.

Result: Highly accurate, top-N documents most relevant to your query.

---


