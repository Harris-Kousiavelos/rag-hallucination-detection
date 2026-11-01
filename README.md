# rag-hallucination-detection
MSc thesis: detection of faithfulness hallucinations in RAG systems.

Architectural description
1. Data loader
      ⬇
2. RAG Pipeline (Indexer, Retriever, Generator)
      ⬇
3. Detector
      ⬇
4. Evaluator
____________________________________________________________________

DATA LOADER
As name suggests this component reads the RAGTruth* files (source files, 
question, llm answer, labels) and outputs a structured version of input data

RAG PIPELINE
1. Indexer:
   - Divides RAGTruth files given by the data loader into chunks
   - Transforms the chunks into vectors called embeddings
   - Stores the embeddings into a Vector Database

2. Retriever
   - Inputs a question
   - Determines the most relevant chunks from the vector database

3. Generator

