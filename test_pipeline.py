import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer

# Load 5 questions from FinanceBench
df = pd.read_csv("financebench_sample.csv")
sample = df.head(5)
print(f"Testing on {len(sample)} questions")
print(sample[['company', 'doc_name', 'question']].to_string())

# Load one PDF
doc_name = sample.iloc[0]['doc_name']
pdf_path = f"data/pdfs/{doc_name}.pdf"

if not os.path.exists(pdf_path):
    print(f"PDF not found: {pdf_path}")
    print("Available PDFs:")
    print(os.listdir("data/pdfs")[:5])
else:
    print(f"\nLoading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    # Chunk it
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    # Embed and index
    print("Embedding chunks (this takes a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Index built successfully")

    # Test retrieval
    question = sample.iloc[0]['question']
    gold_answer = sample.iloc[0]['answer']
    print(f"\nQuestion: {question}")
    print(f"Gold answer: {gold_answer}")

    docs = vectorstore.similarity_search(question, k=10)
    print(f"\nTop retrieved chunk:")
    print(docs[0].page_content[:300])

    # Test generation
    print("\nGenerating answer with GPT-4o...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
    response = llm.invoke(prompt)
    predicted = response.content
    print(f"Predicted: {predicted}")

    # Score it
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(gold_answer, predicted)
    print(f"\nROUGE-L F1: {score['rougeL'].fmeasure:.3f}")
    print("\nPipeline test complete!")