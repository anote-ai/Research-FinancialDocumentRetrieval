import os
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from rouge_score import rouge_scorer as rs
from run_output import create_run_dir

run_dir = create_run_dir("C6_hybrid")

df = pd.read_csv("financebench_sample.csv")
missing = ['ADOBE_2015_10K','ADOBE_2016_10K','ADOBE_2017_10K','ADOBE_2022_10K',
           'JOHNSON_JOHNSON_2022_10K','JOHNSON_JOHNSON_2022Q4_EARNINGS',
           'JOHNSON_JOHNSON_2023_8K_dated-2023-08-30','JOHNSON_JOHNSON_2023Q2_EARNINGS',
           'MGMRESORTS_2022Q4_EARNINGS']
df = df[~df['doc_name'].isin(missing)].reset_index(drop=True)
print(f"Running C6 on {len(df)} questions")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)

# C6 uses recursive chunking (like C2)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
)

# Reranker (like C3)
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=reranker_model, top_n=5)

def parse_doc_metadata(doc_name):
    parts = doc_name.split("_")
    company = parts[0]
    year = next((p for p in parts if p.isdigit() and len(p) == 4), "unknown")
    if "10K" in doc_name: doc_type = "10-K"
    elif "10Q" in doc_name: doc_type = "10-Q"
    elif "EARNINGS" in doc_name: doc_type = "earnings"
    elif "8K" in doc_name: doc_type = "8-K"
    else: doc_type = "unknown"
    return company, year, doc_type

indexes = {}
chunk_cache = {}

def get_retriever(doc_name):
    if doc_name not in indexes:
        pdf_path = f"data/pdfs/{doc_name}.pdf"
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)

        # Add metadata (like C4)
        company, year, doc_type = parse_doc_metadata(doc_name)
        for chunk in chunks:
            chunk.metadata.update({"company": company, "fiscal_year": year, "doc_type": doc_type})
            prefix = f"[Company: {company} | Year: {year} | Type: {doc_type}]\n"
            chunk.page_content = prefix + chunk.page_content

        chunk_cache[doc_name] = chunks
        vectorstore = FAISS.from_documents(chunks, embeddings)
        indexes[doc_name] = vectorstore
        print(f"  Indexed: {doc_name} ({len(chunks)} chunks)")

    chunks = chunk_cache[doc_name]
    vectorstore = indexes[doc_name]

    # BM25 sparse retriever
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20

    # Dense retriever
    dense = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Ensemble: BM25 + dense
    ensemble = EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[0.5, 0.5]
    )

    # Rerank the merged results
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble
    )

results = []
for i, row in df.iterrows():
    print(f"\n[{i+1}/{len(df)}] {row['company']} — {row['question'][:60]}...")
    try:
        start = time.time()
        retriever = get_retriever(row['doc_name'])
        company, year, doc_type = parse_doc_metadata(row['doc_name'])
        enriched_query = f"[Company: {company} | Year: {year} | Type: {doc_type}]\n{row['question']}"
        docs = retriever.invoke(enriched_query)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer concisely:"
        response = llm.invoke(prompt)
        predicted = response.content
        latency = time.time() - start
        score = scorer.score(str(row['answer']), predicted)
        f1 = score['rougeL'].fmeasure
        em = 1.0 if str(row['answer']).strip().lower() in predicted.strip().lower() else 0.0
        results.append({
            "condition": "C6_hybrid",
            "question_num": i+1,
            "company": row['company'],
            "doc_name": row['doc_name'],
            "question_type": row['question_type'],
            "question": row['question'],
            "gold_answer": row['answer'],
            "predicted_answer": predicted,
            "rouge_f1": round(f1, 4),
            "exact_match": em,
            "latency_sec": round(latency, 2),
        })
        print(f"  Gold: {row['answer']}")
        print(f"  Pred: {predicted[:80]}...")
        print(f"  F1: {f1:.3f} | EM: {em} | {latency:.1f}s")
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "condition": "C6_hybrid", "question_num": i+1,
            "company": row['company'], "doc_name": row['doc_name'],
            "question_type": row['question_type'], "question": row['question'],
            "gold_answer": row['answer'], "predicted_answer": f"ERROR: {e}",
            "rouge_f1": 0.0, "exact_match": 0.0, "latency_sec": 0.0,
        })
    if (i + 1) % 10 == 0:
        pd.DataFrame(results).to_csv(run_dir / "progress.csv", index=False)
        print(f"\n  Progress saved — {i+1}/{len(df)} done")

results_df = pd.DataFrame(results)
results_df.to_csv(run_dir / "results.csv", index=False)

print("\n" + "="*50)
print("C6 HYBRID RESULTS")
print("="*50)
print(f"Total questions: {len(results_df)}")
print(f"Mean ROUGE-L F1: {results_df['rouge_f1'].mean():.3f}")
print(f"Mean Exact Match: {results_df['exact_match'].mean():.3f}")
print(f"Mean Latency: {results_df['latency_sec'].mean():.1f}s")
print(f"\nBy question type:")
print(results_df.groupby('question_type')['rouge_f1'].mean().round(3))
print(f"\nResults saved to {run_dir / 'results.csv'}")
