import os
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer as rs

# Load data
df = pd.read_csv("financebench_sample.csv")

# Skip missing PDFs
missing = ['ADOBE_2015_10K','ADOBE_2016_10K','ADOBE_2017_10K','ADOBE_2022_10K',
           'JOHNSON_JOHNSON_2022_10K','JOHNSON_JOHNSON_2022Q4_EARNINGS',
           'JOHNSON_JOHNSON_2023_8K_dated-2023-08-30','JOHNSON_JOHNSON_2023Q2_EARNINGS',
           'MGMRESORTS_2022Q4_EARNINGS']
df = df[~df['doc_name'].isin(missing)].reset_index(drop=True)
print(f"Running on {len(df)} questions")

# Setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# Cache indexes per document
indexes = {}

def get_index(doc_name):
    if doc_name not in indexes:
        pdf_path = f"data/pdfs/{doc_name}.pdf"
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        indexes[doc_name] = FAISS.from_documents(chunks, embeddings)
        print(f"  Indexed: {doc_name} ({len(chunks)} chunks)")
    return indexes[doc_name]

# Run evaluation
results = []
for i, row in df.iterrows():
    print(f"\n[{i+1}/{len(df)}] {row['company']} — {row['question'][:60]}...")
    try:
        start = time.time()

        # Retrieve
        vectorstore = get_index(row['doc_name'])
        docs = vectorstore.similarity_search(row['question'], k=10)
        context = "\n\n".join([d.page_content for d in docs])

        # Generate
        prompt = f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer concisely:"
        response = llm.invoke(prompt)
        predicted = response.content
        latency = time.time() - start

        # Score
        score = scorer.score(str(row['answer']), predicted)
        f1 = score['rougeL'].fmeasure
        em = 1.0 if str(row['answer']).strip().lower() in predicted.strip().lower() else 0.0

        results.append({
            "condition": "C0_baseline",
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
            "condition": "C0_baseline",
            "question_num": i+1,
            "company": row['company'],
            "doc_name": row['doc_name'],
            "question_type": row['question_type'],
            "question": row['question'],
            "gold_answer": row['answer'],
            "predicted_answer": f"ERROR: {e}",
            "rouge_f1": 0.0,
            "exact_match": 0.0,
            "latency_sec": 0.0,
        })

    # Save progress every 10 questions
    if (i + 1) % 10 == 0:
        pd.DataFrame(results).to_csv("results/c0_baseline_progress.csv", index=False)
        print(f"\n  Progress saved — {i+1}/{len(df)} done")

# Save final results
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("results/c0_baseline.csv", index=False)

# Print summary
print("\n" + "="*50)
print("C0 BASELINE RESULTS")
print("="*50)
print(f"Total questions: {len(results_df)}")
print(f"Mean ROUGE-L F1: {results_df['rouge_f1'].mean():.3f}")
print(f"Mean Exact Match: {results_df['exact_match'].mean():.3f}")
print(f"Mean Latency: {results_df['latency_sec'].mean():.1f}s")
print(f"\nBy question type:")
print(results_df.groupby('question_type')['rouge_f1'].mean().round(3))
print(f"\nResults saved to results/c0_baseline.csv")