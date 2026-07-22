import os
import time
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from rouge_score import rouge_scorer as rs
from run_output import create_run_dir

run_dir = create_run_dir("C5_query_expansion")

df = pd.read_csv("financebench_sample.csv")
missing = ['ADOBE_2015_10K','ADOBE_2016_10K','ADOBE_2017_10K','ADOBE_2022_10K',
           'JOHNSON_JOHNSON_2022_10K','JOHNSON_JOHNSON_2022Q4_EARNINGS',
           'JOHNSON_JOHNSON_2023_8K_dated-2023-08-30','JOHNSON_JOHNSON_2023Q2_EARNINGS',
           'MGMRESORTS_2022Q4_EARNINGS']
df = df[~df['doc_name'].isin(missing)].reset_index(drop=True)
print(f"Running C5 on {len(df)} questions")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

hyde_prompt = """You are a financial analyst. Write a short passage that would appear 
in a 10-K, 10-Q, or earnings transcript that answers this question. 
Write only the passage text, as if excerpted from the actual filing:

Question: {question}

Passage:"""

def hyde_retrieve(question, vectorstore, k=10):
    # Generate hypothetical answer passage
    response = llm.invoke(hyde_prompt.format(question=question))
    hypothetical = response.content
    # Retrieve using hypothetical passage embedding
    hyde_docs = vectorstore.similarity_search(hypothetical, k=k)
    # Also retrieve using original question
    orig_docs = vectorstore.similarity_search(question, k=k)
    # Merge and deduplicate
    seen = set()
    merged = []
    for doc in hyde_docs + orig_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            merged.append(doc)
    return merged[:k]

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

results = []
for i, row in df.iterrows():
    print(f"\n[{i+1}/{len(df)}] {row['company']} — {row['question'][:60]}...")
    try:
        start = time.time()
        vectorstore = get_index(row['doc_name'])
        docs = hyde_retrieve(row['question'], vectorstore, k=10)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer concisely:"
        response = llm.invoke(prompt)
        predicted = response.content
        latency = time.time() - start
        score = scorer.score(str(row['answer']), predicted)
        f1 = score['rougeL'].fmeasure
        em = 1.0 if str(row['answer']).strip().lower() in predicted.strip().lower() else 0.0
        results.append({
            "condition": "C5_query_expansion",
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
            "condition": "C5_query_expansion", "question_num": i+1,
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
print("C5 QUERY EXPANSION RESULTS")
print("="*50)
print(f"Total questions: {len(results_df)}")
print(f"Mean ROUGE-L F1: {results_df['rouge_f1'].mean():.3f}")
print(f"Mean Exact Match: {results_df['exact_match'].mean():.3f}")
print(f"Mean Latency: {results_df['latency_sec'].mean():.1f}s")
print(f"\nBy question type:")
print(results_df.groupby('question_type')['rouge_f1'].mean().round(3))
print(f"\nResults saved to {run_dir / 'results.csv'}")
