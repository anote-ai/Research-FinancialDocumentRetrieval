import pandas as pd
import requests
import os
import time

df = pd.read_csv('financebench_sample.csv')
os.makedirs('data/pdfs', exist_ok=True)

docs = df[['doc_name', 'doc_link']].drop_duplicates()
print(f"Downloading {len(docs)} PDFs...")

failed = []
for i, row in docs.iterrows():
    filename = f"data/pdfs/{row['doc_name']}.pdf"
    if os.path.exists(filename):
        print(f"Already exists: {row['doc_name']}")
        continue
    try:
        response = requests.get(row['doc_link'], timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {row['doc_name']}")
        else:
            print(f"Failed ({response.status_code}): {row['doc_name']}")
            failed.append(row['doc_name'])
    except Exception as e:
        print(f"Error: {row['doc_name']} — {e}")
        failed.append(row['doc_name'])
    time.sleep(0.5)

print(f"\nDone. Failed: {len(failed)}")
if failed:
    print("Failed docs:")
    for f in failed:
        print(f" - {f}")
