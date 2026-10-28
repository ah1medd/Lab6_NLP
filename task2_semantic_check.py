from task2_chunks import make_chunks, make_demo_docs
from lab6_search import Retriever

docs = make_demo_docs()
queries = [
    "I was sick and need more time to hand in work",
    "I cannot get into my account because I forgot the password",
    "My delivery arrived broken, how do I report it?"
]

records = make_chunks(docs, "sentence")
for query in queries:
    print("\nQuestion:", query)
    for backend in ["tfidf", "embedding"]:
        try:
            engine = Retriever(records, backend)
            print("Backend:", backend)
            hits = engine.search(query, 3)
            if not hits:
                print("No matching records.")
            for record, score in hits:
                print(record["id"], round(score, 3), record["text"])
        except (ImportError, OSError) as error:
            print("Could not run", backend, ":", error)
