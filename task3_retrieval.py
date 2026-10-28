import csv
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from task2_chunks import make_chunks, make_demo_docs


docs = make_demo_docs()
docs.update({
    "contractor": (
        "Contractors may work remotely only when the signed contract explicitly permits remote "
        "work. The employee remote-work rule does not automatically apply to contractors. A manager "
        "cannot override a contract term without an approved written amendment. Contractors should "
        "check the current signed agreement and ask the contract owner if the wording is unclear. "
        "Employees may request remote work with manager approval, but that separate rule applies to "
        "employees rather than contractors. A remote-work request should include the expected work "
        "location and the dates requested. Equipment and data protection rules still apply when work "
        "is performed away from the office. An informal conversation is not a substitute for written "
        "permission. If the contract is silent, the contractor should obtain clarification before "
        "working remotely. The organization may update procedures, but an update does not change a "
        "signed contract unless the required process is followed."
    ),
    "printer": (
        "For blank printer pages, first check the ink level and confirm that the correct paper is loaded. "
        "Run the printer cleaning utility and print a test page afterwards. If the test page is still blank, "
        "check the device manual before opening any cover. Do not touch hot internal parts while the device "
        "is operating. A scanner producing a blank image has a different procedure: check document placement "
        "and select the correct scanning source. These scanner instructions should not be used as a printer "
        "repair procedure. Record the printer model and any displayed error code before contacting support. "
        "If the printer is shared, tell other users before taking it offline. Replace consumables only with "
        "the type specified for the model. Repeated cleaning cycles may use ink, so follow the device manual. "
        "Print a final test page after any adjustment and keep the result for the support request."
    )
})

queries = [
    {"query": "What proof is required if illness delays my assignment?", "doc": "handbook",
     "evidence": "if medical evidence is sent to the course coordinator before the deadline", "split": "dev"},
    {"query": "Where should coursework be uploaded?", "doc": "handbook",
     "evidence": "uploaded through the learning portal before the deadline", "split": "dev"},
    {"query": "How can I get back into an account after forgetting my password?", "doc": "it_help",
     "evidence": "A forgotten password can be reset from the recovery link on the login page", "split": "dev"},
    {"query": "What is the deadline for reporting a broken delivery?", "doc": "returns",
     "evidence": "contact customer support within 48 hours of delivery", "split": "dev"},
    {"query": "What must a contractor have before working remotely?", "doc": "contractor",
     "evidence": "only when the signed contract explicitly permits remote work", "split": "test"},
    {"query": "What should I try when the printer makes blank pages?", "doc": "printer",
     "evidence": "check the ink level and confirm that the correct paper is loaded", "split": "test"},
    {"query": "Can I return an unused item if I changed my mind?", "doc": "returns",
     "evidence": "unused items may be returned within 14 days", "split": "test"},
    {"query": "What should I do after a suspicious sign-in alert?", "doc": "it_help",
     "evidence": "change the password and report the notice to IT", "split": "test"},
    {"query": "What are tomorrow's flight prices?", "doc": None, "evidence": None, "split": "test"},
    {"query": "Who won the local cricket match?", "doc": None, "evidence": None, "split": "test"}
]


def evaluate(scores, chunks, rows):
    hits = []
    reciprocal = []
    for row, query in zip(scores, rows):
        if query["doc"] is None:
            continue
        top = np.argsort(-row, kind="stable")[:3]
        good = []
        for rank, index in enumerate(top, 1):
            chunk = chunks[index]
            if chunk["doc"] == query["doc"] and query["evidence"] in chunk["text"]:
                good.append(rank)
        hits.append(bool(good))
        reciprocal.append(1 / good[0] if good else 0)
    if not hits:
        return None, None
    return sum(hits) / len(hits), sum(reciprocal) / len(reciprocal)


def main():
    logs = []
    try:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        model_load = time.perf_counter() - start
        print("Embedding model load seconds:", round(model_load, 3))
    except (ImportError, OSError) as error:
        model = None
        print("Embedding model unavailable:", error)

    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        texts = [chunk["text"] for chunk in chunks]
        print("\nStrategy:", strategy, "chunks:", len(chunks))

        for method in ["tfidf", "embedding"]:
            if method == "embedding" and model is None:
                continue

            start = time.perf_counter()
            if method == "tfidf":
                vectorizer = TfidfVectorizer()
                C = vectorizer.fit_transform(texts)
                Q = vectorizer.transform([q["query"] for q in queries])
            else:
                lengths = [len(model.tokenizer(t, truncation=False)["input_ids"])
                           for t in texts + [q["query"] for q in queries]]
                print("Maximum model input tokens:", max(lengths),
                      "limit:", model.max_seq_length)
                if max(lengths) > model.max_seq_length:
                    raise ValueError("Text exceeds model input limit")
                C = model.encode(texts, normalize_embeddings=True)
                Q = model.encode([q["query"] for q in queries],
                                 normalize_embeddings=True)
            encode_time = time.perf_counter() - start

            start = time.perf_counter()
            scores = (Q @ C.T).toarray() if method == "tfidf" else Q @ C.T
            search_time = time.perf_counter() - start

            for split in ["dev", "test"]:
                selected = [i for i, q in enumerate(queries) if q["split"] == split]
                hit, mrr = evaluate(scores[selected], chunks,
                                    [queries[i] for i in selected])
                print(method, split, "Hit@3:", round(hit, 3),
                      "MRR@3:", round(mrr, 3),
                      "encode/index seconds:", round(encode_time, 4),
                      "search seconds:", round(search_time, 4))

            for row, query in zip(scores, queries):
                top = np.argsort(-row, kind="stable")[:3]
                for rank, index in enumerate(top, 1):
                    chunk = chunks[index]
                    entry = {
                        "strategy": strategy, "method": method,
                        "query": query["query"], "split": query["split"],
                        "answerable": query["doc"] is not None, "rank": rank,
                        "score": round(float(row[index]), 6),
                        "chunk_id": chunk["id"], "doc": chunk["doc"],
                        "start": chunk["start"], "end": chunk["end"],
                        "text": chunk["text"]
                    }
                    logs.append(entry)
                    print(rank, entry["chunk_id"], entry["score"],
                          entry["start"], entry["end"], entry["text"])

    if logs:
        with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(logs[0]))
            writer.writeheader()
            writer.writerows(logs)
        print("\nSaved retrieval_log.csv")


if __name__ == "__main__":
    main()
