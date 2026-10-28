Lab 6 NLP files

Python 3.10 or newer is recommended.

Setup:
1. Create and activate a virtual environment.
2. Run: python -m pip install -r requirements.txt

Run the tasks:
- python task1_tokens.py
- python task2_chunks.py
- python task2_semantic_check.py
- python task3_retrieval.py
- python document_chatbot.py

Task 1 compares whitespace and regex-based word-like tokens, optionally shows model subword tokens, and demonstrates exact FAQ answers, suggestions, empty input and unsupported input.

Task 2 contains three synthetic documents and chunking functions for 40-word windows with overlap 0 or 10 and sentence-aware packing. It checks source offsets and boundary cases. The semantic check compares TF-IDF with a pretrained sentence encoder.

Task 3 expands the corpus to five documents and evaluates three chunking strategies with TF-IDF and, when available, sentence embeddings. It writes retrieval_log.csv. The queries are divided into development and test groups. Unanswerable queries are logged but are not counted in Hit@3 or MRR@3.

The embedding model downloads the first time it is used. If it cannot be downloaded, task1 still runs its tokenizer comparison only if the library/model is available; Task 3 continues with TF-IDF if the model import or loading fails.

All policies and records are synthetic examples for this lab, not real university, workplace or customer policies.
