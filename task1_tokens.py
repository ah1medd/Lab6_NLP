import re

samples = [
    "AI, search!",
    "I can't log in.",
    "help@example.com",
    "CS4944 version 2.1",
    "",
    "یہ ایک مثال ہے",
    "Wait... really?",
    "Reset my password"
]

for text in samples:
    whitespace = text.split()
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    print("Original:", repr(text))
    print("Whitespace:", whitespace, len(whitespace))
    print("Word-like:", words, len(words))
    print()

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for text in samples[0:2]:
        print("Model tokens:", repr(text), model.tokenizer.tokenize(text))
except ImportError:
    print("Sentence Transformers is not installed; model token comparison skipped.")
except Exception as error:
    print("Model tokenizer was not available:", error)


FAQ = [
    {"question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    {"question": "How do I report a damaged parcel?",
     "answer": "Contact support within 48 hours and attach photographs.",
     "source": "returns_guide.txt"},
    {"question": "What should I do if printer pages are blank?",
     "answer": "Check ink level, run the cleaning utility, then print a test page.",
     "source": "printer_guide.txt"}
]


def answer_question(query):
    if not query.strip():
        return "Please enter a question."

    for item in FAQ:
        if query.strip().casefold() == item["question"].casefold():
            return item["answer"] + " [source: " + item["source"] + "]"

    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [item["question"] + " " + item["answer"] for item in FAQ]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    if query_vector.nnz == 0:
        return "No matching FAQ. Please contact the help desk."

    scores = (query_vector @ matrix.T).toarray()[0]
    ranked = sorted(range(len(FAQ)), key=lambda i: (-scores[i], i))
    if scores[ranked[0]] <= 0:
        return "No matching FAQ. Please contact the help desk."

    lines = ["Possible matches; choose or rephrase your question:"]
    for i in ranked[:3]:
        if scores[i] > 0:
            lines.append(FAQ[i]["question"] + " (" + str(round(scores[i], 3)) + ")")
    return "\n".join(lines)


if __name__ == "__main__":
    print("FAQ examples")
    print(answer_question("How do I reset my password?"))
    print()
    print(answer_question("password recovery"))
    print()
    print(answer_question(""))
    print()
    print(answer_question("volcano geology"))
