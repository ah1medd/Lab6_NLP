import re


def word_spans(text):
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def windows(text, size=40, overlap=10, offset=0):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Use a positive size and overlap smaller than size")

    spans = word_spans(text)
    result = []
    step = size - overlap

    for i in range(0, len(spans), step):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break

    return result


def sentence_chunks(text, size=40):
    if size <= 0:
        raise ValueError("Size must be positive")

    cuts = [0]
    cuts.extend(match.end() for match in re.finditer(r"[.!?]\s+", text))
    cuts.append(len(text))

    result = []
    pending = None
    count = 0

    for a, b in zip(cuts, cuts[1:]):
        segment = text[a:b]
        n = len(segment.split())
        if n == 0:
            continue

        if n > size:
            if pending is not None:
                result.append(pending)
                pending = None
                count = 0
            result.extend(windows(segment, size, 0, a))
        elif pending is not None and count + n <= size:
            pending = (pending[0], b)
            count += n
        else:
            if pending is not None:
                result.append(pending)
            pending = (a, b)
            count = n

    if pending is not None:
        result.append(pending)

    return result


def make_chunks(docs, strategy):
    result = []
    for doc_id, text in docs.items():
        if strategy == "sentence":
            spans = sentence_chunks(text, 40)
        elif strategy == "0":
            spans = windows(text, 40, 0)
        elif strategy == "10":
            spans = windows(text, 40, 10)
        else:
            raise ValueError("Strategy must be 0, 10 or sentence")

        for number, (start, end) in enumerate(spans):
            result.append({
                "id": doc_id + ":" + str(number),
                "doc": doc_id,
                "start": start,
                "end": end,
                "text": text[start:end]
            })
    return result


def make_demo_docs():
    docs = {
        "handbook": (
            "Assessment work should be uploaded through the learning portal before the deadline. "
            "Late submission is normally penalized. A student who is ill may request an extension "
            "if medical evidence is sent to the course coordinator before the deadline. The evidence "
            "must show the dates affected, but it does not need to include a detailed diagnosis. "
            "Students should keep the submission receipt until marks are released. The receipt is useful "
            "if the portal has a technical problem. Files should be saved as PDF unless the assignment "
            "page asks for another format. A late request without evidence may be refused. The coordinator "
            "reviews requests individually and sends the decision through the student email account. "
            "Students should not assume that asking for an extension automatically changes the deadline. "
            "If the portal is unavailable, take a screenshot and email the help desk before the deadline. "
            "The help desk records the time of the report and forwards it to the course team. "
            "A student who has already submitted a file can replace it only while the portal allows "
            "resubmission. The latest uploaded file is the one normally marked. Check the course page "
            "for the final date and any instructions about file naming."
        ),
        "it_help": (
            "A forgotten password can be reset from the recovery link on the login page. Enter the "
            "email address linked to the account and follow the message sent to that address. If the "
            "message does not arrive, check the junk folder and wait a few minutes before trying again. "
            "Never share a recovery code with another person or place it in a public message. If the "
            "account email is no longer available, contact the IT help desk and provide your student "
            "number through the approved support form. The help desk may ask identity questions before "
            "changing account details. Passwords should be unique and should not be reused on unrelated "
            "websites. Staff will never ask for the full password by email. For an account lockout, wait "
            "ten minutes and try once more before contacting support. A successful reset does not restore "
            "files deleted from a separate service. Keep recovery codes in a private place. When signing "
            "in on a shared computer, do not save the password in the browser and sign out when finished. "
            "If a suspicious login notice appears, change the password and report the notice to IT. "
            "Support requests should include the error message, but must not include a password or code."
        ),
        "returns": (
            "For a damaged parcel, contact customer support within 48 hours of delivery. Attach clear "
            "photographs of the item and its packaging so the support team can inspect the damage. Keep "
            "the delivery label until the report is resolved. A report made after 48 hours may still be "
            "reviewed, but approval is not guaranteed. A change of mind is handled differently: unused "
            "items may be returned within 14 days if the original packaging is available. The 14 day "
            "period does not replace the 48 hour reporting window for damaged goods. Do not send a parcel "
            "back before receiving a return reference number. Include the order number in the support form "
            "and remove personal payment details from attached images. The support team will reply with "
            "the next step after checking the photographs. Items damaged during delivery should not be "
            "described as a change of mind. Refund timing depends on the payment provider and is not "
            "confirmed by the initial report. If the package contains the wrong item, choose the incorrect "
            "item category instead of the damage category. Keep copies of the support message and any "
            "reference number for later follow-up."
        )
    }
    return docs


if __name__ == "__main__":
    docs = make_demo_docs()
    print("Chunk comparison")
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        print(strategy, "chunks:", len(chunks))
        for chunk in chunks[:2]:
            print(chunk["id"], chunk["start"], chunk["end"], len(chunk["text"].split()))

    assert windows("") == []
    assert windows("short text") == [(0, 10)]
    try:
        windows("one two", 2, 2)
    except ValueError:
        print("Invalid overlap check passed")
    else:
        raise AssertionError("Invalid overlap was accepted")

    long_sentence = " ".join(["word"] * 95) + "."
    assert all(len(long_sentence[a:b].split()) <= 40
               for a, b in sentence_chunks(long_sentence))
    for strategy in ["0", "10", "sentence"]:
        for chunk in make_chunks(docs, strategy):
            assert chunk["text"] == docs[chunk["doc"]][chunk["start"]:chunk["end"]]
    print("Boundary and source-offset checks passed")
