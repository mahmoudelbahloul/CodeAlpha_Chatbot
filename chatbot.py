"""FAQ chatbot: TF-IDF + cosine similarity, written from scratch (standard library only).

    python chatbot.py                         # interactive chat
    python chatbot.py "how much does it cost" # one-shot answer

The knowledge base is faqs.json (8 sample FAQs). A question is answered with the
closest FAQ if cosine similarity >= THRESHOLD, otherwise with a fallback message.
"""
import json, math, os, re, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.2
FALLBACK = "I'm not sure about that! Contact us at info@opalmarketing.com"
STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or", "for", "with",
    "do", "you", "can", "me", "my", "your", "we", "i", "how", "what", "who", "when",
    "where", "does", "are", "was", "were", "be", "been", "have", "has", "this", "that",
    "will", "would", "could", "should", "about", "there", "get", "any", "all", "just",
}


def preprocess(text):
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def compute_tf(tokens):
    total = len(tokens) or 1
    return {w: c / total for w, c in Counter(tokens).items()}


def compute_idf(corpus):
    n, df = len(corpus), Counter()
    for doc in corpus:
        df.update(set(doc))
    return {w: math.log((n + 1) / (d + 1)) + 1 for w, d in df.items()}


def cosine_similarity(a, b):
    dot = sum(a[w] * b[w] for w in set(a) & set(b))
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class FAQBot:
    def __init__(self, faqs):
        self.faqs = faqs
        corpus = [preprocess(f["question"] + " " + f["answer"]) for f in faqs]
        self.idf = compute_idf(corpus)
        self.vecs = [self._vector(t) for t in corpus]

    def _vector(self, tokens):
        tf = compute_tf(tokens)
        return {w: tf[w] * self.idf.get(w, 1) for w in tf}

    def reply(self, text):
        """Return (answer, score)."""
        tokens = preprocess(text)
        if not tokens:
            return "Please ask me something about OPAL Marketing!", 0.0
        q = self._vector(tokens)
        scores = [cosine_similarity(q, v) for v in self.vecs]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < THRESHOLD:
            return FALLBACK, scores[best]
        return self.faqs[best]["answer"], scores[best]


def main():
    bot = FAQBot(json.load(open(os.path.join(HERE, "faqs.json"), encoding="utf-8")))
    if len(sys.argv) > 1:
        answer, score = bot.reply(" ".join(sys.argv[1:]))
        print(f"{answer}\n(confidence: {score:.1%})")
        return
    print("OPAL Marketing FAQ bot. Type 'quit' to exit.")
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if q.lower() == "quit":
            break
        if q:
            answer, score = bot.reply(q)
            print(f"Bot: {answer}\n     (confidence: {score:.1%})\n")


if __name__ == "__main__":
    main()
