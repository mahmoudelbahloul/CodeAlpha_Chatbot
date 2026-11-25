# FAQ Chatbot (TF-IDF + cosine similarity)

CodeAlpha internship, Task 2 (Chatbot). A retrieval chatbot that answers a question with the closest FAQ entry.
The TF-IDF vectoriser and the cosine similarity are written from scratch with the Python standard library
(`re`, `math`, `collections`), no scikit-learn or NLP packages.

## How it works

1. **Preprocess:** lowercase, strip punctuation, split into words, remove stopwords and one-letter tokens.
2. **Vectorise:** each FAQ (question + answer) becomes a TF-IDF vector; the user's question is vectorised with the same IDF.
3. **Match:** cosine similarity between the question and every FAQ.
4. **Answer:** return the best FAQ answer if its score is at least 0.2, otherwise a fallback message with a contact address.

The knowledge base is `faqs.json`: 8 **sample** FAQs for a digital marketing agency ("OPAL Marketing"; services, pricing,
consultations, onboarding, social media, branding, reporting). The company details are sample data.

## Run it

```bash
python chatbot.py                          # interactive chat, type 'quit' to exit
python chatbot.py "how much does it cost"  # one-shot
```

```
$ python chatbot.py "Do you manage social media?"
Yes! We handle content planning, post creation, scheduling, community management, and ad management on Facebook, Instagram, TikTok, and LinkedIn.
(confidence: 35.4%)
```

`chatbot.ipynb` is the step-by-step version with the outputs of each stage (preprocessing, TF-IDF, similarity scores, test questions).

## Limitations (it is a keyword matcher)

- It matches **shared words**, not meaning. "How much does it cost?" works (0.33), but "what is the price" scores 0.0
  and falls back, because "price" never appears in the FAQ text.
- No stemming, spelling correction or synonyms, English only, and only 8 FAQs. Scores are low in absolute terms
  (about 0.2-0.4 for correct matches), so the 0.2 threshold was set by hand, not tuned on data.
- No evaluation set. The notebook shows a handful of test questions, not an accuracy measurement.

Natural next steps: sentence embeddings instead of TF-IDF (handles synonyms), a labelled set of test questions to tune the
threshold, and more FAQs. The [agentic-rag-mcp-system](https://github.com/mahmoudelbahloul/agentic-rag-mcp-system) project
takes the embeddings route with an LLM on top.
