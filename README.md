# ML4NLP1 — Machine Learning for Natural Language Processing 1

Course exercises for the ML4NLP1 course at the University of Zurich (UZH), Department of Computational Linguistics, Fall 2025. Each exercise is submitted as a self-contained, executed Jupyter notebook with an embedded lab report.

---

## Repository Structure

```
.
├── Assignment 1/
│   ├── ex1_lr.ipynb               # Language ID with Logistic Regression (sklearn)
│   └── ex1_nn.ipynb               # Language ID with MLP (skorch / PyTorch)
├── Assignment 2/
│   └── ex2_embeddings.ipynb       # CBOW word embeddings (PyTorch) + GloVe comparison
├── Assignment 4/
│   ├── ex4_ner_bert.ipynb         # NER with fine-tuned BERT (HuggingFace)
│   └── ex4_ner_gliner.ipynb       # Zero-shot NER with GLiNER
├── Assignment 5/
│   ├── ex5_prompting_part1.ipynb  # LLM symbolic reasoning & prompt engineering
│   └── ex5_prompting_part2.ipynb  # Instruction-tuned LLMs as linguistic annotators
├── Assignment 6/
│   └── ex6_topic_modeling.ipynb   # Topic modeling with LDA and CTM
└── Assignment AB/
    ├── Popkova_Assignment_AB.pdf  # Paper dissection: GloVe
    └── Popkova_Assignment_AB.tex  # LaTeX source for the paper dissection
```

---

## Exercises

### Exercise 1 — Language Identification (`ex1_lr.ipynb`, `ex1_nn.ipynb`)

**Task:** Multiclass language classification over 20 languages using text snippets.

**Part 1 — Logistic Regression (`ex1_lr.ipynb`):**
- Data exploration and custom train/test splitting across 20 languages (including English, German, Dutch, Danish, Swedish, Norwegian, Japanese, and 13 others).
- Feature engineering with TF-IDF character n-gram vectorization combined with a `text_length` feature.
- Hyperparameter tuning via `GridSearchCV` (penalty, solver, vectorizer parameters).
- Best model: no penalty, `newton-cg` solver — achieving **>0.98** accuracy, precision, recall, and F1 on the test set.
- Error analysis via confusion matrix: Latin-script language pairs (e.g. Norwegian/Danish) account for most misclassifications; languages with distinct scripts (e.g. Thai, Japanese) are near-perfectly classified.
- Ablation study: performance degrades as training text length is reduced, as shorter texts contain less distinctive linguistic signal.
- Feature importance analysis: TF-IDF features dominate, but `text_length` proves informative for Japanese.

**Part 2 — MLP with skorch (`ex1_nn.ipynb`):**
- Simple feedforward neural network built in PyTorch and trained with the `skorch` wrapper on Google Colab (GPU).
- Same 20-language dataset and feature representation as Part 1.
- Hyperparameter experiments over layer sizes, activation functions, regularizers, and early stopping.
- Best model achieves **94.5% accuracy**, exceeding the 87% benchmark.

---

### Exercise 2 — Word Embeddings (`ex2_embeddings.ipynb`)

**Task:** Train CBOW word embeddings from scratch on two domain-specific corpora and compare them against pretrained GloVe vectors.

**Models trained:**
- `CBOW2` on TripAdvisor hotel reviews (context window ±2).
- `CBOW5` on TripAdvisor hotel reviews (context window ±5).
- `CBOW2` on Sci-Fi stories (context window ±2, 3 epochs).

**Preprocessing:** lowercasing, whitespace stripping, punctuation removal, stop word handling.

**Key findings:**
- Hotel review embeddings are more coherent and interpretable than Sci-Fi embeddings (due to domain consistency and more training).
- `CBOW2` captures local syntactic relationships; `CBOW5` captures broader semantic associations.
- GloVe neighbours are semantically broader than domain-specific CBOW results, reflecting its much larger training corpus (Wikipedia, Common Crawl).

---

### Exercise 4 — Named Entity Recognition (`ex4_ner_bert.ipynb`, `ex4_ner_gliner.ipynb`)

**Task:** Sequence labeling in the IOB format on the wikiann dataset.

**Part 1 — Fine-tuned BERT (`ex4_ner_bert.ipynb`):**
- Chosen language: **German** (`de`).
- Train sets of 1,000 and 3,000 sentences; evaluation set of 2,000 sentences.
- Four model variants: fine-tuned (1k / 3k) and frozen backbone (1k / 3k).
- Model: `BertForTokenClassification` with a German BERT base checkpoint (HuggingFace).
- Evaluated with micro and macro F1 scores.
- Training run on Google Colab / Kaggle with GPU.

**Part 2 — Zero-shot NER with GLiNER (`ex4_ner_gliner.ipynb`):**
- Model: `urchade/gliner_multi_pii-v1`, a GLiNER model specialized in personally identifiable information (PII).
- Custom dataset of 10+ data points designed so entity labels must be inferred from context rather than stated explicitly.
- Diverse labels and entity types across data points.
- Inference on CPU (no GPU required).

---

### Exercise 5 — LLM Prompting (`ex5_prompting_part1.ipynb`, `ex5_prompting_part2.ipynb`)

**Task:** Explore prompting strategies for base and instruction-tuned LLMs.

**Part 1 — Symbolic Reasoning & Prompt Engineering (`ex5_prompting_part1.ipynb`):**
- Library: Unsloth (fast LLM inference and finetuning).
- Task: symbolic reasoning benchmark following Wei et al. (2022).
- Experiments with zero-shot, few-shot, and chain-of-thought (CoT) prompting.
- Parameter-efficient supervised finetuning (SFT) on the downstream task.
- Inference pipeline built for processing queries at scale.

**Part 2 — LLMs as Linguistic Annotators (`ex5_prompting_part2.ipynb`):**
- Model: instruction-tuned LLM (comparable to ChatGPT / Claude-style models).
- Task: part-of-speech (POS) tagging with tokenization.
- System prompt manipulation to direct the model's annotation behaviour.
- Evaluation using Levenshtein distance and structured output validation.

---

### Exercise 6 — Topic Modeling (`ex6_topic_modeling.ipynb`)

**Task:** Unsupervised topic discovery over three time periods of computer science publication titles from the DBLP database (before 1990, 1990–2009, 2010 onwards).

**Part 1 — LDA:**
- Preprocessing experiments (tokenization, stopword removal, stemming).
- Multiple topic count experiments (starting at 5+ topics per period).
- Topic naming based on top words; incoherent topics marked explicitly.
- Temporal trend analysis across the three periods.

**Part 2 — Contextualized Topic Models (CTM):**
- Model: `sentence-transformers/paraphrase-mpnet-base-v2` as the sentence encoder backbone.
- Same number of topics as Part 1 for direct comparison.
- Coherence comparison between LDA and CTM topics.
- Discussion of temporal trends and inter-model agreement.
- Suggestion and analysis of an alternative sentence encoder.

---

### Assignment AB — Paper Dissection (`Popkova_Assignment_AB.pdf`)

A structured two-page dissection of the GloVe paper (Pennington et al., 2014) covering: problem statement, ML methods (log-bilinear regression, weighted least-squares objective, co-occurrence matrix factorization), main innovations, key takeaways, and limitations (static embeddings, OOV words, insensitivity to polysemy).

---

## Dependencies

All notebooks are designed to run on **Google Colab** or **Kaggle** with GPU. Key libraries used across exercises:

```
torch
sklearn
skorch
transformers
datasets
gliner
gensim
chaospy
contextualized_topic_models
pyldavis
unsloth
```

---

## Notes

- All notebooks contain executed outputs and are fully re-runnable.
- Lab reports are written in markdown cells at the end of each notebook.
- Each notebook includes a **Declaration of Usage of Generative AI** section describing how AI tools were used for writing assistance and learning support (not for generating core ideas or code).
- Do not include data files when submitting — datasets are loaded directly from hosted URLs inside each notebook.
