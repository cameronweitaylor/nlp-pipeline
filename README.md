# NLP Pipeline — Text Classification

Two NLP preprocessing and classification pipelines implemented from scratch in Python. Both classify text queries into categories using Complement Naive Bayes, but they differ in their preprocessing choices and feature representations — allowing comparison of different design decisions.

## Pipelines

### Pipeline 1 — Stopword removal + TF-IDF
- **Preprocessing:** lowercase → punctuation removal → whitespace normalisation → tokenisation → stopword removal → lemmatisation
- **Features:** bag-of-unigrams (binary) followed by a custom TF-IDF transformation
- **Classifier:** Complement Naive Bayes, tuned with `GridSearchCV`

### Pipeline 2 — POS filtering + spell correction + raw counts
- **Preprocessing:** punctuation removal → lowercase → hard-coded spelling correction → whitespace normalisation → tokenisation → POS filtering (keeps nouns, verbs, adjectives, particles) → lemmatisation
- **Features:** bag-of-unigrams (raw counts)
- **Classifier:** Complement Naive Bayes, tuned with `GridSearchCV`
- **Extras:** confusion matrix visualisation, misclassification analysis, and per-word/class co-occurrence inspection

Both pipelines use the same train/test split (`random_state=32`, 75/25) for comparability, and grid search over `alpha` ∈ [0.01, 1.00] and `norm` ∈ {True, False} with `f1_macro` scoring.

## Setup

```bash
git clone https://github.com/cameronweitaylor/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

Place the dataset `cw_data.csv` in the project root (two columns, no header — text in column 0, class label in column 1), then launch Jupyter:

```bash
jupyter notebook
```

Open and run either notebook top to bottom:
- `NLP_Pipeline_1.ipynb`
- `NLP_Pipeline_2_2.ipynb`

## Requirements

- Python 3.9+
- See `requirements.txt` for package dependencies
- spaCy English model: `en_core_web_sm`

## Project Structure

├── NLP_Pipeline_1.ipynb    # Pipeline 1: stopword removal + TF-IDF
├── NLP_Pipeline_2_2.ipynb  # Pipeline 2: POS filtering + spell correction
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md