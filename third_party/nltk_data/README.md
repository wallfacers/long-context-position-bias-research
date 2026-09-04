# Frozen NLTK data for IFEval

`tokenizers/punkt_tab/english` was obtained with NLTK 3.9.1's official
downloader (`nltk.download("punkt_tab")`) on 2026-08-28. IFEval's official
sentence-count and capital-word verifiers require this resource. The four
files used by the English tokenizer are frozen here and checked against
`punkt-tab-english.sha256` before scoring. They are upstream NLTK data, not
project-authored training examples.

Upstream package URL:
`https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip`
