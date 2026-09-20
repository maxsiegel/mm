# mm

A word-level Markov chain trained on the papers of [Wiktor Młynarski](https://scholar.google.com/citations?user=Fp9SbUwAAAAJ),
with a browser frontend: **https://maxsiegel.github.io/mm/**

The page ships a table of n-gram transition *counts*, not the corpus. No paper text
is redistributed here; `papers/`, `text/` and the corpora are gitignored and are
rebuilt locally from the sources listed below.

## Pipeline

| step | script | output |
|---|---|---|
| extract text from PDFs | `pdftotext` | `text/*.txt` |
| strip references, captions, citations, back matter | `clean_corpus.py` | `corpus.txt` |
| drop duplicate documents and reused paragraphs | `dedupe.py` | `corpus_dedup.txt` |
| train + sample from the CLI | `markov.py` | stdout |
| export counts for the web page | `build_model.py` | `docs/model.json` |

```bash
for f in papers/*.pdf; do pdftotext -q "$f" "text/$(basename "$f" .pdf).txt"; done
python3 clean_corpus.py && python3 dedupe.py
python3 markov.py --order 2 --sentences 5 --seed 1
python3 build_model.py && (cd docs && python3 -m http.server 8777)
```

## Sampling

Training is sentence-aware: each sentence is padded with `order` start tokens and one
end token. Sampling uses the full conditional distribution at the requested order and
backs off to shorter contexts for unseen states.

A sampled sentence end is honoured only when the last emitted word is in the stop set;
otherwise the end token is rejected — including during backoff — and generation
continues. `--stop-mode final` (default) licenses any word the corpus itself places
sentence-finally; `--stop-mode function` licenses only the corpus's function words,
which makes stopping points rare and sentences roughly four times longer.

## Corpus

18 documents, 178,647 words after deduplication, drawn from journal articles, arXiv
and bioRxiv preprints. The corpus is small relative to the state space — at order 2
there are ~2.8 continuations per state, and at order 3 only ~1.5, where the chain
mostly replays source sentences verbatim. Order 2 is the usable regime.

Copyright in the underlying papers rests with their authors and publishers; licenses
vary (CC-BY, CC-BY-NC, CC-BY-NC-ND, and one preprint marked all-rights-reserved).
Treat generated text accordingly — it can reproduce source spans verbatim.
