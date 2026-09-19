"""Deduplicate the paper corpus before language-model training.

Two passes:
  1. Document level - near-duplicate documents (a preprint and its published
     version) are detected by 5-gram containment (overlap over the smaller
     document, which is robust to the two versions differing in length); one is
     dropped, preferring the peer-reviewed version when both are present and
     otherwise the longer text.
  2. Paragraph level - across the surviving documents, a paragraph is dropped
     when most of its 5-grams have already been emitted. This catches partial
     reuse (review articles recycling methods prose, shared boilerplate).
"""
import re, glob, os, sys
from clean_corpus import clean

DOC_SIM   = 0.30   # 5-gram containment above this means "same study"
PARA_SIM  = 0.60   # fraction of already-seen 5-grams that makes a paragraph redundant
MIN_PARA  = 25     # words; shorter fragments are exempt from paragraph dedup

# Filenames of preprints whose peer-reviewed version is also in the corpus get
# dropped first; the published text is the one the author stood behind.
PREPRINT_MARKERS = ('_arxiv', '_biorxiv')

def norm(text):
    return re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()

def grams(tokens, n=5):
    return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def is_preprint(name):
    return any(m in name for m in PREPRINT_MARKERS)

def main():
    docs = {}
    for f in sorted(glob.glob('text/*.txt')):
        text = clean(f)
        docs[os.path.basename(f)[:-4]] = (text, grams(norm(text)))

    names = list(docs)
    dropped = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if a in dropped or b in dropped:
                continue
            A, B = docs[a][1], docs[b][1]
            s = len(A & B) / max(1, min(len(A), len(B)))
            if s < DOC_SIM:
                continue
            # prefer published over preprint, else keep the longer text
            if is_preprint(a) != is_preprint(b):
                loser = a if is_preprint(a) else b
            else:
                loser = a if len(docs[a][0]) < len(docs[b][0]) else b
            winner = b if loser == a else a
            dropped[loser] = (winner, s)
            print(f'doc-dup  {s:.2f}  drop {loser}\n                 keep {winner}')

    kept = [n for n in names if n not in dropped]
    seen, out, para_dropped, para_total = set(), [], 0, 0
    for n in kept:
        chunks = []
        for para in re.split(r'\n\s*\n', docs[n][0]):
            toks = norm(para)
            para_total += 1
            if len(toks) >= MIN_PARA:
                g = grams(toks)
                if g and len(g & seen) / len(g) > PARA_SIM:
                    para_dropped += 1
                    continue
                seen |= g
            chunks.append(para.strip())
        out.append('\n\n'.join(c for c in chunks if c))

    corpus = '\n\n'.join(out)
    open('corpus_dedup.txt', 'w').write(corpus)
    print(f'\ndocuments: {len(names)} -> {len(kept)}')
    print(f'paragraphs dropped: {para_dropped}/{para_total}')
    before = sum(len(t.split()) for t, _ in docs.values())
    print(f'words: {before} -> {len(corpus.split())}  -> corpus_dedup.txt')

if __name__ == '__main__':
    main()
