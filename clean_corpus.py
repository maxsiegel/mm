"""Concatenate extracted paper text into a corpus for language modeling.

Drops reference lists, figure/table captions, page furniture, and lines that are
mostly math/numerals, then rejoins hyphenated line breaks into flowing prose.
"""
import re, glob, os

REF_HEAD = re.compile(r'^\s*(references|bibliography|literature cited|acknowledge?ments?|'
                      r'author contributions|competing interests|funding|data availability)\s*[:.]?\s*$', re.I)
CAPTION  = re.compile(r'^\s*(fig(ure)?\.?|table|supplementary|extended data)\b', re.I)
JUNK     = re.compile(r'^(.{0,3}|\d+|.*(doi:|https?://|©|arXiv:|bioRxiv|preprint|'
                      r'article info|keywords:|all rights reserved|creative commons|'
                      r'corresponding author|e-?mail:).*)$', re.I)

# Inline citations: parenthetical author-year, bracketed numerics, and the
# superscript reference numbers that pdftotext leaves glued to a word.
CITE_PAREN = re.compile(r'\s*\((?:[^()]{0,80}?(?:et al\.?|19\d\d|20\d\d)[^()]{0,40}?)\)')
CITE_BRACK = re.compile(r'\s*\[[\d,\s\u2013-]+\]')
CITE_SUP   = re.compile(r'(?<=[a-z])\d{1,3}(?=[\s.,;])')

def nonword_ratio(line):
    letters = sum(c.isalpha() or c.isspace() for c in line)
    return 1 - letters / max(len(line), 1)

def clean(path):
    out = []
    for raw in open(path, errors='ignore'):
        line = raw.rstrip()
        if REF_HEAD.match(line):
            break
        if not line.strip():
            out.append('')
            continue
        if CAPTION.match(line) or JUNK.match(line.strip()) or nonword_ratio(line) > 0.35:
            continue
        out.append(line)
    text = '\n'.join(out)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)      # rejoin hyphenated breaks
    text = re.sub(r'(?<![.!?:;])\n(?=[a-z])', ' ', text)  # unwrap mid-sentence breaks
    text = CITE_PAREN.sub('', text)
    text = CITE_BRACK.sub('', text)
    text = CITE_SUP.sub('', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

if __name__ == '__main__':
    chunks = []
    for f in sorted(glob.glob('text/*.txt')):
        c = clean(f)
        chunks.append(c)
        print(f'{os.path.basename(f):75s} {len(c.split()):6d} words')
    corpus = '\n\n'.join(chunks)
    open('corpus.txt', 'w').write(corpus)
    print('total', len(corpus.split()), 'words ->', 'corpus.txt')
