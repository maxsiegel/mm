"""Word-level Markov chain trained on the Mlynarski paper corpus.

Training is sentence-aware: each sentence is padded with `order` start tokens and
one end token, so generation begins on a real sentence opening and terminates on
a real sentence ending rather than running off the end of a state.

Sampling uses the full conditional distribution at the requested order, backing
off to shorter contexts when a state is unseen (only reachable with --prime).

A sampled end-of-sentence is only honoured when the last emitted word belongs to
the stop set; otherwise the end token is rejected and generation continues. The
stop set defaults to every word the corpus itself ever places sentence-finally
(--stop-mode final); --stop-mode function uses the corpus's function words.

    python3 markov.py --order 2 --sentences 5
    python3 markov.py --order 3 --prime "efficient coding" --sentences 3
    python3 markov.py --stop-mode function --sentences 5
"""
import argparse, random, re, sys
from collections import defaultdict, Counter

BOS, EOS = '<s>', '</s>'

# Closed-class words, used when --stop-mode function is requested. Intersected
# with the corpus vocabulary at run time, so only forms the papers use count.
FUNCTION_WORDS = set('''
a an the this that these those such each every both either neither all any some no
and or but nor so yet then thus hence therefore however whereas while although though
if unless because since as when where which who whom whose what whether
of in on at by for with without within from to into onto over under between among
across through during before after above below against about around per via
is are was were be been being am do does did done has have had having
can could may might must shall should will would
we us our it its they them their he she his her i my you your one ones
not also only just more most less least very much many few other others same
than there here where how why
'''.split())

def sentences(text):
    """Split into sentences and tokenize, dropping fragments that are mostly noise."""
    text = re.sub(r'\s+', ' ', text)
    for raw in re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text):
        toks = re.findall(r"[A-Za-z][A-Za-z'’-]*|[0-9]+(?:\.[0-9]+)?|[,;:()]|[.!?]", raw)
        words = [t for t in toks if t[0].isalpha()]
        if len(words) >= 4 and len(words) / len(toks) > 0.5:
            yield toks

def train(text, order):
    """Build the n-gram tables plus the set of words seen ending a sentence.

    Returns (models, sentence_final_words, n_sentences), where models[k] maps a
    k-word context to a Counter over next tokens.
    """
    models = [defaultdict(Counter) for _ in range(order + 1)]
    final = Counter()
    n_sent = 0
    for toks in sentences(text):
        seq = [BOS] * order + toks + [EOS]
        n_sent += 1
        for w in reversed(toks):            # last actual word, skipping punctuation
            if w[0].isalpha():
                final[w.lower()] += 1
                break
        for i in range(order, len(seq)):
            for k in range(1, order + 1):
                models[k][tuple(seq[i-k:i])][seq[i]] += 1
    return models, final, n_sent


def stop_set(final, models, mode):
    """Words on which a sampled sentence end is allowed to take effect."""
    if mode == 'final':
        return set(final)
    vocab = {t.lower() for ctx in models[1] for t in models[1][ctx]}
    return FUNCTION_WORDS & vocab

def step(models, context, rng, forbid_eos=False):
    """Sample the next token, backing off to shorter contexts when unseen.

    With forbid_eos, the end token is removed from each distribution before
    sampling, so backoff continues past contexts that only ever end sentences.
    """
    for k in range(len(context), 0, -1):
        dist = models[k].get(tuple(context[len(context)-k:]))
        if not dist:
            continue
        items = [(t, c) for t, c in dist.items() if not (forbid_eos and t == EOS)]
        if not items:
            continue
        toks, counts = zip(*items)
        return rng.choices(toks, weights=counts)[0]
    return EOS


def last_word(toks):
    for t in reversed(toks):
        if t[0].isalpha():
            return t.lower()
    return None

def generate(models, order, stops, rng, max_tokens=120, prime=None):
    """Sample one sentence, ending only after a word in `stops`."""
    context = [BOS] * order
    out = []
    if prime:
        primed = re.findall(r"[A-Za-z][A-Za-z'’-]*|[0-9.]+|[,;:()]|[.!?]", prime)
        out.extend(primed)
        context = ([BOS] * order + primed)[-order:]
    for _ in range(max_tokens):
        tok = step(models, context, rng)
        if tok == EOS:
            if out and last_word(out) in stops:
                break
            # Not a licensed stopping point: reject the end token and keep going.
            tok = step(models, context, rng, forbid_eos=True)
            if tok == EOS:                  # nothing else this context can emit
                break
        out.append(tok)
        context = (context + [tok])[-order:]
    return detokenize(out)

def detokenize(toks):
    s = ''
    for t in toks:
        if re.match(r'^[,;:.!?)]$', t) or (s.endswith('(')):
            s += t
        elif not s:
            s = t
        else:
            s += ' ' + t
    return s

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='corpus_dedup.txt')
    p.add_argument('--order', type=int, default=2, help='context length in words')
    p.add_argument('--sentences', type=int, default=5)
    p.add_argument('--prime', help='force the opening words')
    p.add_argument('--seed', type=int, help='RNG seed for reproducible output')
    p.add_argument('--stop-mode', choices=['final', 'function'], default='final',
                   help='which words may end a sentence: words the corpus ends '
                        'sentences with (default), or its function words')
    a = p.parse_args()

    text = open(a.corpus, errors='ignore').read()
    models, final, n_sent = train(text, a.order)
    stops = stop_set(final, models, a.stop_mode)
    rng = random.Random(a.seed)
    states = len(models[a.order])
    print(f'# {n_sent} sentences, {states} order-{a.order} states, '
          f'{sum(sum(c.values()) for c in models[a.order].values())} transitions, '
          f'{len(stops)} stop words ({a.stop_mode})\n', file=sys.stderr)
    for _ in range(a.sentences):
        print(generate(models, a.order, stops, rng, prime=a.prime))

if __name__ == '__main__':
    main()
