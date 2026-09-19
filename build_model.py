"""Export the trained chain as transition counts for the browser frontend.

Ships integer count tables only - no corpus text is included in the payload.
Layout is a vocabulary array plus, for each order, four parallel arrays in
CSR-style form (contexts, row offsets, next-token ids, counts), which keeps the
JSON small and lets the page rebuild lookup maps in one pass.
"""
import json, sys
import markov

ORDER = 2

def csr(model, ids):
    """Flatten {context: Counter} into parallel context/offset/next/count arrays."""
    ctx, off, nxt, cnt = [], [0], [], []
    for context, dist in sorted(model.items()):
        ctx.extend(ids[t] for t in context)
        for tok, c in sorted(dist.items()):
            nxt.append(ids[tok]); cnt.append(c)
        off.append(len(nxt))
    return {'ctx': ctx, 'off': off, 'nxt': nxt, 'cnt': cnt}

def main(corpus='corpus_dedup.txt', out='docs/model.json'):
    text = open(corpus, errors='ignore').read()
    models, final, n_sent = markov.train(text, ORDER)

    vocab = sorted({t for k in (1, ORDER) for ctx in models[k]
                    for t in list(ctx) + list(models[k][ctx])})
    ids = {t: i for i, t in enumerate(vocab)}

    stops = {m: sorted(ids[t] for t in vocab
                       if t.lower() in markov.stop_set(final, models, m))
             for m in ('final', 'function')}

    payload = {'order': ORDER, 'sentences': n_sent, 'vocab': vocab,
               'bos': ids[markov.BOS], 'eos': ids[markov.EOS],
               'm1': csr(models[1], ids), 'm2': csr(models[ORDER], ids),
               'stops': stops}
    json.dump(payload, open(out, 'w'), separators=(',', ':'))
    import os
    print(f'{len(vocab)} vocab, {len(models[ORDER])} order-2 states, '
          f'{sum(sum(c.values()) for c in models[ORDER].values())} transitions')
    print(f'{out}: {os.path.getsize(out)/1e6:.1f} MB')

if __name__ == '__main__':
    main(*sys.argv[1:])
