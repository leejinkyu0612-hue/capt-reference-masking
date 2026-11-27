"""Analyses added after review round R3 (exploratory/post-hoc; stored outputs only).
1) speaker influence: share of speakers with a positive Delta-FAR, leave-one-speaker-out range
2) severity: Delta-FAR for mispronounced words with word accuracy 0-3 vs 4-6 (and >= 7)
3) sentence clustering: number of distinct target sentences; sentence-cluster bootstrap CI for Delta-FAR
4) alignment sensitivity: 'bag-of-words' acceptance (target word appears anywhere in the transcript)
Output: results/r3_analyses.json"""
import os, json
import numpy as np
import analysis as A
from common import RUNS, RESULTS, read_jsonl, norm_words, load_rows

rows = {r['uid']: r for r in load_rows(with_audio=False)}
pos = A.LAB['primary'] == 1
SYS = [('Whisper tiny', 'whisper-tiny_R0', 'whisper-tiny_R1', A.mask_all), ('Whisper base', 'whisper-base_R0', 'whisper-base_R1', A.mask_all),
       ('Whisper small', 'whisper-small_R0', 'whisper-small_R1', A.mask_all), ('gpt-4o-mini-transcribe', 'gpt-4o-mini-transcribe_R0_n1000', 'gpt-4o-mini-transcribe_R1_n1000', A.mask_1000),
       ('gemini-2.5-flash', 'gemini-2.5-flash_R0V0_n1000', 'gemini-2.5-flash_R1V0_n1000', A.mask_1000), ('gpt-audio-mini', 'gpt-audio-mini_R0V0_n1000', 'gpt-audio-mini_R1V0_n1000', A.mask_1000)]
acc_word = np.array([rows[u]['words'][k]['accuracy'] for u, k in zip(A.UID, A.W[:, 1])])
sent = np.array([rows[u]['text'] for u in A.UID])
sent_ids = {s: i for i, s in enumerate(sorted(set(sent)))}
SI = np.array([sent_ids[s] for s in sent])
out = {'speaker': [], 'severity': [], 'sentence': {}, 'bag_of_words': []}
for name, a0, b0, m in SYS:
    a, b = A.runs[a0][0], A.runs[b0][0]
    sp = m & (a >= 0) & (b >= 0) & pos
    d_all = b[sp].mean() - a[sp].mean()
    spk = np.unique(A.SPI[sp])
    per = [(b[sp & (A.SPI == s)].mean() - a[sp & (A.SPI == s)].mean()) for s in spk]
    loo = [(b[sp & (A.SPI != s)].mean() - a[sp & (A.SPI != s)].mean()) for s in spk]
    out['speaker'].append({'system': name, 'n_speakers': int(len(spk)), 'share_positive': float(np.mean(np.array(per) > 0)), 'share_negative': float(np.mean(np.array(per) < 0)),
                           'loo_min': float(min(loo)), 'loo_max': float(max(loo)), 'dFAR': float(d_all)})
    sev = {}
    for lab, cond in (('acc0_3', acc_word <= 3), ('acc4_6', (acc_word >= 4) & (acc_word <= 6)), ('acc7plus', acc_word >= 7)):
        s_ = sp & cond
        sev[lab] = {'n': int(s_.sum()), 'FAR_R0': float(a[s_].mean()) if s_.any() else None, 'FAR_R1': float(b[s_].mean()) if s_.any() else None}
    out['severity'].append({'system': name, **sev})
    # sentence-cluster bootstrap
    r = np.random.default_rng(2027)
    sids = np.unique(SI[sp]); ns = np.bincount(SI[sp], minlength=len(sent_ids)).astype(float)
    sa = np.bincount(SI[sp], a[sp].astype(float), len(sent_ids)); sb = np.bincount(SI[sp], b[sp].astype(float), len(sent_ids))
    ds = []
    for _ in range(10000):
        w = np.bincount(r.choice(sids, size=len(sids), replace=True), minlength=len(sent_ids)).astype(float)
        ds.append(((w * sb).sum() - (w * sa).sum()) / (w * ns).sum())
    out['sentence'][name] = {'n_sentences_with_errors': int(len(sids)), 'dFAR': float(d_all), 'CI_sentence_cluster': np.percentile(ds, [2.5, 97.5]).tolist()}
    # bag-of-words acceptance
    def bow(run):
        hyp = {x['uid']: set(norm_words(x['hyp'])) for x in read_jsonl(os.path.join(RUNS, run + '.jsonl')) if x.get('hyp') is not None}
        v = np.full(len(A.UID), -1)
        for i, (u, k) in enumerate(zip(A.UID, A.W[:, 1])):
            if u in hyp:
                v[i] = int(all(t in hyp[u] for t in norm_words(rows[u]['words'][k]['text'])))
        return v
    ba, bb = bow(a0), bow(b0)
    s2 = m & (ba >= 0) & (bb >= 0) & pos
    out['bag_of_words'].append({'system': name, 'FAR_R0': float(ba[s2].mean()), 'FAR_R1': float(bb[s2].mean()), 'dFAR': float(bb[s2].mean() - ba[s2].mean())})
out['n_distinct_sentences_test'] = int(len(set(r['text'] for r in rows.values())))
json.dump(out, open(os.path.join(RESULTS, 'r3_analyses.json'), 'w', encoding='utf-8'), indent=1)
print(json.dumps(out, indent=1))
