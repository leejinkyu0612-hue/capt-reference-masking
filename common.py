"""Shared utilities: data loading, text normalisation, word alignment, labels."""
import io, os, re, json, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)                      # EXP_실험
DATA = os.path.join(EXP, 'data', 'test-00000-of-00001.parquet')
RUNS = os.path.join(EXP, 'runs')
LOGS = os.path.join(EXP, 'logs')
RESULTS = os.path.join(EXP, 'results')
for d in (RUNS, LOGS, RESULTS):
    os.makedirs(d, exist_ok=True)


def utt_id(row):
    return os.path.splitext(os.path.basename(row['audio']['path']))[0]


def load_rows(with_audio=True):
    import pyarrow.parquet as pq
    t = pq.read_table(DATA)
    if not with_audio:
        t = t.drop(['audio']).append_column('audio_path', t.column('audio').combine_chunks().field('path'))
        rows = t.to_pylist()
        for r in rows:
            r['uid'] = os.path.splitext(os.path.basename(r['audio_path']))[0]
        return rows
    rows = t.to_pylist()
    for r in rows:
        r['uid'] = utt_id(r)
    return rows


def wav_array(b):
    w = wave.open(io.BytesIO(b))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return x, w.getframerate()


_punct = re.compile(r"[^A-Z0-9' ]+")


def norm_words(text):
    t = (text or '').upper().replace('\u2019', "'")
    t = _punct.sub(' ', t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def align(ref, hyp):
    """Levenshtein word alignment. Returns list over ref words: True if matched to identical hyp word."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + c)
    # backtrace (prefer diagonal)
    i, j = n, m
    acc = [False] * n
    while i > 0 and j > 0:
        c = 0 if ref[i - 1] == hyp[j - 1] else 1
        if D[i, j] == D[i - 1, j - 1] + c:
            acc[i - 1] = (c == 0); i -= 1; j -= 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return acc, int(D[n, m])


def word_labels(row):
    """Per target word: primary label (1 mispronounced, 0 correct, None excluded) + sensitivity labels."""
    out = []
    for w in row['words']:
        acc = w['accuracy']
        mis = bool(w['mispronunciations'])
        prim = 1 if mis else (0 if acc == 10 else None)
        s5 = 1 if acc <= 5 else (0 if acc == 10 else None)
        s3 = 1 if acc <= 3 else (0 if acc == 10 else None)
        out.append({'text': w['text'], 'primary': prim, 'acc_le5': s5, 'acc_le3': s3, 'accuracy': acc})
    return out


def jsonl_append(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]
