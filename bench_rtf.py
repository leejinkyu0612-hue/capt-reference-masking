"""Dedicated CPU timing (pre-specified secondary analysis 6), run after the main runs when no other heavy job was active.
Whisper tiny/base/small, R0 and R1, greedy, 8 threads, first 200 utterances of the seeded subset, one process at a time.
Output: results/rtf_benchmark.json"""
import os, sys, json, time, platform
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import load_rows, wav_array, RUNS, RESULTS
from prompts import asr_context

torch.set_num_threads(8)
ids = json.load(open(os.path.join(RUNS, 'subsets.json')))['subset1000'][:200]
rows = {r['uid']: r for r in load_rows() if r['uid'] in set(ids)}
data = [(wav_array(rows[u]['audio']['bytes']), rows[u]['text']) for u in ids]
P = os.path.join(RESULTS, 'rtf_benchmark.json')
out = json.load(open(P)) if os.path.exists(P) else {}
out.update({'cpu': platform.processor(), 'threads': 8, 'n_utts': len(data), 'torch': torch.__version__})
for size in (sys.argv[1:] or ['tiny', 'base', 'small']):
    proc = WhisperProcessor.from_pretrained('openai/whisper-' + size)
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-' + size).eval()
    for cond in ('R0', 'R1'):
        tot = dur = 0.0
        for (x, sr), ref in data:
            f = proc(x, sampling_rate=sr, return_tensors='pt').input_features
            kw = dict(language='en', task='transcribe', num_beams=1, do_sample=False)
            if cond == 'R1':
                kw['prompt_ids'] = torch.tensor(proc.get_prompt_ids(asr_context(ref)))
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(f, **kw)
            tot += time.perf_counter() - t0; dur += len(x) / sr
        out[f'{size}_{cond}'] = {'rtf': tot / dur, 'mean_sec': tot / len(data), 'audio_sec': dur}
        print(size, cond, round(tot / dur, 3), flush=True)
        json.dump(out, open(P, 'w'), indent=1)
json.dump(out, open(os.path.join(RESULTS, 'rtf_benchmark.json'), 'w'), indent=1)

