"""Control experiment added after review round R2 (exploratory, local CPU only).
Whisper base, greedy, on 300 utterances (seeded sample from the 1,000-utterance subset):
  C: silence of the same duration + target-sentence prompt
  D: audio of a different utterance (derangement) + target-sentence prompt
  E: real audio + an unrelated sentence (another utterance's text) as prompt
Output: runs/control_whisper-base_{C,D,E}.jsonl"""
import os, sys, json, random, time
import numpy as np, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from common import load_rows, wav_array, RUNS, jsonl_append, read_jsonl
from prompts import asr_context

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
subs = json.load(open(os.path.join(RUNS, 'subsets.json')))
rng = random.Random(2027)
pick = rng.sample(subs['subset1000'], 300)
rows = {r['uid']: r for r in load_rows() if r['uid'] in set(pick)}
order = [u for u in pick if u in rows]
der = order[1:] + order[:1]          # derangement: each utterance gets the next one's audio / text
proc = WhisperProcessor.from_pretrained('openai/whisper-base')
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-base').eval()


def run(cond):
    out = os.path.join(RUNS, f'control_whisper-base_{cond}.jsonl')
    done = {x['uid'] for x in read_jsonl(out)}
    for u, v in zip(order, der):
        if u in done:
            continue
        r = rows[u]
        x, sr = wav_array(r['audio']['bytes'])
        if cond == 'C':
            x = np.zeros_like(x); prompt = asr_context(r['text'])
        elif cond == 'D':
            x, sr = wav_array(rows[v]['audio']['bytes']); prompt = asr_context(r['text'])
        else:
            prompt = asr_context(rows[v]['text'])
        f = proc(x, sampling_rate=sr, return_tensors='pt').input_features
        with torch.no_grad():
            ids = model.generate(f, language='en', task='transcribe', num_beams=1, do_sample=False,
                                 prompt_ids=torch.tensor(proc.get_prompt_ids(prompt)), max_new_tokens=96)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        jsonl_append(out, {'uid': u, 'ref': r['text'], 'hyp': hyp, 'prompt': prompt, 'audio_from': v if cond == 'D' else (u if cond == 'E' else 'silence')})


for c in ('C', 'D', 'E'):
    run(c); print('done', c)
