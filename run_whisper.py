"""Local CPU Whisper runs. Usage: python run_whisper.py <tiny|base|small> <R0|R1> [beams] [threads]"""
import sys, os, time, json, platform
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import transformers
from common import load_rows, wav_array, RUNS, jsonl_append, read_jsonl
from prompts import asr_context

size, cond = sys.argv[1], sys.argv[2]
beams = int(sys.argv[3]) if len(sys.argv) > 3 else 1
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 6
torch.set_num_threads(threads)
name = f"whisper-{size}_{cond}" + (f"_beam{beams}" if beams > 1 else "")
out = os.path.join(RUNS, name + '.jsonl')
done = {r['uid'] for r in read_jsonl(out)}
proc = WhisperProcessor.from_pretrained('openai/whisper-' + size)
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-' + size).eval()
meta = {'run': name, 'model': 'openai/whisper-' + size, 'condition': cond, 'beams': beams, 'threads': threads,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'cpu': platform.processor(),
        'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
json.dump(meta, open(os.path.join(RUNS, name + '.meta.json'), 'w'), indent=1)
rows = load_rows()
for r in rows:
    if r['uid'] in done:
        continue
    x, sr = wav_array(r['audio']['bytes'])
    feats = proc(x, sampling_rate=sr, return_tensors='pt').input_features
    kw = dict(language='en', task='transcribe', num_beams=beams, do_sample=False)
    if cond == 'R1':
        kw['prompt_ids'] = torch.tensor(proc.get_prompt_ids(asr_context(r['text'])))
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(feats, **kw)
    el = time.perf_counter() - t0
    txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
    if cond == 'R1':  # strip echoed prompt if any (decode skips prompt tokens in recent versions)
        pass
    jsonl_append(out, {'uid': r['uid'], 'ref': r['text'], 'hyp': txt.strip(), 'sec': round(el, 4),
                       'dur': round(len(x) / sr, 4)})
print('done', name)
