"""OpenRouter runs. Usage: python run_api.py <system> <cond> [subset_n] [tag]
system: gpt-4o-mini-transcribe | gemini-2.5-flash | gpt-audio-mini
cond: R0 / R1 (ASR) or R0V0 R1V0 R0V1 R1V1 (audio LLM); add tag 'alt' for alternative verbatim wording, 'rep' for repeat run.
Key is read from the desktop key file and never written to logs."""
import sys, os, time, json, base64, threading, random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_rows, RUNS, LOGS, jsonl_append, read_jsonl
from prompts import asr_context, llm_instruction

KEY = open(r'C:\Users\leeji\Desktop\openrouter_apikey.txt').read().strip()
H = {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json',
     'HTTP-Referer': 'https://github.com/leejinkyu0612-hue/capt-reference-masking', 'X-Title': 'capt-reference-masking'}
MODELS = {'gpt-4o-mini-transcribe': 'openai/gpt-4o-mini-transcribe', 'gemini-2.5-flash': 'google/gemini-2.5-flash',
          'gpt-audio-mini': 'openai/gpt-audio-mini'}
system, cond = sys.argv[1], sys.argv[2]
subset_n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '0' else 0
tag = sys.argv[4] if len(sys.argv) > 4 else ''
name = f"{system}_{cond}" + (f"_{tag}" if tag else '') + (f"_n{subset_n}" if subset_n else '')
out = os.path.join(RUNS, name + '.jsonl')
LOG = os.path.join(LOGS, 'api_calls.jsonl')
lock = threading.Lock()
done = {r['uid'] for r in read_jsonl(out) if r.get('hyp') is not None}
rows = load_rows()
if subset_n:
    rng = random.Random(2027)
    ids = sorted(r['uid'] for r in rows)
    perm = rng.sample(ids, len(ids))
    keep = set(perm[:subset_n])
    rows = [r for r in rows if r['uid'] in keep]


def call(r):
    b64 = base64.b64encode(r['audio']['bytes']).decode()
    model = MODELS[system]
    if system == 'gpt-4o-mini-transcribe':
        url = 'https://openrouter.ai/api/v1/audio/transcriptions'
        js = {'model': model, 'input_audio': {'data': b64, 'format': 'wav'}, 'language': 'en', 'temperature': 0}
        if cond == 'R1':
            js['prompt'] = asr_context(r['text'])
        prompt_text = js.get('prompt', '')
    else:
        url = 'https://openrouter.ai/api/v1/chat/completions'
        R, V = int(cond[1]), int(cond[3])
        prompt_text = llm_instruction(r['text'], R, V, alt=(tag == 'alt'))
        js = {'model': model, 'temperature': 0, 'max_tokens': 200,
              'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt_text},
                                                        {'type': 'input_audio', 'input_audio': {'data': b64, 'format': 'wav'}}]}]}
    for attempt in range(6):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=H, json=js, timeout=120)
            lat = time.time() - t0
            j = resp.json()
            if resp.status_code == 200:
                if system == 'gpt-4o-mini-transcribe':
                    hyp = j.get('text'); usage = j.get('usage'); rmodel = model
                else:
                    hyp = j['choices'][0]['message'].get('content'); usage = j.get('usage'); rmodel = j.get('model')
                if hyp is not None:
                    rec = {'uid': r['uid'], 'ref': r['text'], 'hyp': hyp.strip(), 'latency': round(lat, 3)}
                    with lock:
                        jsonl_append(out, rec)
                        jsonl_append(LOG, {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'run': name, 'uid': r['uid'],
                                           'endpoint': url.rsplit('/', 1)[-1], 'model_requested': model, 'model_returned': rmodel,
                                           'prompt': prompt_text, 'temperature': 0, 'status': 200, 'response': hyp,
                                           'usage': usage, 'latency': round(lat, 3), 'attempt': attempt})
                    return True
            with lock:
                jsonl_append(LOG, {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'run': name, 'uid': r['uid'], 'status': resp.status_code,
                                   'error': str(j)[:300], 'attempt': attempt})
        except Exception as e:
            with lock:
                jsonl_append(LOG, {'ts': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'run': name, 'uid': r['uid'], 'status': 'exc',
                                   'error': str(e)[:300], 'attempt': attempt})
        time.sleep(2 * (attempt + 1))
    return False


todo = [r for r in rows if r['uid'] not in done]
ok = 0
with ThreadPoolExecutor(max_workers=3) as ex:
    for f in as_completed([ex.submit(call, r) for r in todo]):
        ok += bool(f.result())
print('done', name, 'ok', ok, 'of', len(todo))


