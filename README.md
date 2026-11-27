# capt-reference-masking

Code, fixed prompts and logged outputs for the paper
**"Does Canonical-Text Prompting Mask Mispronunciations? Context and Verbatim Instructions in ASR-Based Read-Aloud Pronunciation Feedback"** (submitted to IC4e 2027).

The study asks whether giving a speech recognizer the sentence a learner is reading (canonical text) hides real pronunciation errors in word-level feedback. It compares the same audio with and without canonical-text context, and for audio LLMs with and without a verbatim instruction, against expert word labels.

## Data
- speechocean762 test split (2,500 utterances, 125 Mandarin-L1 speakers), CC BY 4.0, from Hugging Face `mispeech/speechocean762`.
- Audio is **not** redistributed. Only transcripts, utterance IDs and computed results are stored.

## Plan before outcomes
- `PREREGISTRATION.md`: hypotheses, metrics (FAR, FRR), alignment rule and tests, committed before the main run.
- `AMENDMENT_1.md`: budget amendment (API systems on a seeded 1,000-utterance subset), committed before any API outcome was computed.
- Analyses added after reviewer feedback are in `posthoc.py`, `r1_analyses.py`, `r2_analyses.py`, `r3_analyses.py` and are labelled post hoc / exploratory in the paper.

## Systems
| System | Run | Conditions |
|---|---|---|
| Whisper tiny / base / small | local CPU (Transformers), greedy; base also beam 5 on 500 utts | R0 (no context), R1 (canonical text as prompt) |
| gpt-4o-mini-transcribe | API via OpenRouter, 1,000-utt subset | R0, R1 (R1 incomplete: 827 of 1,000 because credit ran out; 811 paired) |
| gemini-2.5-flash, gpt-audio-mini | API via OpenRouter, 1,000-utt subset | R0/R1 x V0/V1 (verbatim instruction) |

## Files
- `common.py` data loading, normalisation, alignment and FAR/FRR.
- `prompts.py` exact prompt wording.
- `seed_subsets*.py`, `seed_beam.py` seeded subsets (`outputs/runs/subsets.json`).
- `run_whisper.py`, `run_api.py`, `run_control.py`, `bench_rtf.py` runs.
- `analysis.py` pre-specified analysis; `outputs/results/analysis_main.json`.
- `outputs/runs/*.jsonl` one line per utterance: `uid`, `ref`, `hyp`, timing.
- `outputs/logs/api_calls.jsonl` API call log (model, condition, status, tokens, cost); no audio, no keys.
- `outputs/results/Supplementary_Table_S1.*` sensitivity by threshold and age.
- `figures.py`, `make_figures_v2.py`, `figures_out/` paper figures.

## Reproduce
```
pip install torch transformers datasets jiwer numpy scipy pandas matplotlib
python run_whisper.py --model tiny --cond R0   # etc.
python analysis.py
```
API runs need `OPENROUTER_API_KEY` in the environment (not stored here).

## Notes
- Some audio-LLM outputs are refusals or non-transcripts; they are counted and reported, and a sensitivity analysis excludes them.
- One gemini R0V0 request failed (999 paired utterances).

## Disclosure
Authors: Jinkyu Lee and Chang Soo Sung (corresponding author), Department of Technology Entrepreneurship, Graduate School, Dongguk University, Seoul, Republic of Korea. Jinkyu Lee is the founder and CEO of HappyAI; no HappyAI product is evaluated. ChatGPT and Claude were used for language editing and review support of the manuscript; all numbers come from the logged outputs in this repository.

## License
Code: MIT. Derived transcripts and results: CC BY 4.0 (same as speechocean762).
