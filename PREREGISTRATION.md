# 사전계획 (Pre-specified analysis plan) — 실험 전 고정

Written 2026-11-27 (KST) before any main-experiment output was generated. Committed to the private repository before experiments; analyses not listed here are reported as exploratory.

## Title (working)
Does knowing the target sentence make speech recognizers hide pronunciation errors? Reference-text context and verbatim instructions in read-aloud pronunciation feedback.

## Pilot disclosure
Before this plan, a feasibility pilot ran Whisper tiny/base/small on 10 test utterances (timing only; real-time factor about 0.06/0.11/0.40 on CPU) and called each API once on one utterance to confirm that the parameters work. No FAR/FRR was computed in the pilot. The pilot showed that the OpenRouter whisper-large-v3 endpoint returned identical text with and without a prompt, so it is excluded (prompt use cannot be verified).

## Data
- speechocean762 test split (Hugging Face mispeech/speechocean762, CC BY 4.0): 2,500 read utterances, 125 speakers, 15,967 target words.
- Positive (mispronounced) word: at least one phone whose expert-average accuracy is below 0.5 (the corpus "mispronunciations" block). n = 783.
- Negative (correctly pronounced) word: word accuracy score = 10. n = 14,319.
- Other words (accented, no phone < 0.5) are excluded from FAR/FRR and reported only descriptively.
- Sensitivity labels (pre-specified): positive = word accuracy <= 5; positive = word accuracy <= 3.

## Systems and conditions
- ASR: Whisper tiny, base, small (openai/whisper-*, Hugging Face transformers, CPU, English, task transcribe, greedy decoding, no timestamps); gpt-4o-mini-transcribe (OpenRouter /audio/transcriptions, temperature 0).
  - R0: no context. R1: target sentence as decoding context (Whisper prompt_ids; API `prompt` argument). Prompt text = target sentence in sentence case with a final period.
- Audio LLMs: google/gemini-2.5-flash and openai/gpt-audio-mini (OpenRouter chat completions, temperature 0, max_tokens 200).
  - 2x2: R (target sentence given in the instruction or not) x V (verbatim instruction or not). Exact prompt wording is fixed in code (prompts.py) before the run.
- Each system-condition is run once on all 2,500 utterances.

## Scoring
- Normalize text: upper case, remove punctuation except apostrophes, collapse spaces; digits are left as is.
- Align hypothesis words to target words with Levenshtein alignment (unit costs). A target word is "accepted" if aligned to an identical word, otherwise "rejected".
- FAR = accepted positives / positives. FRR = rejected negatives / negatives. WER against the target sentence (proxy; no verbatim reference exists).

## Hypotheses (confirmatory)
- H1: For each ASR system, R1 increases FAR relative to R0 (4 contrasts).
- H2a: For each audio LLM, R increases FAR (main effect within V0: R1V0 vs R0V0; 2 contrasts).
- H2b: For each audio LLM, V decreases FAR (within R0: R0V1 vs R0V0; 2 contrasts).
- H3: Among Whisper models under R0, FAR increases with model size (small minus tiny > 0; 1 contrast).
- FRR changes for the same contrasts are reported as the cost side (not separately tested for direction).

## Statistics
- Effect: difference in FAR (percentage points) with 95% speaker-cluster bootstrap CI (2,000 resamples of speakers, seed 2027).
- Test: exact McNemar on paired word outcomes; Holm correction over the 9 confirmatory contrasts (alpha 0.05).
- A hypothesis is supported if the Holm-adjusted p < 0.05 and the bootstrap CI excludes 0 in the predicted direction.

## Robustness and secondary analyses (pre-specified)
1. Sensitivity labels (word accuracy <= 5, <= 3).
2. Age subgroups (speaker age <= 15 vs > 15).
3. Whisper base with beam search 5 (R0, R1).
4. Alternative wording of the verbatim instruction for gemini-2.5-flash on a random 500-utterance subset (seed 2027).
5. API repeatability: R0V0 and R1V0 for gemini-2.5-flash re-run on the same 500-utterance subset; report word-level agreement.
6. CPU efficiency: real-time factor per Whisper model (i5-13600K, 8 threads) and median API latency (descriptive only).
7. Interaction R x V for audio LLMs (R1V1 vs others) reported descriptively.

## Exploratory (not confirmatory)
- Token-confidence signal, sentence length, word frequency, gender. Any analysis added after seeing results will be labelled exploratory/post-hoc.

## Budget and integrity
- OpenRouter budget cap for this study: US$8. All API requests and responses are logged as JSONL. Audio files are not redistributed.
