# Amendment 1 to the pre-specified plan (2026-11-27, before any outcome was computed)

Reason: the shared OpenRouter account balance was found to be about US$1.8 after ~2,900 API calls of the main run (credit errors HTTP 402). No FAR/FRR or other outcome had been computed or inspected. To stay within the remaining balance without any payment:

1. API systems (gpt-4o-mini-transcribe, gemini-2.5-flash, gpt-audio-mini) are evaluated on a fixed random subset of 1,000 test utterances (seed 2027: a seeded permutation of all utterance IDs, first 1,000). Already completed calls for these utterances under the same condition and settings are reused, not repeated.
2. Whisper tiny/base/small (local CPU, no cost) remain on all 2,500 utterances; for cross-family comparison they are also reported on the same 1,000-utterance subset.
3. Secondary analyses 4 (alternative verbatim wording) and 5 (API repeatability) move from gemini-2.5-flash to gpt-audio-mini (lower cost) on the first 500 utterances of the same permutation (nested in the 1,000 subset).
4. The confirmatory hypotheses, labels, scoring, statistics, and Holm family are unchanged; H2a/H2b contrasts are computed on the 1,000-utterance subset.
