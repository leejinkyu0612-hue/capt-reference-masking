# Supplementary Table S1. Delta-FAR (pp) for canonical-text prompting under alternative analyses

|system|dFAR_primary_pp|acc_le5|acc_le3|age_le15|age_gt15|digits_spelled_out(post hoc)|frequent_words_only(post hoc)|
|---|---|---|---|---|---|---|---|
|whisper-tiny|48.7|50.9|46.3|48.0|48.8|48.5|48.5|
|whisper-base|51.3|53.1|48.1|54.2|50.5|51.2|51.2|
|whisper-small|50.1|51.6|49.4|51.4|49.7|50.1|49.2|
|gpt-4o-mini-transcribe|25.8|23.0|21.1|31.7|13.8|25.8|25.7|
|gemini-2.5-flash|72.4|67.4|74.2|79.0|70.6|72.7|71.7|
|gpt-audio-mini|72.9|69.7|76.4|79.0|71.3|72.6|71.5|

Whisper models: 2,500 utterances; API systems: 1,000-utterance subset (gpt-4o-mini-transcribe: 811 paired utterances). Source: outputs/results/analysis_main*.json.
