"""Fixed prompt wording (pre-specified, committed before the main run)."""

def sentence_case(ref: str) -> str:
    s = ref.strip().lower()
    return (s[:1].upper() + s[1:]).rstrip('.') + '.'

# ASR decoding context (Whisper prompt_ids / transcription API `prompt`)
def asr_context(ref: str) -> str:
    return sentence_case(ref)

_BASE = "Transcribe the English speech in this audio. Output only the transcript."
_VERB = ("Transcribe this English speech verbatim, exactly as it was pronounced. "
         "The speaker is a language learner: do not correct mispronounced words; "
         "if a word was mispronounced, write the word or spelling that matches what was actually said. "
         "Output only the transcript.")
_REF = 'The speaker was asked to read this sentence aloud: "{ref}" '
# robustness wording (secondary analysis 4)
_VERB_ALT = ("Write down exactly what you hear, word by word, like a strict listener. "
             "Do not fix errors and do not guess the intended words. Output only the transcript.")

def llm_instruction(ref: str, R: int, V: int, alt: bool = False) -> str:
    body = (_VERB_ALT if alt else _VERB) if V else _BASE
    return (_REF.format(ref=sentence_case(ref)) + body) if R else body
