"""
Adds timestamped print statements (flush=True) around the two most likely
hang points -- the model call itself (chat_query) and the span-parsing step
(perform_span) -- by monkey-patching them at runtime. This lets us see
exactly which one is hanging on the next run, without needing py-spy or
editing algorithms.py/models_extra.py directly.
"""
import time
import functools
from algorithms import Algorithm
from models_extra import ChatModel

# --- instrument ChatModel.chat_query (the actual network call to Ollama) ---
_orig_chat_query = ChatModel.chat_query
@functools.wraps(_orig_chat_query)
def _timed_chat_query(self, msgs):
    t0 = time.time()
    print(f"[TIMING] chat_query() START", flush=True)
    try:
        result = _orig_chat_query(self, msgs)
        print(f"[TIMING] chat_query() END after {time.time()-t0:.2f}s", flush=True)
        return result
    except Exception as e:
        print(f"[TIMING] chat_query() RAISED after {time.time()-t0:.2f}s: {e!r}", flush=True)
        raise
ChatModel.chat_query = _timed_chat_query

# --- instrument Algorithm.perform_span (the parsing/matching step) ---
_orig_perform_span = Algorithm.perform_span
@functools.wraps(_orig_perform_span)
def _timed_perform_span(self, true_tokens=None, verbose=False):
    t0 = time.time()
    print(f"[TIMING] perform_span() START", flush=True)
    result = _orig_perform_span(self, true_tokens=true_tokens, verbose=verbose)
    print(f"[TIMING] perform_span() END after {time.time()-t0:.2f}s", flush=True)
    return result
Algorithm.perform_span = _timed_perform_span

# --- also instrument perform() itself, since perform_span calls it internally ---
_orig_perform = Algorithm.perform
@functools.wraps(_orig_perform)
def _timed_perform(self, verbose=True, deduplicate=True):
    t0 = time.time()
    print(f"[TIMING] perform() START", flush=True)
    result = _orig_perform(self, verbose=verbose, deduplicate=deduplicate)
    print(f"[TIMING] perform() END after {time.time()-t0:.2f}s", flush=True)
    return result
Algorithm.perform = _timed_perform

# --- instrument parse_span specifically, since that's where the manual
#     string-matching / repeated-word logic lives ---
_orig_parse_span = Algorithm.parse_span
@functools.wraps(_orig_parse_span)
def _timed_parse_span(self, answers, typestrings, metadata, true_tokens=None):
    t0 = time.time()
    print(f"[TIMING] parse_span() START (n_answers={len(answers)})", flush=True)
    result = _orig_parse_span(self, answers, typestrings, metadata, true_tokens=true_tokens)
    print(f"[TIMING] parse_span() END after {time.time()-t0:.2f}s", flush=True)
    return result
Algorithm.parse_span = _timed_parse_span


print("[TIMING] Instrumentation installed.", flush=True)


if __name__ == "__main__":
    from models_extra import get_ollama_model
    from run_extra import run_all_datasets_with_model

    qwen = get_ollama_model("qwen2.5:7b")
    results_qwen = run_all_datasets_with_model(qwen, name_meta="qwen2_5_7b_timed_",
                                                dataset_exclude=["genia"], limit=5)
    print("Qwen2.5-7B results:", results_qwen)
