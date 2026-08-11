"""
patch_eval_dataset.py

Run this once, on the pod, from /pvcvolume/PromptNER:
    python3 patch_eval_dataset.py

It replaces the entire eval_dataset() function in run.py with a version
that uses a hard, subprocess-based timeout (via timeout_utils.py) instead
of signal.alarm(), which we found does not reliably interrupt the hang
we hit in CrossNER evaluation.
"""
import re

with open("run.py") as f:
    content = f.read()

new_eval_dataset = '''from timeout_utils import run_with_hard_timeout

def eval_dataset(val, model, algorithm, sleep_between_queries=None, print_every=10, per_example_timeout=60):
    algorithm.set_model_fn(model)
    columns = ["text", "entities", "truth", "pred", "meta", "f1"]
    data = []
    preds, truths = [], []
    for i in range(len(val)):
        q = val.iloc[i]
        algorithm.set_para(q["text"])
        subdata = [q["text"], q["entities"]]
        flag = False
        while not flag:
            try:
                true_tokens = None
                if "true_tokens" in val.columns:
                    true_tokens = q["true_tokens"]
                ok, result = run_with_hard_timeout(
                    algorithm.perform_span,
                    kwargs={"true_tokens": true_tokens, "verbose": False},
                    timeout=per_example_timeout,
                )
                if not ok:
                    print(f"Skipping example {i}: exceeded {per_example_timeout}s hard timeout")
                    flag = True
                    continue
                span_pred, meta = result
                p = [span_pred]
                t = [q['exact_types']]
                preds.append(span_pred)
                truths.append(q['exact_types'])
                mini_f1 = f1_score(t, p)
                subdata.extend([span_pred, meta, mini_f1])
                data.append(subdata)
                f1_micro = f1_score(truths, preds, average="micro")
                flag = True
            except openai.RateLimitError:
                time.sleep(0.5)
            except IndexError:
                flag = True
        if i % print_every == 0:
            print(f"Iteration {i}: micro f1: {f1_micro if preds else 0.0}, macro f1: {f1_score(truths, preds, average='macro') if preds else 0.0}")
        if sleep_between_queries:
            time.sleep(sleep_between_queries)
    df = pd.DataFrame(data, columns=columns)
    f1_micro = f1_score(truths, preds, average="micro") if preds else 0.0
    f1_macro = f1_score(truths, preds, average="macro") if preds else 0.0
    print(f"Finally: micro f1: {f1_micro}, macro f1: {f1_macro}")
    return f1_micro, f1_macro, df
'''

# Replace from "def eval_dataset" up to (but not including) the next
# top-level "def " -- this removes the old signal-based version entirely,
# including any leftover _ExampleTimeout / signal imports tied to it.
pattern = re.compile(r"(?:^import signal\n)?(?:^class _ExampleTimeout.*?\n\n)?(?:^def _timeout_handler.*?\n\n)?^def eval_dataset\(.*?\n(?=^def )", re.DOTALL | re.MULTILINE)

new_content, n = pattern.subn(new_eval_dataset + "\n", content, count=1)

if n == 0:
    print("ERROR: could not find eval_dataset() to replace. No changes made.")
    print("Please check run.py manually.")
else:
    with open("run.py", "w") as f:
        f.write(new_content)
    print(f"Replaced eval_dataset() successfully ({n} replacement made).")
