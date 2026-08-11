"""
run_extra.py

run.py's `run()` function only knows how to build two models internally:
OpenAIGPT() when gpt=True, or a hardcoded Alpaca(size='base') when gpt=False.
There's no way to pass in an arbitrary model instance (e.g. our new Ollama or
newer-OpenAI ChatModel from models_extra.py) without editing run.py itself.

This file re-implements `run()` and `run_all_datasets()` so they accept a
pre-built model object directly. It reuses the eval_* functions from run.py
unchanged (those already accept `model` as an argument).
"""
from algorithms import Algorithm
from run import (eval_conll, eval_genia, eval_cross_ner, eval_few_nerd_intra,
                  eval_tweetner, eval_fabner)


def run_with_model(model, dataset="conll", subdataset=None, exemplar=True, coT=True,
                    defn=True, tf=True, name_meta="", n_runs=1, limit=5,
                    sleep_between_queries=None):
    print(f"Running for: {dataset}, {subdataset}, model={getattr(model, 'model', model)}")

    eval_fn_map = {
        "conll": eval_conll,
        "genia": eval_genia,
        "crossner": eval_cross_ner,
        "fewnerd": eval_few_nerd_intra,
        "tweetner": eval_tweetner,
        "fabner": eval_fabner,
    }
    if dataset not in eval_fn_map:
        raise ValueError(f"Unknown Dataset: {dataset}")
    eval_fn = eval_fn_map[dataset]

    # Use the model's own pacing if it defines one (e.g. OpenAI rate limits);
    # local Ollama models typically don't need any delay.
    if sleep_between_queries is None:
        sleep_between_queries = getattr(model, "seconds_per_query", None)

    micros, macros, df = eval_fn(
        model, Algorithm(), n_runs=n_runs, sleep_between_queries=sleep_between_queries,
        limit=limit, exemplar=exemplar, coT=coT, defn=defn, tf=tf, add_info=subdataset,
    )

    print(f"Final Results For {name_meta} | {dataset} "
          f"{'(' + subdataset + ')' if subdataset is not None else ''} "
          f"| CoT {coT} | Exemplar {exemplar} (tf {tf}) | Defn {defn}")
    print(f"Micro f1_means: {micros.mean()}  Micro f1_stds: {micros.std()}")
    print(f"Macro f1_means: {macros.mean()}  Macro f1_stds: {macros.std()}")

    save_path = f"results/{name_meta}{dataset}{subdataset}.csv"
    df.to_csv(save_path, index=False)
    return micros, macros


def run_all_datasets_with_model(model, name_meta="", exemplar=True, coT=True, defn=True, tf=True,
                                 n_runs=1, limit=5, dataset_exclude=(), subdataset_exclude=()):
    d = {}
    datasets = ["conll", "genia", "crossner", "fewnerd", "tweetner", "fabner"]
    subdatasets = {"crossner": ['politics', 'literature', 'ai', 'science', 'music'],
                   'fewnerd': ["test"]}

    for dataset in datasets:
        if dataset in dataset_exclude:
            continue
        sub = subdatasets.get(dataset, None)
        if sub is None:
            micro, macro = run_with_model(model, dataset=dataset, coT=coT, exemplar=exemplar,
                                           defn=defn, tf=tf, name_meta=name_meta,
                                           n_runs=n_runs, limit=limit)
            d[dataset] = [(macro * 100).mean(), (macro * 100).std(),
                          (micro * 100).mean(), (micro * 100).std()]
        else:
            for s in sub:
                if s in subdataset_exclude:
                    continue
                micro, macro = run_with_model(model, dataset=dataset, subdataset=s, coT=coT,
                                               exemplar=exemplar, defn=defn, tf=tf,
                                               name_meta=name_meta, n_runs=n_runs, limit=limit)
                d[f"{dataset}_{s}"] = [(macro * 100).mean(), (macro * 100).std(),
                                        (micro * 100).mean(), (micro * 100).std()]
    return d


if __name__ == "__main__":
    # Example: compare a local Ollama Qwen model against a newer hosted OpenAI model
    from models_extra import get_ollama_model, get_openai_model

    qwen = get_ollama_model("qwen2.5:7b")
    results_qwen = run_all_datasets_with_model(qwen, name_meta="qwen2_5_7b_",
                                                dataset_exclude=["genia"])
    print("Qwen2.5-7B results:", results_qwen)

    # gpt4o_mini = get_openai_model("gpt-4o-mini")
    # results_gpt4o_mini = run_all_datasets_with_model(gpt4o_mini, name_meta="gpt4o_mini_",
    #                                                   dataset_exclude=["genia"])
    # print("gpt-4o-mini results:", results_gpt4o_mini)
