# FLAN-T5 replication runner

`t5_replication_extra.py` is an isolated FLAN-T5-XXL control experiment. It
does not modify the original PromptNER parser, metric, or evaluation scripts.
For each selected dataset it evaluates both the original parser (`legacy`) and
the opt-in schema-validating parser (`normalized`), writing one CSV per seed
and a `summary.json` report.

## Pod setup

Use Python 3.10 or later and a CUDA-enabled PyTorch installation suitable for
the Pod. FLAN-T5-XXL is intended to run with an NVIDIA GPU and 8-bit loading.

```sh
python3 -m pip install -r requirements-t5.txt
python3 -m nltk.downloader stopwords
```

Download and extract the PromptNER data archive linked from `readme.md` so the
repository contains `data/CrossNER`, `data/FewNERD`, and the CoNLL files
expected by `load_conll2003`. Hugging Face will download the FLAN-T5 weights on
the first run (authenticate with `huggingface-cli login` first if the Pod
requires it).

## Run

Start with a small GPU smoke run; it writes CSV output and a summary beneath
`results/t5xxl_replication`:

```sh
python3 t5_replication_extra.py --smoke --datasets conll
```

Run the full intended matrix (seven datasets, 500 sampled test examples, five
seeds, both parser modes):

```sh
python3 t5_replication_extra.py --limit 500 --runs 5
```

For a narrower resume/run, for example only the normalized parser on two
CrossNER domains:

```sh
python3 t5_replication_extra.py --datasets politics literature --modes normalized --limit 500 --runs 5
```

Use `--output-dir /path/to/output` to put results on persistent Pod storage.
Pass `--no-8bit` only when the Pod has enough memory for a full-precision
model; normal 8-bit mode requires CUDA.
