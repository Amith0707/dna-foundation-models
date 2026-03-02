# DNABERT-2 Integration for DeepChem

This repo documents work for integrating DNABERT-2 into DeepChem.
A cleaned up Kaggle notebook will be linked here once ready.

---

## What is DNABERT-2

DNABERT-2 is a 117M parameter foundation model for DNA sequences. Unlike the original DNABERT
which used k-mer tokenization, DNABERT-2 uses Byte-Pair Encoding and is built on the MosaicBERT
architecture with ALiBi positional embeddings (another positional embedding technique used rather than the sinusoidal one)
and an optional Flash Attention kernel.

Paper: [arXiv:2306.15006](https://arxiv.org/abs/2306.15006)

---

## What's been done so far

A `DNABERT2` class that subclasses `HuggingFaceModel` following the same pattern as `Chemberta`.
It supports five tasks out of the box:

- `classification` — binary or multi-class sequence classification
- `regression` — single-target scalar regression  
- `mtr` — multi-task regression
- `mlm` — masked language modelling
- `feature_extractor` — CLS-token embeddings

---

## Integration challenges and how I solved them

This was not a straightforward wrap. DNABERT-2 ships with custom remote code that has several
compatibility issues on modern environments.Sometimes it used to run on my local code editor but failed in kaggle/colab.
Here is each issues that I faced and the fix to ensure it's environment agnostic.

### 1. bert_layers.py never reaches the cache

The patch needs to modify `bert_layers.py` in the HuggingFace modules cache before the model
loads. The problem is that `AutoTokenizer` and `AutoConfig` only pull `configuration_bert.py` —
`bert_layers.py` only downloads when the actual model architecture is requested. On Kaggle this
meant the patch function would run and find nothing, then the unpatched file would arrive later
and break everything.

Fix: call `AutoModel.from_pretrained` as a warm-up before anything else. The init fails because
`pad_token_id` isn't set on the raw config yet that's expected and safe to swallow. We only
need the file on disk.
```python
try:
    _ = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    del _
except Exception:
    pass
```

### 2. ALiBi meta-device crash

On PyTorch >= 2.0 with transformers >= 4.38, HuggingFace initialises tensors on the `meta`
device during lazy loading. DNABERT-2's `BertEncoder.__init__` calls `rebuild_alibi_tensor`
without a `device` argument, which crashes on the meta device.

The authors already defined `device` as a parameter in `rebuild_alibi_tensor` and handle
device movement in `forward()` — they just never passed it at init. The fix patches the cached
file directly:
```python
# before
self.rebuild_alibi_tensor(size=config.alibi_starting_size)

# after
self.rebuild_alibi_tensor(size=config.alibi_starting_size, device='cpu')
```

The patch is idempotent safe to run multiple times and uses `hf_constants.HF_HOME` so it
works on Kaggle, Colab, and local machines without hardcoded paths.

### 3. AutoConfig creates a type mismatch (Major one)

Using `AutoConfig.from_pretrained` loads DNABERT-2's own custom config class from the remote
code. This creates a type mismatch with the standard `transformers.BertConfig` that the rest of
the pipeline expects. Using `BertConfig.from_pretrained` directly ensures we always get the
standard class.

### 4. Flash Attention / Triton

DNABERT-2 uses a Triton-based Flash Attention kernel by default. The authors documented in their
own `configuration_bert.py` that the kernel only activates when `attention_probs_dropout_prob == 0`.
Setting it to any positive value disables it and falls back to pure PyTorch attention — no Triton
required, works on CPU and GPU.
```python
config.attention_probs_dropout_prob = 0.1
```

### 5. MLM loss was nan

When training with `task='mlm'`, the loss came out as nan. HuggingFace was trying to tie
`bert.embeddings.word_embeddings.weight` to `cls.predictions.decoder.weight`, but both weights
exist independently in the DNABERT-2 checkpoint. The tie operation was corrupting one of them.

Fix:
```python
config.tie_word_embeddings = False
```

### 6. Batch shape mismatch in multi-task regression

DeepChem's dataloader wraps each batch in an extra dimension. `y` arrives as
`array([actual_labels])` so indexing `y[0]` is needed to get the actual labels. Without this,
multi-task regression was seeing `target size (1, 2, 2)` against `input size (2, 2)` and
producing wrong gradients. This matches exactly what `HuggingFaceModel._prepare_batch` and
`Chemberta` both do.

---

## Results

### 200 samples, CPU, frozen encoder + pooler trainable (592k params)

| Metric | Score |
|--------|-------|
| Test Accuracy | 87.5% |
| ROC AUC | 0.977 |

### 60k real biological sequences, Colab T4 GPU, 10 epochs

| Metric | Score |
|--------|-------|
| Test Accuracy | 77.1% |
| ROC AUC | 0.935 |
| Non-promoter precision | 0.69 |
| Non-promoter recall | 0.99 |
| Promoter precision | 0.98 |
| Promoter recall | 0.55 |

The low promoter recall is a class imbalance issue — the dataset has significantly more
non-promoter sequences. Adding class weights to the loss is the next step.

---

My next sequence of action involves writing test cases for this model.