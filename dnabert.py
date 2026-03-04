import gc
import glob
import logging
import os
from typing import Any, Tuple

import numpy as np
from deepchem.models.torch_models.hf_models import HuggingFaceModel

logger = logging.getLogger(__name__)

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

try:
    from huggingface_hub import constants as hf_constants
    has_huggingface_hub = True
except ImportError:
    has_huggingface_hub = False

try:
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
    )
    has_transformers = True
except ImportError:
    has_transformers = False

def patch_dnabert2_cache(
    model_name: str = "zhihan1996/DNABERT-2-117M"
) -> None:
    """Apply minimal targeted patches to DNABERT-2's cached custom files.

    DNABERT-2 is shipped with two compatibility issues on modern environments:

    1. **ALiBi meta-device conflict**: ``BertEncoder.__init__`` calls
       ``rebuild_alibi_tensor`` with no ``device`` argument, causing a
       ``RuntimeError`` when HuggingFace's lazy loader initialises tensors
       on the ``meta`` device.The fix passes ``device='cpu'`` -a parameter 
       the authors already defined so the tensor is built on CPU and moved 
       to the correctdevice during the first forward pass via the authors' 
       own device catch-up logic in ``BertEncoder.forward``.

    2. **Flash-Attention / Triton**: controlled externally via
       ``config.attention_probs_dropout_prob > 0`` so no file-level patch is needed.
       
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Used only for error messages.
    """
    if not has_huggingface_hub:
        raise ImportError("huggingface_hub is required.")
    
    modules_cache = os.path.join(
        hf_constants.HF_HOME, "modules", "transformers_modules"
    )
    pattern = os.path.join(modules_cache, "**", "bert_layers.py")
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(
            f"bert_layers.py not found in HuggingFace modules cache at "
            f"'{modules_cache}'. Ensure the tokenizer has been downloaded "
            f"first via AutoTokenizer.from_pretrained('{model_name}', "
            f"trust_remote_code=True)."
        )

    old = "self.rebuild_alibi_tensor(size=config.alibi_starting_size)"
    new = (
        "self.rebuild_alibi_tensor("
        "size=config.alibi_starting_size, device='cpu')" # pulling up the AliBi fix in here by adding device as per authors code
    )

    # Confirimng whether the patch has worked or not in this loop 
    # meaning checking if the patch has been applied onto the new and old does not exist
    # not to be removed as glob might return multiple patches as user may download the model 
    # and may happen that each having different commit hashes and HF loads unpatched one

    for path in matches:
        with open(path, "r") as file:
            src = file.read()
        if old in src:
            src = src.replace(old, new)
            with open(path, "w") as fh:
                fh.write(src)
            logger.debug("DNABERT-2 ALiBi patch applied: %s", path)
        else:
            logger.debug("DNABERT-2 ALiBi patch already applied: %s", path)


class DNABERT2(HuggingFaceModel):
    """DNABERT-2 Model for DNA sequence analysis.

    DNABERT-2 is a foundation model for DNA sequences based on the
    MosaicBERT architecture with Byte-Pair Encoding (BPE) tokenization.
    It replaces the k-mer tokenization used in the original DNABERT with
    a data-driven BPE vocabulary that generalises across species and
    sequence types.

    The model supports five tasks:

    * ``mlm`` — masked language modelling (pre-training)
    * ``classification`` — binary or multi-class sequence classification
    * ``regression`` — single-target scalar regression
    * ``mtr`` — multi-task regression
    * ``feature_extractor`` — returns CLS-token embeddings

    Parameters
    ----------
    task : str
        Learning task. One of ``'mlm'``, ``'classification'``,
        ``'regression'``, ``'mtr'``, or ``'feature_extractor'``.
    model_name : str, optional
        HuggingFace model identifier.
        Defaults to ``'zhihan1996/DNABERT-2-117M'``.
    n_tasks : int, optional
        Number of prediction targets. Used for ``'classification'``,
        ``'regression'``, and ``'mtr'``. Defaults to ``1``.
    attention_probs_dropout_prob : float, optional
        Dropout probability for attention layers. Any value ``> 0`` 
        uses a pure-PyTorch attention implementation instead of the 
        Triton Flash-Attention kernel, ensuring cross-platform stability. 
        Defaults to ``0.1``.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~deepchem.models.torch_models.hf_models.HuggingFaceModel`.

    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.dnabert import DNABERT2

    >>> # Classification
    >>> sequences = ["ATCGATCG", "GCTAGCTA", "TTAACCGG"]
    >>> labels = [0, 1, 0]
    >>> dataset = dc.data.NumpyDataset(X=sequences, y=labels)
    >>> model = DNABERT2(task='classification', model_dir='/tmp/dnabert2')
    >>> loss = model.fit(dataset, nb_epoch=1)

    >>> # Feature extraction
    >>> model = DNABERT2(task='feature_extractor', model_dir='/tmp/dnabert2')
    >>> embeddings = model.predict(dataset)
    >>> embeddings.shape
    (3, 768)

    References
    ----------
    .. Zhou, Z., Ji, Y., Li, W., Dutta, P., Davuluri, R., & Liu, H.
    (2023). DNABERT-2: Efficient Foundation Model and Benchmark For
    Multi-Species Genome. arXiv:2306.15006.
    """

    def __init__(
        self,
        task: str,
        model_name: str = "zhihan1996/DNABERT-2-117M",
        n_tasks: int = 1,
        attention_probs_dropout_prob: float = 0.1,
        **kwargs,
    ):
        if not has_torch:
            raise ImportError(
                "PyTorch is required. Install: pip install torch"
            )
        if not has_transformers:
            raise ImportError(
                "transformers is required. Install: pip install transformers>=4.29,<5" 
            )
        if not has_huggingface_hub:
            raise ImportError(
                "huggingface_hub is required. Install: pip install huggingface_hub"
            )

        supported_tasks = [
            "mlm", "classification", "regression", "mtr","feature_extractor"
        ]
        if task not in supported_tasks:
            raise ValueError(
                f"Unsupported task '{task}'. "
                f"Choose one of: {supported_tasks}"
            )

        self.n_tasks = n_tasks
        self.model_name = model_name

        # Pull bert_layers.py into the HF modules cache before patching.
        # AutoModel is the only call that reliably triggers the download of
        # bert_layers.py on all platforms (Kaggle, Colab, local).
        # The model init may fail due to missing pad_token_id in the raw
        # config — that's expected and safe to swallow since I only need
        # the file on disk.
        # so in simple words forcing bert_layers.py to be downloaded on HF cache so that 
        # in a fresh env like kaggle or colab my glob module can fetch the file path to prevent the mentioned challenegs faced so far
        try:
            temp= AutoModel.from_pretrained(model_name, trust_remote_code=True)
            del temp
        except Exception:
            pass
        gc.collect() # adding to save my disk memory

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        config = BertConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        patch_dnabert2_cache(model_name)

        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        config.pad_token_id = tokenizer.pad_token_id
        config.is_decoder = False

        if task == "classification":
            config.num_labels = 2 if n_tasks == 1 else n_tasks
            config.problem_type = (
                "single_label_classification" # Cross Entropy Loss
                if n_tasks == 1
                else "multi_label_classification" # BCEWithLogitsLoss
            )
        elif task in ("regression", "mtr"):
            config.num_labels = n_tasks
            config.problem_type = "regression" # MSE Loss

        if task == "mlm":
            config.tie_word_embeddings = False   
            model = AutoModelForMaskedLM.from_pretrained(
                model_name, config=config, trust_remote_code=True
            )
        elif task == "feature_extractor":
            model = AutoModel.from_pretrained(
                model_name, config=config, trust_remote_code=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, config=config, trust_remote_code=True
            )

        super(DNABERT2, self).__init__(
            model=model,
            task=task,
            tokenizer=tokenizer,
            **kwargs,
        )

    def predict(self, dataset, transformers=[],**kwargs):
        """Run inference.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset whose ``X`` contains DNA sequences as plain strings.

        Returns
        -------
        np.ndarray
            * ``classification`` — ``(N, num_labels)`` logits
            * ``regression`` / ``mtr`` — ``(N, n_tasks)`` values
            * ``mlm`` — ``(N, seq_len, vocab_size)`` logits
            * ``feature_extractor`` — ``(N, 768)`` CLS-token embeddings
        """
        if self.task == "feature_extractor":
            return self._predict_embeddings(dataset)
        return super(DNABERT2, self).predict(dataset,transformers=transformers, **kwargs)

    def fit(self, dataset, nb_epoch: int = 1, **kwargs):
        """Train the model.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset whose ``X`` contains DNA sequences as plain strings.
        nb_epoch : int, optional
            Number of training epochs. Defaults to ``1``.

        Returns
        -------
        float
            Mean training loss over the last epoch.
        """
        if self.task == "feature_extractor":
            raise ValueError(
                """fit() is not supported for task='feature_extractor' Use task='mlm' for pre-training 
                or 'classification, regression for fine-tuning."""
            )
        return super(DNABERT2, self).fit(dataset, nb_epoch=nb_epoch, **kwargs)

    def _predict_embeddings(self, dataset) -> np.ndarray:
        """Return CLS-token embeddings for every sequence in dataset.

        Parameters
        ----------
        dataset : dc.data.Dataset
            Dataset whose ``X`` contains DNA sequences as plain strings.

        Returns
        -------
        np.ndarray, shape (N, hidden_size)
            One CLS-token vector per input sequence.
        """
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for seq in dataset.X:
                tokens = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                cls_vec = outputs[0][:, 0, :]
                embeddings.append(cls_vec.cpu().numpy())

        return np.vstack(embeddings)

    def _prepare_batch(self, batch: Tuple[Any, Any, Any]):
        """Prepare a batch for the model.

        Handles:

        * DNA sequences stored as raw strings in ``X``
        * Correct label dtype per task — ``long`` for single-label
          classification, ``float`` for everything else
        * MLM dynamic masking via the tokenizer's data collator

        Parameters
        ----------
        batch : tuple
            ``(X, y, w)`` triple from a DeepChem DataLoader.

        Returns
        -------
        tuple
            ``(inputs_dict, y_tensor, w)`` ready for ``model.forward``.
        """
        X, y, w = batch
        sequences = np.array(X[0]).tolist()

        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if self.task == "mlm":
            input_ids, labels = self.data_collator.torch_mask_tokens(
                tokens["input_ids"]
            )
            inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": tokens["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }
            return inputs, None, w

        inputs = {k: v.to(self.device) for k, v in tokens.items()}

        if y is not None:
            y_tensor = torch.from_numpy(np.asarray(y[0]))
            if self.task == "classification" and self.n_tasks == 1:
                y_tensor = y_tensor.view(-1).long().to(self.device)
            else:
                y_tensor = y_tensor.float().to(self.device)
            inputs["labels"] = y_tensor
        else:
            y_tensor = None

        return inputs, y_tensor, w