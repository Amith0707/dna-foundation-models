from transformers import AutoTokenizer
from deepchem.feat.huggingface_featurizer import HuggingFaceFeaturizer

class DNAFeaturizer(HuggingFaceFeaturizer):
    """Wrapper class for DNA foundation model tokenizers.

    The `DNAFeaturizer` provides a thin wrapper around Hugging Face
    DNA tokenizers (eg- DNABERT) so they can be used as DeepChem
    featurizers. This allows raw DNA sequences to be converted into
    transformer ready features compatible with DeepChem datasets.

    Example
    -------
    >>> from deepchem.feat import DNAFeaturizer
    >>> featurizer = DNAFeaturizer()
    >>> result = featurizer.featurize(["ACGTACGT"])
    """

    def __init__(
        self,
        tokenizer_name: str = "zhihan1996/DNA_bert_6",
        max_length: int = 512 #  removed that true_remote_code remember
    ):
        """Initializes a DNA tokenizer wrapper.

        Parameters
        ----------
        tokenizer_name : str, optional (default "zhihan1996/DNA_bert_6")
            Name of the Hugging Face DNA tokenizer to use.
        max_length : int, optional (default 512)
            Maximum sequence length for padding and truncation.
        """

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            model_max_length=max_length
        )

        super().__init__(tokenizer=tokenizer)
