"""High-performance Code Tokenizer with Byte-Level BPE, FIM support, and whitespace preservation."""

from typing import Dict, List, Optional, Tuple, Union
import json
import os
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import ByteLevel as ByteLevelProcessor

from tokenizer.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_MIDDLE,
    SPECIAL_TOKENS_LIST,
)


class CodeTokenizer:
    """Production-grade Code Tokenizer designed for 50B coding language models."""

    def __init__(self, tokenizer: Optional[Tokenizer] = None, vocab_size: int = 65536):
        self.vocab_size = vocab_size
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._build_default_tokenizer()

        # Cache special token IDs
        self.bos_token_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(UNK_TOKEN)
        self.fim_prefix_id = self.tokenizer.token_to_id(FIM_PREFIX)
        self.fim_suffix_id = self.tokenizer.token_to_id(FIM_SUFFIX)
        self.fim_middle_id = self.tokenizer.token_to_id(FIM_MIDDLE)

    def _build_default_tokenizer(self) -> Tokenizer:
        """Constructs an initial Byte-Level BPE tokenizer with code regex splitting."""
        # Use Byte-Level BPE model
        tokenizer = Tokenizer(models.BPE())
        
        # ByteLevel pre-tokenizer preserves all indentation, newlines, and spaces
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

        # Add initial special tokens
        tokenizer.add_special_tokens(SPECIAL_TOKENS_LIST)

        # Use ByteLevel alphabet to guarantee 100% round-trip fidelity for all 256 bytes,
        # spaces, indentation, tabs, and newlines.
        alphabet = pre_tokenizers.ByteLevel.alphabet()
        initial_vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS_LIST)}
        offset = len(initial_vocab)
        for char in alphabet:
            if char not in initial_vocab:
                initial_vocab[char] = offset
                offset += 1

        tokenizer.model = models.BPE(vocab=initial_vocab, merges=[])
        return tokenizer

    def train_from_iterator(self, iterator, vocab_size: Optional[int] = None) -> None:
        """Trains the BPE tokenizer on a stream of source code files."""
        target_vocab_size = vocab_size or self.vocab_size
        trainer = trainers.BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS_LIST,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._refresh_token_ids()

    def _refresh_token_ids(self) -> None:
        self.bos_token_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_token_id = self.tokenizer.token_to_id(PAD_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(UNK_TOKEN)
        self.fim_prefix_id = self.tokenizer.token_to_id(FIM_PREFIX)
        self.fim_suffix_id = self.tokenizer.token_to_id(FIM_SUFFIX)
        self.fim_middle_id = self.tokenizer.token_to_id(FIM_MIDDLE)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes source code into token IDs preserving exact whitespace."""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back into source code text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def format_fim(
        self,
        prefix: str,
        suffix: str,
        middle: Optional[str] = None,
        mode: str = "PSM",
    ) -> Union[str, Tuple[List[int], Optional[List[int]]]]:
        """Formats code for Fill-In-The-Middle (FIM) training and inference.
        
        Supported Modes:
        - PSM: <fim_prefix> PRE <fim_suffix> SUF <fim_middle> MID
        - SPM: <fim_suffix> SUF <fim_prefix> PRE <fim_middle> MID
        """
        if mode == "PSM":
            fim_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        elif mode == "SPM":
            fim_text = f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"
        else:
            raise ValueError(f"Unsupported FIM mode: {mode}. Must be 'PSM' or 'SPM'.")

        if middle is not None:
            fim_text += f"{middle}{EOS_TOKEN}"

        return fim_text

    def save(self, filepath: str) -> None:
        """Saves tokenizer state to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.tokenizer.save(filepath)

    @classmethod
    def from_file(cls, filepath: str) -> "CodeTokenizer":
        """Loads tokenizer state from disk."""
        tokenizer = Tokenizer.from_file(filepath)
        return cls(tokenizer=tokenizer)


def create_default_code_tokenizer(output_path: Optional[str] = None) -> CodeTokenizer:
    """Helper to instantiate and optionally persist the default code tokenizer."""
    code_tok = CodeTokenizer()
    if output_path:
        code_tok.save(output_path)
    return code_tok
