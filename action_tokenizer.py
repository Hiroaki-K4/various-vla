from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase


class ActionTokenizer:
    """
    1. Tokenize (continuous -> token ID)
        - Divide each dimention into n_bins equal-width bins over [min_action, max_action]
        - Obtain 1-indexed bin indices via np.digitize
        - Map to the last n_bins tokens of the vocabulary:
            token_id = vocab_size - bin_index

    2. Detokenize (token ID -> continuous):
        - Recover bin index: bin_index = vocab_size - token_id
        - Convert to 0-indexed, clip to valid range, then look up bin_centers
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        n_bins: int = 256,
    ) -> None:
        """
        Parameters
        - tokenizer : HuggingFace tokenizer
        - n_bins : Number of discretization bins
        """
        self.tokenizer = tokenizer
        self.n_bins = n_bins

        self._bins, self._bin_centers = self._make_bins(n_bins)
        self.action_token_begin_idx: int = int(tokenizer.vocab_size - (n_bins + 1))

    def tokenize(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Parameters
        - action : (action_dim,) or (B, action_dim), values in [-1, 1]

        Returns
        - token_ids: int64 array of the same shape
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        action = np.clip(action, -1.0, 1.0)
        discretized = np.digitize(action, self._bins)  # 1-indexed
        return (self.tokenizer.vocab_size - discretized).astype(np.int64)

    def detokenize(self, token_ids: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert token IDs back to continuous action values.
        Accepts the raw output of model.generate() directly.

        Parameters
        - token_ids : (action_dim,) or (B, action_dim)

        Returns
        - actions : float32 array of the same shape, values in [-1, 1]
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        # Recover 1-indexed bin index, then shift to 0-indexed
        bin_idx = self.tokenizer.vocab_size - token_ids
        bin_idx = np.clip(bin_idx - 1, 0, len(self._bin_centers) - 1)
        return self._bin_centers[bin_idx].astype(np.float32)

    @staticmethod
    def _make_bins(n: int) -> Tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(-1.0, 1.0, n)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return edges, centers
