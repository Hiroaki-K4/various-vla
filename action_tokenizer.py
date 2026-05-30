from typing import Tuple, Union

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

    def decode_model_output(
        self,
        output_ids: np.ndarray | torch.Tensor,
        action_dim: int = 7,
    ) -> np.ndarray:
        """
        Extract continuous actions from the output of VLAModel.generate()

        Parameters
        - output_ids : (B, max_new_tokens) token IDs from generate()
        - action_dim : Robot action dimensionality

        Returns
        - actions : (B, action_dim) or (action_dim,) continuous action array
        """
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().numpy()

        if output_ids.ndim == 2:
            ids = output_ids[:, -action_dim:]
        else:
            ids = output_ids[-action_dim:]

        return self.detokenize(ids)

    def get_action_token_range(self) -> Tuple[int, int]:
        """
        Return the [begin, end) range of token IDs reserved for actions
        """
        begin = self.tokenizer.vocab_size - self.n_bins
        end = self.tokenizer.vocab_size
        return begin, end

    def is_action_token(self, token_id: int) -> bool:
        begin, end = self.get_action_token_range()
        return begin <= token_id < end

    def __repr__(self) -> str:
        return (
            f"ActionTokenizer(n_bins={self.n_bins},"
            f"action_token_range={self.get_action_token_range()})"
        )

    @staticmethod
    def _make_bins(n: int) -> Tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(-1.0, 1.0, n)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return edges, centers


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    at = ActionTokenizer(tokenizer, n_bins=256)
    print(at)

    # 1. Basic round-trip
    action = np.array([0.1, -0.5, 0.9, 0.0, -1.0, 1.0, 0.3])  # (7,)
    token_ids = at.tokenize(action)
    reconstructed = at.detokenize(token_ids)
    print("original:", action)
    print("token_ids:", token_ids)
    print("reconstruct:", reconstructed)
    print("max error:", np.max(np.abs(action - reconstructed)))

    # 2. Batch input
    rng = np.random.default_rng(0)
    batch = rng.uniform(-1.0, 1.0, size=(4, 7)).astype(np.float32)
    ids = at.tokenize(batch)
    rec = at.detokenize(ids)
    print(f"\nbatch shape: {batch.shape} -> token_ids: {ids.shape} -> rec: {rec.shape}")
    print("batch max error:", np.max(np.abs(batch - rec)))

    # 3. decode_model_output
    fake_output = torch.tensor(ids)  # (4, 7)
    actions_from_model = at.decode_model_output(fake_output, action_dim=7)
    print(f"\ndecode_model_output: {actions_from_model}")
