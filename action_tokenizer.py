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
    def __init__(self, tokenizer):
        pass