"""Fill-In-the-Middle (FIM) transformation pipeline supporting PSM and SPM objectives."""

import random
from typing import List, Tuple
from tokenizer.special_tokens import FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, EOT_TOKEN


class FIMTransform:
    """Dynamically applies Fill-in-the-Middle transformations to code token sequences.
    
    Paper reference: "Efficient Training of Language Models to Fill in the Middle" (Bavarian et al., OpenAI)
    """

    def __init__(
        self,
        fim_rate: float = 0.5,
        spm_rate: float = 0.5,
        prefix_token_id: int = 5,
        suffix_token_id: int = 6,
        middle_token_id: int = 7,
        eot_token_id: int = 4,
    ):
        self.fim_rate = fim_rate
        self.spm_rate = spm_rate
        self.prefix_token_id = prefix_token_id
        self.suffix_token_id = suffix_token_id
        self.middle_token_id = middle_token_id
        self.eot_token_id = eot_token_id

    def transform_tokens(self, tokens: List[int], rng: random.Random = random) -> Tuple[List[int], bool]:
        """Splits token sequence into prefix, middle, suffix and reconstructs in FIM format.
        
        Returns:
            (transformed_tokens, was_fim_applied)
        """
        if len(tokens) < 16 or rng.random() > self.fim_rate:
            return tokens, False

        # Randomly choose split boundaries
        split_point_1 = rng.randint(1, len(tokens) - 2)
        split_point_2 = rng.randint(1, len(tokens) - 2)

        start = min(split_point_1, split_point_2)
        end = max(split_point_1, split_point_2)

        prefix = tokens[:start]
        middle = tokens[start:end]
        suffix = tokens[end:]

        # Select mode: PSM vs SPM
        if rng.random() < self.spm_rate:
            # SPM: <fim_suffix> suffix <fim_prefix> prefix <fim_middle> middle <eot>
            transformed = (
                [self.suffix_token_id]
                + suffix
                + [self.prefix_token_id]
                + prefix
                + [self.middle_token_id]
                + middle
                + [self.eot_token_id]
            )
        else:
            # PSM: <fim_prefix> prefix <fim_suffix> suffix <fim_middle> middle <eot>
            transformed = (
                [self.prefix_token_id]
                + prefix
                + [self.suffix_token_id]
                + suffix
                + [self.middle_token_id]
                + middle
                + [self.eot_token_id]
            )

        return transformed, True
