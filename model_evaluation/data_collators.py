from dataclasses import dataclass
from typing import Optional, Union, List, Dict

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorForSequenceClassificationWithAdditionalItemData:
    """

    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True #todo ideally, this would be a union of bool, str, PaddingStrategy, however,
                         # this would mean we have to align the padding strategy, currently True corresponds to longest
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        feature_names = features[0].keys()
        features_to_be_padded = [{k:v[0] for k, v in feature.items() if k != "labels"} for feature in features]
        return_attention_masks = bool("attention_mask" in features_to_be_padded[0])
        batch = self.tokenizer.pad(
            features_to_be_padded,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="np",
            return_attention_mask=return_attention_masks
        )

        target_shape = batch.data['input_ids'].shape
        def pad_longest(input_array: np.array, target_len):
            output = []
            for row in input_array:
                output.append(np.concatenate([row, np.zeros(target_len - len(row))], axis=0))
            return np.array(output)
        for key, val in batch.items():
            if val.shape != target_shape:
                batch[key] = pad_longest(val, target_shape[1])

        if "labels" in feature_names or "labels" in feature_names:
            batch["labels"] = [x["labels"].numpy() for x in features]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items() }

        return batch
