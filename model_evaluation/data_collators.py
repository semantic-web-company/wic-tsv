from dataclasses import dataclass
from typing import Optional, Union, List, Dict

import torch
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase


@dataclass
class DataCollatorForSequenceClassificationWithAdditionalItemData:
    """

    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        feature_names_to_be_padded = ["input_ids", "token_type_ids", "attention_mask"]
        features_to_be_padded = [{k:v[0] for k, v in feature.items() if k in feature_names_to_be_padded} for feature in features]
        feature_names_in_items = features[0].keys()
        return_attention_masks = bool("attention_mask" in feature_names_in_items)
        batch = self.tokenizer.pad(
            features_to_be_padded,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            return_attention_mask=return_attention_masks
        )

        for item_data in feature_names_in_items:
            if not item_data in feature_names_to_be_padded:
                batch[item_data] = [x[item_data].numpy() for x in features]
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items() }

        return batch
