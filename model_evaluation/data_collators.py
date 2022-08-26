from dataclasses import dataclass
from typing import Union, Optional

import torch
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy


@dataclass
class DataCollatorForTokenClassificationWithOffsetMapping:
    """
    Copied and adapted from transformers.data.data_collator.DataCollatorForTokenClassification
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name].tolist() for feature in features] if label_name in features[0].keys() else None
        offset_mappings = [feature['offset_mapping'].tolist() for feature in features] if 'offset_mapping' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None and offset_mappings is None else "np",
        )

        if labels is None and offset_mappings is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if labels is not None:
                batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label))
                                   for label in labels]
            if offset_mappings is not None:
                batch["offset_mapping"] = [offset_mapping + [[0, 0]] * (sequence_length - len(offset_mapping))
                                           for offset_mapping in offset_mappings]
        else:
            if labels is not None:
                batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label
                                   for label in labels]
            if offset_mappings is not None:
                batch["offset_mapping"] = [[[0, 0]] * (sequence_length - len(offset_mapping)) + offset_mapping
                                           for offset_mapping in offset_mappings]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items() if k != "offset_mapping"}
        return batch