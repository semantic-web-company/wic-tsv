from enum import Enum
from typing import Tuple, List, Union

import torch
from transformers import PreTrainedTokenizer


class TTVDatasetEncodingOptions(Enum):
    CTX__CLS_DEF_HYP = '[CLS] context [SEP] class; definition; hypernyms [SEP]'


def tokenize_and_preserve_labels(contexts: Union[List[List[str]], List[str]],
                                 labels: Union[List[List[str]], List[str]],
                                 tokenizer,
                                 subtoken_label: str = "##"):
    if isinstance(contexts[0], str):
        contexts = [x.split(' ') for x in contexts]
    tokenized_contexts = []
    tokenized_labels = []
    for sent, sent_labels in zip(contexts, labels):
        sent_tokens = []
        token_labels = []
        for word, label in zip(sent, sent_labels):
            tokens = tokenizer.tokenize(word)
            token_labels += [label]
            if subtoken_label is not None:
                token_labels += [subtoken_label] * (len(tokens) - 1)
            else:
                token_labels += [label] * (len(tokens) - 1)
            sent_tokens += tokens
        tokenized_contexts.append(sent_tokens)
        tokenized_labels.append(token_labels)
    return tokenized_contexts, tokenized_labels


class TTVDataset(torch.utils.data.Dataset):

    def __init__(self,
                 contexts: List[List[str]],
                 hypernyms: dict,
                 definitions: dict,
                 tokenizer: PreTrainedTokenizer,
                 labels: List[List[str]] = None,
                 target_classes: List[str] = None,
                 # neg_examples_per_annotation: int = 1,
                 tag2idx_null_subtoken: Tuple[dict, str, str] = None,
                 encoding_type=TTVDatasetEncodingOptions.CTX__CLS_DEF_HYP):
        """

        @param contexts: list of tokenized input strings
        @param hypernyms: dictionary of hypernyms for a given target class
        @param definitions: dictionary of definitions for a given target class
        @param tokenizer:
        @param labels: target labels to train from (e.g. B-Person, I-Person, O)
        @param target_classes: target classes to predict (e.g., Person). If labels are provided, these will be ignored
        @param tag2idx_nulllabel: a tuple of (dict{tag_str:id}, string_of_null_label, string_of_subtoken).
        If not provided, the BIO (O as null label, ## as subtoken label) is used, with {##:0, O:1, I:2, B:3}
        @param encoding_type:
        """
        self.len = len(contexts)
        self.tokenizer = tokenizer
        if labels is None and target_classes is None:
            raise RuntimeError("either target_classes or labels must be provided (XOR)")
        self.labels = None

        # is UNK relevant here?
        if tag2idx_null_subtoken is None:
            null_label = 'O'
            subtoken_label = "##"
            self.tag2idx = dict()
            self.tag2idx[subtoken_label] = 0
            self.tag2idx[null_label] = 1
            self.tag2idx['I'] = 2
            self.tag2idx['B'] = 3
            # self.tag2idx['E'] = 4
            # self.tag2idx['S'] = 5
        else:
            self.tag2idx, null_label, subtoken_label = tag2idx_null_subtoken

        tokenizer_input = []
        output_seqs = []
        for i, sent in enumerate(contexts):
            unique_sent_classes = set([x.split('-')[-1] for x in labels[i]]) if labels is not None else set(
                target_classes)
            for token_cls in unique_sent_classes:
                if encoding_type is TTVDatasetEncodingOptions.CTX__CLS_DEF_HYP:
                    tokenizer_input.append([" ".join(sent), token_cls + "; "
                                            + definitions[token_cls] + "; "
                                            + ', '.join(hypernyms[token_cls])])
                else:
                    raise NotImplementedError
                ## negative example through label switching?
                ## negative example through changing span?
                if labels is not None:
                    token_ids = [j for j, label in enumerate(labels[i]) if label.endswith("-" + token_cls)]
                    labels_seq = [label.split('-')[0] if i in token_ids else null_label for i, label in
                                  enumerate(labels[i])]
                    output_seqs.append(labels_seq)

        if labels is not None:
            _, tokenized_labels = tokenize_and_preserve_labels(contexts=contexts,
                                                               labels=labels,
                                                               tokenizer=self.tokenizer,
                                                               subtoken_label=subtoken_label)
            self.labels = torch.tensor([[self.tag2idx[tag] for tag in sent_labels] for sent_labels in tokenized_labels],
                                       dtype=torch.float)

        self.encodings = tokenizer(tokenizer_input, return_tensors='pt', truncation=True, padding=True)


    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


    def __len__(self):
        return self.len
