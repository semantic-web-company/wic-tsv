from enum import Enum
from typing import Tuple, List, Union

import torch
from transformers import PreTrainedTokenizer


class TTRDatasetEncodingOptions(Enum):
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


class TTRDataset(torch.utils.data.Dataset):

    def __init__(self,
                 contexts: List[List[str]],
                 hypernyms: dict,
                 definitions: dict,
                 tokenizer: PreTrainedTokenizer,
                 labels: List[List[str]] = None,
                 target_classes: List[str] = None,
                 # neg_examples_per_annotation: int = 1,
                 tag2idx_null_subtoken: Tuple[dict, str, str] = None,
                 encoding_type=TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP):
        """

        @param contexts: list of tokenized input strings
        @param hypernyms: dictionary of hypernyms for a given target class
        @param definitions: dictionary of definitions for a given target class
        @param tokenizer:
        @param labels: target labels to train from (e.g. B-Person, I-Person, O)
        @param target_classes: target classes to predict (e.g., Person). If labels are provided, these will be ignored
        @param tag2idx_nulllabel: a tuple of (dict{tag_str:id}, string_of_null_label, string_of_subtoken).
        0 is reserved for out-of-focus tokens.
        If not provided, the BIO (O as null label, ## as subtoken label) is used, with {##:1, O:2, I:3, B:4}
        @param encoding_type:
        """
        self.tokenizer = tokenizer

        if labels is None and target_classes is None:
            raise RuntimeError("either target_classes or labels must be provided (XOR)")
        self.labels = None

        # is UNK relevant here?
        if tag2idx_null_subtoken is None:
            null_label = 'O'
            subtoken_label = "##"

            self.tag2idx = dict()
            self.tag2idx[subtoken_label] = 1
            self.tag2idx[null_label] = 2
            self.tag2idx['I'] = 3
            self.tag2idx['B'] = 4
            # self.tag2idx['E'] = 4
            # self.tag2idx['S'] = 5
        else:
            self.tag2idx, null_label, subtoken_label = tag2idx_null_subtoken
        #out-of-focus token, for e.g. [CLS], or anything after the [SEP]
        out_of_focus_label = 'oof'
        self.tag2idx[out_of_focus_label] = 0

        self.definitions = definitions
        self.definitions[null_label] = ""
        self.hypernyms = hypernyms
        self.hypernyms[null_label] = [""]

        tokenizer_input = []
        output_tags = []
        for i, sent in enumerate(contexts):
            unique_sent_classes = set([x.split('-')[-1] for x in labels[i]]) if labels is not None else set(
                target_classes)
            for token_cls in unique_sent_classes:
                #should we ignore target class O?
                if token_cls != null_label or len(unique_sent_classes) == 1:
                    if encoding_type is TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP:
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
                        output_tags.append(labels_seq)

        self.encodings = tokenizer(tokenizer_input, return_tensors='pt', truncation=True, padding=True, max_length=512)
        self.len = self.encodings['input_ids'].shape[0]
        self.max_len = self.encodings['input_ids'].shape[1]

        if labels is not None:
            if encoding_type is TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP:
                input_contexts = [x[0] for x in tokenizer_input]
            else:
                raise NotImplementedError
            _, tokenized_tags = tokenize_and_preserve_labels(contexts=input_contexts,
                                                               labels=output_tags,
                                                               tokenizer=self.tokenizer,
                                                               subtoken_label=subtoken_label)
            label_encodings = [[self.tag2idx[tag] for tag in sent_tags] for sent_tags in tokenized_tags]
            self.labels = torch.tensor([l[:self.max_len] + [self.tag2idx[out_of_focus_label]] * max(0,(self.max_len-len(l)))
                                        for l in label_encodings],
                                       dtype=torch.long)



    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


    def __len__(self):
        return self.len
