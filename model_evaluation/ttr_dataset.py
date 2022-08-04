import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Union

import torch
from transformers import PreTrainedTokenizerFast


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
                 hypernyms: dict,
                 definitions: dict,
                 tokenizer: PreTrainedTokenizerFast,
                 contexts: List[List[str]]=None,
                 cls_labels: dict = None,
                 labels: List[List[str]] = None,
                 target_classes: List[str] = None,
                 neg_examples_per_annotation: int = None,
                 tag2idx_null_subtoken: Tuple[dict, str, str] = None,
                 encoding_type=TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP,
                 instance_tuples = None):
        """

        @param contexts: list of tokenized input strings
        @param hypernyms: dictionary of hypernyms for a given target class
        @param definitions: dictionary of definitions for a given target class
        @param cls_labels: dictionary of verabilzed target labels (e.g., {PER : Person}
        @param tokenizer:
        @param labels: target labels of the contexts to train from (e.g. [B-PER, I-PER, O, B-ORG])
        @param target_classes: target classes to predict (e.g., PER). It is expected to set this in test sets, if set
        in training sets, this is equivalent to train with negative examples. If neither target_classes nor labels are provided,
        target_classes will be derived from the keys of definitions dict.
        @param tag2idx_null_subtoken: a tuple of (dict{tag_str:id}, string_of_null_label, string_of_subtoken).
        0 is reserved for out-of-focus tokens.
        If not provided, the BIO (O as null label, ## as subtoken label) is used, with {##:1, O:2, I:3, B:4}
        @param encoding_type:
        @param instance_tuples: list of the form [context_id, context, token_cls]. context_id needs to be aligned with
        the labels (if provided), context is the non-tokenized context string, and token_cls represents the target class
        of the instance (e.g., PER)
        """
        self.tokenizer = tokenizer

        self.sent_labels = labels
        if labels is None and target_classes is None:
            target_classes = list(definitions.keys())

        # is UNK relevant here?
        if tag2idx_null_subtoken is None:
            self.null_label = 'O'
            self.subtoken_label = "##"

            self.tag2idx = dict()
            self.tag2idx[self.subtoken_label] = 1
            self.tag2idx[self.null_label] = 2
            self.tag2idx['I'] = 3
            self.tag2idx['B'] = 4
            # self.tag2idx['E'] = 5
            # self.tag2idx['S'] = 6
        else:
            self.tag2idx, self.null_label, self.subtoken_label = tag2idx_null_subtoken
        #out-of-focus token, for e.g. [CLS], or anything after (including) the [SEP]
        self.out_of_focus_label = 'oof'
        self.tag2idx[self.out_of_focus_label] = 0

        #todo remove unnecessary sense descriptors? (such not present in target classes if provided?)
        self.definitions = definitions
        self.definitions[self.null_label] = ""
        self.hypernyms = hypernyms
        self.hypernyms[self.null_label] = [""]
        self.cls_labels = cls_labels

        self.encoding_type = encoding_type

        self.instance_tuples = instance_tuples
        self.instance_dict = None

        if self.instance_tuples is None:
            if contexts is None:
                raise RuntimeError("If no instance_tuples are provided, contexts need to be provided")
            self.instance_tuples = []
            self.instance_dict = defaultdict(list)
            for i, sent in enumerate(contexts):
                unique_sent_classes = set(target_classes) if target_classes is not None \
                    else set([x.split('-')[-1] for x in labels[i]])
                if target_classes is None and neg_examples_per_annotation is not None:
                    other_senses = [x for x in self.definitions.keys() if x not in unique_sent_classes and x != self.null_label]
                    unique_sent_classes.update(random.choices(other_senses, k=neg_examples_per_annotation))

                for token_cls in unique_sent_classes:
                    #should we ignore target class O?
                    if token_cls != self.null_label or len(unique_sent_classes) == 1:
                        instance_tuple = [i, " ".join(sent), token_cls]
                        self.instance_tuples.append(instance_tuple)
                        self.instance_dict[token_cls].append(len(self.instance_tuples) - 1)

        if self.instance_dict is None:
            self.instance_dict = defaultdict(list)
            for i, (_, _, target_cls) in enumerate(self.instance_tuples):
                self.instance_dict[target_cls].append(i)


        self.len = len(self.instance_tuples)
        self.max_len = 512

    def get_sub_dataset(self, target_cls):
        filtered_instances = [x for i, x in enumerate(self.instance_tuples) if i in self.instance_dict[target_cls]]
        return TTRDataset(hypernyms=self.hypernyms,
                          definitions=self.definitions,
                          tokenizer=self.tokenizer,
                          instance_tuples=filtered_instances,
                          labels=self.sent_labels)

    def get_offsets(self, idx=None):
        indices = [idx] if idx is not None else list(range(self.len))
        offsets = []
        for i in indices:
            offsets.append(self[i]['offset_mapping'])
        return offsets

    def get_len_tokenized_context(self, idx=None, encodings=None):
        indices = [idx] if idx is not None else list(range(self.len))
        if encodings is None:
            encodings = [self[i] for i in indices]
        else:
            encodings = [encodings.data]

        lens_tokenized_context = []
        for e in encodings:
            len_total_input = int((e["attention_mask"] == 1).sum()) # len of everything before the padding
            len_attention_first_seq = (e["token_type_ids"][ :len_total_input] == 0).sum() # len of everything before (including) the first [SEP]
            lens_tokenized_context.append(int(len_attention_first_seq - 2)) # minus [CLS] and [SEP]
        return lens_tokenized_context

    def get_contexts(self, idx=None):
        indices = [idx] if idx is not None else list(range(self.len))
        contexts = []
        for i in indices:
            _, context, _ = self.instance_tuples[i]
            contexts.append(context)
        return contexts


    def __getitem__(self, idx):
        sent_i, context, token_cls = self.instance_tuples[idx]

        if self.encoding_type is TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP:
            definition = self.definitions[token_cls]
            hypernyms = ', '.join(self.hypernyms[token_cls])
            token_cls_name = token_cls
            if self.cls_labels is not None and token_cls in self.cls_labels.keys():
                token_cls_name = self.cls_labels[token_cls]
            tokenizer_input = [[context, token_cls_name + "; " + definition + "; " + hypernyms]]
        else:
            raise NotImplementedError
        #todo truncation of first or second sequence?
        encodings = self.tokenizer(tokenizer_input,
                                   return_tensors='pt',
                                   truncation=True,
                                   padding="max_length",
                                   max_length=512,
                                   return_offsets_mapping=True)
        item = {key: val[0] for key, val in encodings.items()}

        if self.sent_labels is not None:
            token_ids = [j for j, label in enumerate(self.sent_labels[sent_i]) if label.endswith("-" + token_cls)]
            labels_seq = [label.split('-')[0] if i in token_ids else self.null_label for i, label in
                          enumerate(self.sent_labels[sent_i])]
            _, tokenized_tags = tokenize_and_preserve_labels(contexts=[context],
                                                             labels=[labels_seq],
                                                             tokenizer=self.tokenizer,
                                                             subtoken_label=self.subtoken_label)
            label_encodings = [[self.tag2idx[tag] for tag in sent_tags] for sent_tags in tokenized_tags]
            labels = torch.tensor([[self.tag2idx[self.out_of_focus_label]] # [CLS]
                                   + l[:self.get_len_tokenized_context(encodings=encodings)[0]] +           # labels for context
                                   [self.tag2idx[self.out_of_focus_label]] *
                                   max(0,(self.max_len - self.get_len_tokenized_context(encodings=encodings)[0] -1)) # oof labels for sense descriptors and padding
                                   for l in label_encodings],
                                  dtype=torch.long)
#             item['labels'] = labels[0]
            item['label_ids'] = labels[0]
        return item


    def __len__(self):
        return self.len
