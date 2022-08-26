import logging
import random
from collections import defaultdict
from enum import Enum
from typing import List, Union

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
        assert (len(tokenized_contexts) == len(tokenized_labels))
    return tokenized_contexts, tokenized_labels


class TTRDataset(torch.utils.data.Dataset):

    def __init__(self,
                 hypernyms: dict,
                 definitions: dict,
                 tokenizer: PreTrainedTokenizerFast,
                 contexts: List[List[str]]=None,
                 cls_labels: dict = None,
                 reduce_classes = False,
                 labels: List[List[str]] = None,
                 target_classes: List[str] = None,
                 neg_examples_per_annotation: int = None,
                 tag2idx: dict= None,
                 encoding_type=TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP,
                 instance_tuples = None,
                 seed = 42):
        """
        @param hypernyms: dictionary of hypernyms for a given target class (e.g., {PER : human being})
        @param definitions: dictionary of definitions for a given target class (e.g., {PER : an weird kind of ape })
        @param tokenizer: FastTokenizer to be used
        @param contexts: list of tokenized input strings (e.g. [["this", "is", "doc", "1"], ["this", "is", "doc", "2"]]
        @param cls_labels: dictionary of verbalized target labels (e.g., {PER : Person})
        @param reduce_classes: If set to true, hypernym, definition, cls_labels and target classes dicts will be reduced to only those
        classes that appear in the labels (e.g. to avoid information leakage when training on a subset of classes)
        alternatively, a list of classes can be provided (e.g., when training only on a subset of the training set,
        but all original classes should be taken into account)
        @param labels: target labels of the contexts to train from (e.g. [B-PER, I-PER, O, B-ORG])
        @param target_classes: target classes to predict (e.g., [PER, ORG]). It is expected to set this in test sets,
        if set in training sets, this is equivalent to train with (|target_classes| - 1) negative examples .
        If neither target_classes nor labels are provided, target_classes will be derived from the keys of definitions dict.
        @param neg_examples_per_annotation: number of negative examples (by target class switching) per annotation.
        Is is expected to set this in train sets, if set in test sets, the resulting metrics might be skewed. If not set
        or set to 0 for training, the model might lean towards an all-true classifier
        Only taken into account if target_classes is not set.
        @param tag2idx: dictionary of {tag_str:id} -100 is reserved for ignore tokens (CLS tokens, subtokens, etc).
        If not provided, the BIO (-100 as ignore label) is used, with {O:0, I:1, B:2}
        @param encoding_type:
        @param instance_tuples: list of the form [context_id, context, token_cls]. context_id needs to be aligned with
        the labels (if provided), context is the non-tokenized context string, and token_cls represents the target class
        of the instance (e.g., PER)
        """
        random.seed(seed)
        self.tokenizer = tokenizer

        if reduce_classes:
            if isinstance(reduce_classes, list):
                label_set = set(reduce_classes)
            elif labels is not None:
                label_set = set(["-".join(l.split('-')[-len(l.split('-')) + 1:]) for label_list in labels for l in label_list])
            else:
                raise ValueError("either provide a list of classes to reduce to or labels")
            definitions = {k:v for k, v in definitions.items() if k in label_set}
            hypernyms = {k:v for k, v in hypernyms.items() if k in label_set}
            cls_labels = {k:v for k, v in cls_labels.items() if k in label_set} if cls_labels is not None else cls_labels
            target_classes = [x for x in target_classes if x in label_set] if target_classes is not None else target_classes
            logging.log(logging.INFO, f"Classes are reduced, final number {len(definitions)}")

        self.sent_labels = labels
        if labels is None and target_classes is None:
            target_classes = sorted(list(definitions.keys()))

        # is UNK relevant here?
        if tag2idx is None:
            self.null_label = 'O'

            self.tag2idx = dict()
            self.tag2idx[self.null_label] = 0
            self.tag2idx['I'] = 1
            self.tag2idx['B'] = 2
            # self.tag2idx['E'] = 3
            # self.tag2idx['S'] = 4
        else:
            self.tag2idx = tag2idx
        # tokens to be ignored, for e.g. [CLS], or anything after (including) the [SEP], sub-word tokens
        # -100 is ignored by the loss from loss computation
        self.ignore_label = 'ignore'
        self.tag2idx[self.ignore_label] = -100

        self.definitions = definitions
        self.definitions[self.null_label] = ""
        self.hypernyms = hypernyms
        self.hypernyms[self.null_label] = [""]
        self.cls_labels = cls_labels

        self.encoding_type = encoding_type

        self.instance_tuples = instance_tuples
        self.tgt_cls_instance_dict = None

        if self.instance_tuples is None:
            if contexts is None:
                raise ValueError("If no instance_tuples are provided, contexts need to be provided")
            self.instance_tuples = []
            self.tgt_cls_instance_dict = defaultdict(list)
            for i, sent in enumerate(contexts):
                unique_sent_classes = set(target_classes) if target_classes is not None \
                else set(["-".join(x.split('-')[-len(x.split('-')) + 1 :]) for x in labels[i]])
                if target_classes is None and neg_examples_per_annotation is not None:
                    other_senses = sorted([x for x in self.definitions.keys() if x not in unique_sent_classes
                                           and x != self.null_label])
                    unique_sent_classes.update(random.choices(other_senses, k=neg_examples_per_annotation, ))

                for token_cls in unique_sent_classes:
                    #should we ignore target class O?
                    if token_cls != self.null_label: # or len(unique_sent_classes) == 1:
                        instance_tuple = [i, " ".join(sent), token_cls]
                        self.instance_tuples.append(instance_tuple)
                        self.tgt_cls_instance_dict[token_cls].append(len(self.instance_tuples) - 1)

        if self.tgt_cls_instance_dict is None:
            self.tgt_cls_instance_dict = defaultdict(list)
            for i, (_, _, target_cls) in enumerate(self.instance_tuples):
                self.tgt_cls_instance_dict[target_cls].append(i)


        self.len = len(self.instance_tuples)
        self.max_len = 512

    def get_sub_dataset(self, target_cls=None, context_id=None):
        filtered_instances = self.instance_tuples
        if target_cls:
            filtered_instances = [x for i, x in enumerate(filtered_instances)
                                  if i in self.tgt_cls_instance_dict[target_cls]]
        if context_id:
            filtered_instances = [x for x in enumerate(filtered_instances) if x[0] == context_id]
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
            len_attention_first_seq = (e["token_type_ids"].squeeze()[:len_total_input] == 0).sum() # len of everything before (including) the first [SEP]
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
        #padding should be handeled in the data loader
        encodings = self.tokenizer(tokenizer_input,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=512,
                                   return_offsets_mapping=True
                                   )
        item = {key: val[0] for key, val in encodings.items()}

        if self.sent_labels is not None:
            token_ids = [j for j, label in enumerate(self.sent_labels[sent_i]) if label.endswith("-" + token_cls)]
            labels_seq = [label.split('-')[0] if i in token_ids else self.null_label for i, label in
                          enumerate(self.sent_labels[sent_i])]
            _, tokenized_tags = tokenize_and_preserve_labels(contexts=[context],
                                                             labels=[labels_seq],
                                                             tokenizer=self.tokenizer,
                                                             subtoken_label=self.ignore_label)
            label_encodings = [[self.tag2idx[tag] for tag in sent_tags] for sent_tags in tokenized_tags]
            labels = torch.tensor([[self.tag2idx[self.ignore_label]] # [CLS]
                                   + l[:self.get_len_tokenized_context(encodings=encodings)[0]] +  # labels for context
                                   [self.tag2idx[self.ignore_label]] * (encodings.data["token_type_ids"].squeeze() == 1).sum()
                                   # ignore labels for sense descriptors
                                   for l in label_encodings],
                                  dtype=torch.long)
            item['labels'] = labels[0]
        return item


    def __len__(self):
        return self.len



class TTRSepDataset(TTRDataset):
    def __getitem__(self, idx):
        sent_i, context, token_cls = self.instance_tuples[idx]

        if self.encoding_type is TTRDatasetEncodingOptions.CTX__CLS_DEF_HYP:
            definition = self.definitions[token_cls]
            hypernyms = ', '.join(self.hypernyms[token_cls])
            token_cls_name = token_cls
            if self.cls_labels is not None and token_cls in self.cls_labels.keys():
                token_cls_name = self.cls_labels[token_cls]
            _1_tokenizer_input = [context]
            _2_tokenizer_input = [["", token_cls_name + "; " + definition + "; " + hypernyms]]
        else:
            raise NotImplementedError
        #todo truncation of first or second sequence?
        _1_encodings = self.tokenizer(_1_tokenizer_input,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=512,
                                   return_offsets_mapping=True)
        _2_encodings = self.tokenizer(_2_tokenizer_input,
                                      return_tensors='pt',
                                      truncation=True,
                                      max_length=512,
                                      return_offsets_mapping=True)
        item = {"_1_" + key: val[0] for key, val in _1_encodings.items()}
        item.update({"_2_" + key: val[0] for key, val in _2_encodings.items()})

        if self.sent_labels is not None:
            token_ids = [j for j, label in enumerate(self.sent_labels[sent_i]) if label.endswith("-" + token_cls)]
            labels_seq = [label.split('-')[0] if i in token_ids else self.null_label for i, label in
                          enumerate(self.sent_labels[sent_i])]
            _, tokenized_tags = tokenize_and_preserve_labels(contexts=[context],
                                                             labels=[labels_seq],
                                                             tokenizer=self.tokenizer,
                                                             subtoken_label=self.ignore_label)
            label_encodings = [[self.tag2idx[tag] for tag in sent_tags] for sent_tags in tokenized_tags]
            labels = torch.tensor([[self.tag2idx[self.ignore_label]] # [CLS]
                                   + l[:self.get_len_tokenized_context(encodings=_1_encodings)[0]]           # labels for context
                                   # [self.tag2idx[self.ignore_label]] *
                                   # max(0,(self.max_len - self.get_len_tokenized_context(encodings=_1_encodings)[0] -1)) # oof labels for sense descriptors and padding
                                   for l in label_encodings],
                                  dtype=torch.long)
            item['labels'] = labels[0]
        return item



