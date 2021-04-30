from collections import defaultdict
from enum import Enum

import torch
from transformers import PreTrainedTokenizer


class WiCTSVDatasetEncodingOptions(Enum):
    CTX_DEF = '[CLS] context [SEP] definition; hypernyms [SEP]'


class WiCTSVDataset(torch.utils.data.Dataset):
    def __init__(self,
                 contexts,
                 target_inds,
                 hypernyms,
                 definitions,
                 tokenizer: PreTrainedTokenizer,
                 labels=None,
                 focus_token=None,
                 encoding_type=WiCTSVDatasetEncodingOptions.CTX_DEF):
        self.len = len(contexts)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.labels = None
        self.tokenizer = tokenizer
        if focus_token is not None:
            contexts, target_inds = self.mark_target_in_context(contexts=contexts,
                                                                target_inds=target_inds,
                                                                focus_char=focus_token)

        targets = [cxt.split(' ')[tgt_ind] for cxt, tgt_ind in zip(contexts, target_inds)]
        self.tgt_start_len = []
        self.descr_start_len = []
        sense_ids_strs = []
        for cxt, tgt_ind, def_, hyps, tgt in zip(contexts, target_inds, definitions, hypernyms, targets):
            cxt_index_map, cxt_index_list = self._get_token_index_map_and_list(cxt, tokenizer)

            target_start_ind = cxt_index_map[tgt_ind][0] + len(['[CLS]'])
            target_len = len(cxt_index_map[tgt_ind])
            self.tgt_start_len.append((target_start_ind, target_len))

            sense_identifiers_str = def_ + '; ' + ', '.join(hyps)
            sense_ids_strs.append(sense_identifiers_str)
            descrs_start_ind = len(cxt_index_list) + len(['[CLS]', '[SEP]'])
            descrs_len = len(tokenizer.tokenize(sense_identifiers_str))
            self.descr_start_len.append((descrs_start_ind, descrs_len))

        if encoding_type == WiCTSVDatasetEncodingOptions.CTX_DEF:
            tokenizer_input = [[context, sense_ids] for context, sense_ids in zip(contexts, sense_ids_strs)]
        # elif
        # tokenizer_input = [[context, def_and_hyp] for context, def_and_hyp in
        #                   zip(contexts, self._concatenate_definitions_and_hypernyms(definitions,
        #                                                                             hypernyms,
        #                                                                             sep=' ',
        #                                                                             hypernym_sep=' '))]
        else:
            raise NotImplementedError

        self.encodings = tokenizer(tokenizer_input, return_tensors='pt', truncation=True, padding=True)


    @staticmethod
    def _get_token_index_map_and_list(text, tokenizer):
        """
        creates a mapping between indices of original tokens and indices of tokens after bert tokenization
        :param text: text to be tokenized
        :return: dict of the format { original_index : [bert_indices]}, list original token index for each bert token
        """
        original_tokens = text.split(' ')
        index_list = []
        index_map = defaultdict(list)

        for original_index in range(len(original_tokens)):
            bert_tokens = tokenizer.tokenize(original_tokens[original_index])
            index_list += [original_index] * len(bert_tokens)
        for bert_index, original_index in enumerate(index_list):
            index_map[original_index].append(bert_index)

        bert_tokens = tokenizer.tokenize(text)
        assert len(bert_tokens) == len(sum(index_map.values(), [])), (bert_tokens, index_map)

        return index_map, index_list


    # @staticmethod
    # def _concatenate_definitions_and_hypernyms(definitions, hypernyms, sep="; ", hypernym_sep=", "):
    #     return [sep.join([d, hypernym_sep.join(h_list)]) for d, h_list in zip(definitions, hypernyms)]


    @staticmethod
    def mark_target_in_context(contexts: list, target_inds: list, focus_char: str = "$"):
        """
        This method will mark the target word in a context with a special character before and after it,
        e.g. "This is the target in this sentence" --> "This is the $ target $ in this sentence"
        :param contexts: list of context strings
        :param target_inds: list of target indices
        :param focus_char: character which should be taken to mark the target
        :return: list of marked context strings, updated target indices
        """
        marked_contexts = []
        marked_target_inds = []
        for context, target_i in zip(contexts, target_inds):
            context_tokens = context.split()
            before_target = context_tokens[:target_i]
            after_target = context_tokens[target_i + 1:]
            marked_contexts.append(' '.join(before_target +
                                            [focus_char] +
                                            context_tokens[target_i:target_i + 1] +
                                            [focus_char] +
                                            after_target))
            marked_target_inds.append(target_i + 1)
        assert len(marked_contexts) == len(marked_target_inds)

        return marked_contexts, marked_target_inds


    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['target_start_len'] = torch.tensor(self.tgt_start_len[idx])
        item['descr_start_len'] = torch.tensor(self.descr_start_len[idx])
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


    def __len__(self):
        return self.len
