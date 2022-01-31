from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import torch
from transformers import PreTrainedTokenizer


class WiCTSVDatasetEncodingOptions(Enum):
    CTX_DEF = '[CLS] context [SEP] definition; hypernyms [SEP]'


def char_offset2token_offset(
        cxt: str,
        si: int,
        ei: int) -> Tuple[List[str], int, int]:
    cxt_ = cxt[:si] + ' MATCH ' + cxt[ei:]
    target_toks = cxt[si:ei].split()
    tokens = cxt_.split()
    for i, w in enumerate(tokens):
        if w == 'MATCH':
            tgt_si = i
            break
    else:
        raise IndexError(f'No MATCH found in {tokens}')
    tgt_ei = tgt_si + len(target_toks)
    tokens[tgt_si:tgt_si + 1] = target_toks
    return tokens, tgt_si, tgt_ei


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

        self.tgt_start_len = []
        self.descr_start_len = []
        sense_ids_strs = []
        for cxt, tgt_ind, def_, hyps in zip(contexts, target_inds, definitions, hypernyms):
            cxt_index_map, cxt_index_list = self._get_token_index_map_and_list(cxt, tokenizer)
            if type(tgt_ind) == int:
                tgt_ind_begin = tgt_ind
                tgt_ind_end = tgt_ind + 1
            elif type(tgt_ind) == tuple:
                tgt_ind_begin, tgt_ind_end = tgt_ind
            else:
                raise ValueError(f"target indices may only be of type int or tuple, but is {type(tgt_ind)}")

            target_start_ind = cxt_index_map[tgt_ind_begin][0] + len(['[CLS]'])
            target_len = sum([len(cxt_index_map[idx]) for idx in range(tgt_ind_begin, tgt_ind_end)])
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
        original_tokens = text.split()
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
        :param target_inds: list of target indices, or list of ranges in the form of (begin_ind, end_ind)
        :param focus_char: character which should be taken to mark the target
        :return: list of marked context strings, updated target indices
        """
        marked_contexts = []
        marked_target_inds = []
        for context, target_i in zip(contexts, target_inds):
            context_tokens = context.split()
            if type(target_i) == int:
                begin_i = target_i
                end_i = target_i + 1
            elif type(target_i) == tuple:
                begin_i, end_i = target_i
            else:
                raise ValueError(f"target indices may only be of type int or tuple, but is {type(target_i)}")
            before_target = context_tokens[:begin_i]
            after_target = context_tokens[end_i:]
            marked_contexts.append(' '.join(before_target +
                                            [focus_char] +
                                            context_tokens[begin_i:end_i] +
                                            [focus_char] +
                                            after_target))
            new_begin_i = begin_i + 1
            new_end_i = end_i + 1
            if new_begin_i == new_end_i - 1:
                marked_target_inds.append(new_begin_i)
            else:
                marked_target_inds.append((new_begin_i, new_end_i))
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


class WiCTSVDatasetCharOffsets(WiCTSVDataset):
    def __init__(self,
                 contexts: List[str],
                 target_ses: List[Tuple[int, int]],
                 hypernyms: List[List[str]],
                 definitions: List[str],
                 tokenizer: PreTrainedTokenizer,
                 labels: List[int] = None,
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
                                                                target_ses=target_ses,
                                                                focus_char=focus_token)

        targets = [cxt[tgt_si:tgt_ei] for cxt, (tgt_si, tgt_ei) in zip(contexts, target_ses)]
        self.tgt_start_len = []
        self.descr_start_len = []
        sense_ids_strs = []
        for cxt, (tgt_si, tgt_ei), def_, hyps, tgt in zip(contexts, target_ses, definitions, hypernyms, targets):
            cxt_index_map, cxt_index_list = self._get_token_index_map_and_list(cxt, tokenizer)
            tokens, tgt_word_si, tgt_word_ei = char_offset2token_offset(cxt, tgt_si, tgt_ei)
            target_start_ind = cxt_index_map[tgt_word_si][0] + len(['[CLS]'])
            target_len = len(sum([cxt_index_map[i] for i in range(tgt_word_si, tgt_word_ei)], []))
            # assert len(tokenizer.tokenize(tgt)) == target_len
            assert tokens[tgt_word_si:tgt_word_ei] == tgt.split()
            self.tgt_start_len.append((target_start_ind, target_len))

            sense_identifiers_str = def_ + '; ' + ', '.join(hyps)
            sense_ids_strs.append(sense_identifiers_str)
            descrs_start_ind = len(cxt_index_list) + len(['[CLS]', '[SEP]'])
            descrs_len = len(tokenizer.tokenize(sense_identifiers_str))
            self.descr_start_len.append((descrs_start_ind, descrs_len))

        if encoding_type == WiCTSVDatasetEncodingOptions.CTX_DEF:
            tokenizer_input = [[context, sense_ids] for context, sense_ids in zip(contexts, sense_ids_strs)]
        else:
            raise NotImplementedError

        self.encodings = tokenizer(tokenizer_input, return_tensors='pt', truncation=True, padding=True)

    @staticmethod
    def mark_target_in_context(contexts: List[str], target_ses: List[Tuple[int, int]], focus_char: str = "$"):
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

        if focus_char is not None:
            for cxt, (tgt_si, tgt_ei) in zip(contexts, target_ses):
                before_target = cxt[:tgt_si]
                after_target = cxt[tgt_ei:]
                marked_cxt = ' '.join([before_target, focus_char, cxt[tgt_si:tgt_ei], focus_char, after_target])
                assert marked_cxt[tgt_si + 3:tgt_ei + 3] == cxt[tgt_si:tgt_ei]
                marked_contexts.append(marked_cxt)
                marked_target_inds.append( (tgt_si+3, tgt_ei+3) )
        assert len(marked_contexts) == len(marked_target_inds)
        return marked_contexts, marked_target_inds
