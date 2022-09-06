from typing import List

import numpy as np
from scipy.special import softmax
import torch

from model_evaluation.ttr_dataset import TTRDataset,  tokenize_and_preserve_labels, TTRSepDataset


def align_sent_preds(cur_sent_target_cls_ids: List[str],
                     cur_sent_preds: np.ndarray,
                     b_pos: int, i_pos: int):
    b_scores = cur_sent_preds[:, :, b_pos]  # B tag
    b_max_inds = b_scores.argmax(axis=0)
    b_max_vals = b_scores.max(axis=0)
    in_scores = cur_sent_preds[:, :, i_pos]  # I tag
    in_max_inds = in_scores.argmax(axis=0)
    in_max_vals = in_scores.max(axis=0)

    out_labels = []
    for i, (b_val, i_val) in enumerate(zip(b_max_vals, in_max_vals)):
        if b_val >= i_val and b_val >= 0.5:
            out_labels.append(f'B-{cur_sent_target_cls_ids[b_max_inds[i]]}')
        elif i_val >= b_val and i_val >= 0.5:
            out_labels.append(f'I-{cur_sent_target_cls_ids[in_max_inds[i]]}')
        else:
            out_labels.append('O')

    return out_labels


def tokenize_and_align_labels(context: List[str],
                              tokenized_labels: List[str],
                              tokenizer):
    tokenized_labels_copy = tokenized_labels[1:]  # leave out the first label of the CLS token
    token_labels = []
    for word in context:
        tokens = tokenizer.tokenize(word)
        first_subword_label = tokenized_labels_copy.pop(0)
        token_labels += [first_subword_label]
        for _ in tokens[1:]:  # we pop as many labels as many subwords we have
            tokenized_labels_copy.pop(0)
    assert (len(context) == len(token_labels))
    return context, token_labels


def predict(model,
            ds: TTRDataset,
            dataloader,
            ):
    preds: np.ndarray = None
    sent_is = []
    target_cls_ids = []

    model.eval()
    with torch.no_grad():
        predict_iter = iterate_predictions(model, dataloader)
        current_idx = 0
        for batch_eval_preds in predict_iter:
            if preds is None:
                preds = batch_eval_preds
            else:
                existing_pred_token_size = preds.shape[1]
                new_pred_token_size = batch_eval_preds.shape[1]
                new_preds = batch_eval_preds
                if existing_pred_token_size > new_pred_token_size:
                    npad = ((0, 0), (0, existing_pred_token_size - new_pred_token_size), (0, 0))
                    new_preds = np.pad(new_preds, pad_width=npad, mode='constant', constant_values=0)
                elif existing_pred_token_size < new_pred_token_size:
                    npad = ((0, 0), (0, new_pred_token_size - existing_pred_token_size), (0, 0))
                    preds = np.pad(preds, pad_width=npad, mode='constant', constant_values=0)
                preds = np.append(preds, new_preds, axis=0)

            for batch_i in range(len(batch_eval_preds)):
                sent_i, context, token_cls = ds.instance_tuples[current_idx]

                current_idx += 1
                sent_is.append(sent_i)
                target_cls_ids.append(token_cls)
                # we are assuming that different instances of the same context are one after the other
                if len(sent_is) >= 2 and sent_is[-1] != sent_is[-2]:
                    this_sent_i = sent_is[-2]
                    #
                    current_sent_inds = [i for i, x in enumerate(sent_is) if x == this_sent_i]
                    cur_sent_target_cls_ids = [target_cls_ids[x] for x in current_sent_inds]
                    cur_sent_preds = np.asarray([softmax(preds[x], axis=1) for x in current_sent_inds])
                    cur_sent_pred_labels = align_sent_preds(cur_sent_target_cls_ids, cur_sent_preds,
                                                            b_pos=ds.tag2idx['B'], i_pos=ds.tag2idx['I'])

                    pred_tokens, pred_labels = tokenize_and_align_labels(context=ds.contexts[this_sent_i],
                                                                         tokenized_labels=cur_sent_pred_labels,
                                                                         tokenizer=ds.tokenizer)
                    # ds_labels = ds.sent_labels[this_sent_i]
                    yield pred_tokens, pred_labels, ds.sent_labels[this_sent_i] if ds.sent_labels is not None else [None]*len(pred_labels)


def iterate_predictions(model,
                        dataloader):
    for step, batch in enumerate(dataloader):
        model_output = model(**batch)
        logits = model_output[-1]
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.cpu().numpy()
        yield preds
