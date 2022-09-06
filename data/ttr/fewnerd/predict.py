import json
import logging
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import dicts
from HyperBert.TtrBert import TtrBert
import model_evaluation.predict_ttr as pred
import model_evaluation.ttr_dataset as ttr_ds
import model_evaluation.data_collators as ttr_dc


def read_id2tags(path):
    with open(path) as f:
        id2tags = json.loads(f.read())
        id2tags = {int(k): f"I-{v}" if v != 'O' else v for k, v in
                   id2tags.items()}  # <--we can now handle dashes in labels
    return id2tags


def read_jsonlines(jsonl_file_path: str):
    tokens = []
    coarse_tags = []
    fine_tags = []
    ids = []
    with open(jsonl_file_path) as f:
        for line in f:
            instance = json.loads(line)
            tokens.append(instance['tokens'])
            coarse_tags.append(instance['coarse_tags'])
            fine_tags.append(instance['fine_tags'])
            ids.append(instance['id'])
    return tokens, coarse_tags, fine_tags, ids


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', )

    def_dict = dicts.wn_def
    hyper_dict = dicts.wn_hyper

    # Data loading
    logging.log(logging.INFO, "Load test dataset")
    fine_tags_path = Path(__file__).parent / 'id2fine_tags.json'
    id2fine_tags = read_id2tags(fine_tags_path)
    test_path = Path(__file__).parent / "dev_subset.json"
    test_tokens, test_int_coarse_tags, test_int_fine_tags, test_ids = read_jsonlines(test_path)
    test_fine_tags = [[id2fine_tags[t] for t in sent] for sent in test_int_fine_tags]
    test_labels_cnt = sum([Counter(l) for l in test_fine_tags], Counter())
    print('test', test_labels_cnt)
    print(len(test_labels_cnt))
    test_labels = {x[2:] if x.startswith('I-') else x for tags_ls in test_fine_tags for x in tags_ls}

    # base_path = Path("/kaggle/input/legalentityrecognition")
    # output_path = Path("/kaggle/working")
    # ttr_path = base_path / 'ler.conll'

    logging.log(logging.INFO, "Load model")
    model = TtrBert.from_pretrained('/home/artemrevenko/local_data/ML_models/TTR/models/ttr bilin 2epoch wn 30000 7neg', num_labels=3)
    # model = BertForNer.from_pretrained('/kaggle/input/fewnerdinterttrmodels/wn 30000 neg9 dynb 2epoch', num_labels=3)
    #
    tok = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True)
    data_collator = ttr_dc.DataCollatorForTokenClassificationWithSeparateSequencesAndOffsetMapping(tok, max_length=512)
    # data_collator = DataCollatorForTokenClassificationWithOffsetMapping(tok, max_length=512)
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    test_ds = ttr_ds.TTRSepDataset(hypernyms=hyper_dict,
                                   definitions=def_dict,
                                   contexts=test_tokens,
                                   cls_labels=dicts.cls_label_dict,
                                   tokenizer=tok,
                                   # labels=test_fine_tags,
                                   target_classes=list(test_labels))
    batch_size = 8
    dataloader = DataLoader(test_ds,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=lambda x: {k: v.to(device) for k, v in data_collator(x).items()}
                            )


    print(f"***** Running predictions in eval mode *****")
    print(f"  Num examples = {len(test_ds.contexts)}")
    preds = pred.predict(
        model,
        ds=test_ds,
        dataloader=dataloader)
    for i, x in enumerate(preds):
        print()
        print(i)
        for token, pred_label, gold_label in zip(*x):
            print(f'{token}\t{pred_label}\t{gold_label}')
        if i >= 3: break
