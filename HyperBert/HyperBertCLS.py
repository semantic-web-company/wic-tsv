from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, BertModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EvalPrediction
from transformers import Trainer, TrainingArguments

from model_evaluation.wictsv_dataset import WiCTSVDataset


class HyperBertCLS(BertPreTrainedModel):
    """
    This model only takes the representation of [CLS] token to classify
    """
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hyper_classifier = nn.Linear(config.hidden_size, 1)  # BERT
        self.dropout = nn.Dropout(2*config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
            *args,
            **kwargs
    ):
        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_lastbutone_layer = bert_output[2][-2]  # (bs, seq_len, dim)
        cls_output = hidden_lastbutone_layer[:, 0]  # one layer before the last one

        pooled_output = cls_output

        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.hyper_classifier(pooled_output)  # (bs, 1)

        outputs = (logits,)  # + bert_output[1:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits,  # (hidden_states), (attentions)


class WiCTSVDatasetCLSCharOffsets(torch.utils.data.Dataset):
    """
    This dataset class is useful if you want to provide target offset as individual chars indices instead of tokens
    """
    def __init__(self, tok, contexts, target_ses, hypernyms, definitions, labels=None, focus_token='$'):
        self.len = len(contexts)
        self.labels = labels
        if focus_token is not None:
            prep_cxts = []
            for cxt, (tgt_si, tgt_ei) in zip(contexts, target_ses):
                prep_cxt = cxt[:tgt_si] + f' {focus_token} ' + cxt[tgt_si:tgt_ei] + f' {focus_token} ' + cxt[tgt_ei:]
                # prep_cxt.insert(tgt_si + 1, f' {focus_token} ')  # after target
                # prep_cxt.insert(tgt_ei, f' {focus_token} ')  # before target
                assert prep_cxt[tgt_si + 3:tgt_ei + 3] == cxt[tgt_si:tgt_ei]
                prep_cxts.append(prep_cxt)
        else:
            prep_cxts = contexts
        self.encodings = tok([[context, definition + ' ; ' + f' {focus_token} '.join(hyps)]
                              for context, definition, hyps in zip(prep_cxts, definitions, hypernyms)],
                             return_tensors='pt', truncation=True, padding=True)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(float(self.labels[idx]))
        return item

    def __len__(self):
        return self.len


def compute_metrics(p: EvalPrediction) -> Dict:
    fp = p.predictions
    binary_preds = (p.predictions > 0).astype(type(p.label_ids[0]))
    binary = binary_preds.T == p.label_ids
    acc = binary.mean()
    precision, r, f1, _ = precision_recall_fscore_support(y_true=p.label_ids, y_pred=binary_preds, average='binary')
    return {
        "acc": acc,
        "F_1": f1,
        "P": precision,
        "R": r,
        "Positive": binary_preds.sum() / binary_preds.shape[0]
    }


if __name__ == '__main__':
    import logging
    import argparse

    from model_evaluation import data_processors as dp

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', nargs='?', default='../data', type=str)
    parser.add_argument('--model_output_path', nargs='?', default='./model', type=str)
    parser.add_argument('--model_name', nargs='?', default='bert-base-uncased', type=str)
    args = parser.parse_args()
    base_path = Path(args.dataset_path)
    output_path = Path(args.model_output_path)
    wic_tsv_train = base_path / 'Training'
    wic_tsv_dev = base_path / 'Development'
    wic_tsv_test = base_path / 'Test'

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    model = HyperBertCLS.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    contexts, target_ses, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_train)
    print('train', Counter(labels))
    train_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                             tokenizer=tok,
                             focus_token='$',
                             labels=labels)

    contexts, target_ses, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_dev)
    print('dev', Counter(labels))
    dev_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                           tokenizer=tok,
                           focus_token='$',
                           labels=labels)

    contexts, target_ses, hypernyms, definitions, labels = dp.read_wic_tsv(wic_tsv_test)
    if labels is not None:
        print('test', Counter(labels))
    test_ds = WiCTSVDataset(contexts, target_ses, hypernyms, definitions,
                            tokenizer=tok,
                            focus_token='$',
                            labels=labels)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        num_train_epochs=10,
        per_device_train_batch_size=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    output = trainer.train()
    print(f'Training output: {output}')
    trainer.save_model()
    preds = trainer.predict(test_dataset=test_ds)
    print(preds)
    print(preds.predictions.tolist())
