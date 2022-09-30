from collections import Counter
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, EvalPrediction, DistilBertModel, DistilBertPreTrainedModel
from transformers import BertPreTrainedModel, BertModel
from transformers import Trainer, TrainingArguments

from model_evaluation.data_collators import DataCollatorForSequenceClassificationWithAdditionalItemData
from model_evaluation.wictsv_dataset import WiCTSVDataset


class HyperBert3(BertPreTrainedModel):
    """
    This model takes 3 different token representations to classify: [CLS], target, and definition+hypernyms
    """
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.hyper_classifier = nn.Linear(3*config.hidden_size, 1)  # BERT
        self.dropout = nn.Dropout(2*config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            target_mask=None,
            descr_mask=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
    ):
        #tgt_inds = {}
        #descr_inds = {}
        #tgt_embeds = []
        #descr_embeds = []
        #for row_ind, tgt_ind in [(int(x[0]),int(x[1])) for x in target_mask.nonzero()]:
        #    if row_ind in tgt_inds:
        #        tgt_inds[row_ind].append(tgt_ind)
        #    else:
        #        tgt_inds[row_ind] = [tgt_ind]
        #for row_ind, tgt_ind in [(int(x[0]),int(x[1])) for x in descr_mask.nonzero()]:
        #    if row_ind in descr_inds:
        #        descr_inds[row_ind].append(tgt_ind)
        #    else:
        #        descr_inds[row_ind] = [tgt_ind]

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
        cls_output = bert_output[1]  # (bs, dim)

        # get mean representation of target tokens and description
        target_output = torch.div(
                            torch.matmul(target_mask.unsqueeze(1).type(torch.float),
                                         hidden_lastbutone_layer
                                         ).squeeze(),
                            target_mask.sum(dim=1).unsqueeze(dim=1)
                        )
        descr_output = torch.div(
                            torch.matmul(descr_mask.unsqueeze(1).type(torch.float),
                                         hidden_lastbutone_layer
                                         ).squeeze(),
                            descr_mask.sum(dim=1).unsqueeze(dim=1)
                        )

        #embedding_dim = hidden_lastbutone_layer.shape[2]
        #target_mask_matrix = torch.repeat_interleave(target_mask.unsqueeze(2),embedding_dim, dim=2)
        #descr_mask_matrix = torch.repeat_interleave(descr_mask.unsqueeze(2),embedding_dim, dim=2)
        #
        #target_output_2 = (hidden_lastbutone_layer * target_mask_matrix).sum(dim=1) / target_mask_matrix.sum(dim=1)
        #descr_output_2 = (hidden_lastbutone_layer * descr_mask_matrix).sum(dim=1) / descr_mask_matrix.sum(dim=1)
        #
        #for i, seq_out in enumerate(hidden_lastbutone_layer.split(1, dim=0)):
        #    seq_out = seq_out.squeeze()
        #    row_tgt_embeds = seq_out[tgt_inds[i]]
        #    row_tgt_mean_embeds = torch.mean(row_tgt_embeds, dim=0).squeeze()  # (1, dim)
        #    row_descr_embeds = seq_out[descr_inds[i]]
        #    row_descr_mean_embeds = torch.mean(row_descr_embeds, dim=0).squeeze()  # (1, dim)
        #    tgt_embeds.append(row_tgt_mean_embeds)
        #    descr_embeds.append(row_descr_mean_embeds)
        #target_output_3 = torch.stack(tgt_embeds)  # (bs, dim)
        #descr_output_3 = torch.stack(descr_embeds)  # (bs, dim)

        #(torch.allclose(target_output, target_output_2))
        #(torch.allclose(target_output, target_output_3))
        #(torch.allclose(target_output_2, target_output_3))

        #(torch.allclose(descr_output, descr_output_2))
        #(torch.allclose(descr_output, descr_output_3))
        #(torch.allclose(descr_output_2, descr_output_3))

        pooled_output = torch.cat((
            target_output,
            descr_output,
            cls_output
        ), 1)  # (bs, 3*dim)

        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.hyper_classifier(pooled_output)  # (bs, 1)

        outputs = (logits,)  # + bert_output[1:]  # add hidden states and attention if they are here

        if labels is not None:
            if labels.dtype != torch.float:
                labels = labels.type(torch.float)

            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels.squeeze())
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits,  # (hidden_states), (attentions)


class HyperDistilBert3(DistilBertPreTrainedModel):
    """
    This model takes 3 different token representations to classify: [CLS], target, and definition+hypernyms
    """

    def __init__(self, config):
        super().__init__(config)

        self.bert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim) # needed because DistilBERT does not have a pooling layer
        self.hyper_classifier = nn.Linear(3 * config.hidden_size, 1)  # BERT
        self.dropout = nn.Dropout(2 * config.dropout)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            target_mask=None,
            descr_mask=None,
            labels=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=True,
    ):

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            #token_type_ids=token_type_ids,
            #position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_lastbutone_layer = bert_output[1][-2]  # (bs, seq_len, dim)
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        context_output = self.pre_classifier(pooled_output)  # (bs, dim)

        # get mean representation of target tokens and description
        target_output = torch.div(
            torch.matmul(target_mask.unsqueeze(1).type(torch.float),
                         hidden_lastbutone_layer
                         ).squeeze(),
            target_mask.sum(dim=1).unsqueeze(dim=1)
        )
        descr_output = torch.div(
            torch.matmul(descr_mask.unsqueeze(1).type(torch.float),
                         hidden_lastbutone_layer
                         ).squeeze(),
            descr_mask.sum(dim=1).unsqueeze(dim=1)
        )

        pooled_output = torch.cat((
            target_output,
            descr_output,
            context_output
        ), 1)  # (bs, 3*dim)

        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.hyper_classifier(pooled_output)  # (bs, 1)

        outputs = (logits,)  # + bert_output[1:]  # add hidden states and attention if they are here

        if labels is not None:
            if labels is not None:
                if labels.dtype != torch.float:
                    labels = labels.type(torch.float)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.squeeze(1), labels.squeeze())
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits,  # (hidden_states), (attentions)


if __name__ == '__main__':
    import logging
    import argparse

    from model_evaluation import data_processors as dp

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', nargs='?', default='../data/en', type=str)
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
    model = HyperBert3.from_pretrained(model_name)
    data_collator = DataCollatorForSequenceClassificationWithAdditionalItemData(tok)
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

    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        num_train_epochs=10,
        per_device_train_batch_size=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # prediction_loss_only=True,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        # eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    output = trainer.train()
    print(f'Training output: {output}')
    trainer.save_model()
    preds = trainer.predict(test_dataset=test_ds)
    print(preds)
    print(preds.predictions.tolist())
