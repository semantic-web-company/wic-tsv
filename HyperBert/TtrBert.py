from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer, EvalPrediction

from model_evaluation.ttr_dataset import TTRDataset


class TtrBert():
    def __init__(self):
        #model = BertForTokenClassification.from_pretrained("bert-base-german-cased", num_labels=len(ner2indx))
        bert = BertForNer.from_pretrained("bert-base-german-cased", num_labels=5)


class BertForNer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None
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
        reduced_bert_output = bert_output[0]
        dropout_output = self.dropout(reduced_bert_output)
        logits = self.classifier(dropout_output)
        outputs = (logits,)


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


if __name__ == '__main__':

    import logging
    import argparse
    from sklearn.model_selection import train_test_split
    from transformers import Trainer, TrainingArguments
    from model_evaluation import data_processors as dp

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', nargs='?', default='../data/ttf', type=str)
    parser.add_argument('--model_output_path', nargs='?', default='./model', type=str)
    parser.add_argument('--model_name', nargs='?', default='bert-base-uncased', type=str)
    args = parser.parse_args()
    base_path = Path(args.dataset_path)
    output_path = Path(args.model_output_path)

    ttr_path = base_path / 'deutf.train'
    logging.log(logging.INFO, "Load model")

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name)
    model = BertForNer.from_pretrained(model_name, num_labels=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokens, nes, unique_nes = dp.read_conll(ttr_path,0, 4)

    train_token, val_token, train_labels, val_labels = train_test_split(tokens, nes, test_size=0.15, random_state = 42)
    train_token, val_token, train_labels, val_labels = list(train_token), list(val_token), list(train_labels), list(val_labels)
    def_dict = {"PER" : "eine Person",
                  "LOC" : "ein Ort",
                  "ORG": "eine Organisation",
                  "MISC": "ein Ding"}
    hyper_dict = {"PER" : ["Person"],
                  "LOC" : ["Ort"],
                  "ORG": ["Organisation"],
                  "MISC": ["Ding"]}

    #print('train', sum([Counter(l) for l in train_labels]))
    logging.log(logging.INFO, "Load test dataset")
    train_ds = TTRDataset(train_token,hypernyms=hyper_dict, definitions=def_dict, tokenizer=tok, labels=train_labels)

    #print('test', sum([Counter(l) for l in val_labels]))
    logging.log(logging.INFO, "Load val dataset")
    val_ds = TTRDataset(val_token,hypernyms=hyper_dict, definitions=def_dict, tokenizer=tok, labels=val_labels)




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
        eval_dataset=val_ds,
        # eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    logging.log(logging.INFO, "Start training")
    output = trainer.train()
    print(f'Training output: {output}')
    trainer.save_model()
    #preds = trainer.predict(test_dataset=test_ds)
    #print(preds)
    #print(preds.predictions.tolist())
