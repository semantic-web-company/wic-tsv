from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, AutoTokenizer

from model_evaluation.data_collators import DataCollatorForTokenClassificationWithOffsetMapping, \
    DataCollatorForTokenClassificationWithSeparateSequencesAndOffsetMapping
from model_evaluation.ttr_dataset import TTRDataset, TTRSepDataset


class TtrBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.num_labels = config.num_labels
        self.classifier = nn.Linear(2 * config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self,
                _1_input_ids=None,
                _1_attention_mask=None,
                _1_token_type_ids=None,
                _1_position_ids=None,
                _1_head_mask=None,
                _1_inputs_embeds=None,
                _1_offset_mapping=None,
                _1_output_attentions=None,
                _1_output_hidden_states=None,
                _2_input_ids=None,
                _2_attention_mask=None,
                _2_token_type_ids=None,
                _2_position_ids=None,
                _2_head_mask=None,
                _2_inputs_embeds=None,
                _2_offset_mapping=None,
                _2_output_attentions=None,
                _2_output_hidden_states=None,
                _2_reduced_bert_output=None,
                labels=None
                ):

        _1_bert_output = self.bert(
            _1_input_ids,
            attention_mask=_1_attention_mask,
            token_type_ids=_1_token_type_ids,
            position_ids=_1_position_ids,
            head_mask=_1_head_mask,
            inputs_embeds=_1_inputs_embeds,
            output_attentions=_1_output_attentions,
            output_hidden_states=_1_output_hidden_states,
        )
        _1_reduced_bert_output = _1_bert_output[0]  # last_hidden_state, shape=(bs, seq_len, hidden_dim)
        _1_dropout_output = self.dropout(_1_reduced_bert_output)

        if _2_reduced_bert_output is None:
            _2_bert_output = self.bert(
                _2_input_ids,
                attention_mask=_2_attention_mask,
                token_type_ids=_2_token_type_ids,
                position_ids=_2_position_ids,
                head_mask=_2_head_mask,
                inputs_embeds=_2_inputs_embeds,
                output_attentions=_2_output_attentions,
                output_hidden_states=_2_output_hidden_states,
            )
            _2_reduced_bert_output = _2_bert_output[0]
        token_type_1_total_per_line = torch.sum(_2_token_type_ids, dim=1)
        sum_output = torch.sum(input=(_2_reduced_bert_output * _2_token_type_ids.unsqueeze(-1)), dim=1)
        mean_output = torch.div(sum_output, token_type_1_total_per_line.unsqueeze(-1))
        _2_dropout_output = self.dropout(mean_output)  # (bs, hidden_dim)

        expanded = _2_dropout_output.unsqueeze(1).expand(-1, _1_dropout_output.size(1), -1)
        pooled_output = torch.cat((
            _1_dropout_output,
            expanded
        ), 2)  # (bs, seq_size, 2*dim)

        logits = self.classifier(pooled_output)
        outputs = (logits,)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


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
                output_hidden_states=None,
                offset_mapping=None
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
                        format='%(asctime)s %(levelname)-8s %(message)s', )

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', nargs='?', default='../data/ttr', type=str)
    parser.add_argument('--model_output_path', nargs='?', default='./model', type=str)
    parser.add_argument('--model_name', nargs='?', default='bert-base-german-cased', type=str)
    args = parser.parse_args()
    base_path = Path(args.dataset_path)
    output_path = Path(args.model_output_path)

    # ttr_path = base_path / 'deuutf.dev'
    ttr_path = base_path / 'ler.conll'
    logging.log(logging.INFO, "Load model")

    model_name = args.model_name
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = TtrBert.from_pretrained(model_name, num_labels=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # tokens, nes, unique_nes = dp.read_conll(ttr_path,0, 4)
    tokens, nes, unique_nes = dp.read_conll(ttr_path, 0, 1)

    train_token, val_token, train_labels, val_labels = train_test_split(tokens, nes, test_size=0.15, random_state=42)
    small_train_token, small_val_token, small_train_labels, small_val_labels = list(train_token)[:20], \
                                                                               list(val_token)[:20], \
                                                                               list(train_labels)[:20], \
                                                                               list(val_labels)[:20]
    def_dict = {"PER": "ein Mensch",
                "RR": "Person, welche die Rechtsprechung ausübt oder bei Gericht Recht spricht",
                "AN": "bevollmächtigter Rechtsvertreter",

                "LD": "Glied- oder Teilstaat (in bestimmten Staaten), abgrenzbares, historisch oder natürlich zusammengehöriges Gebiet",
                "ST": "meist größere, zivile, zentralisierte, abgegrenzte, häufig und oft historisch mit Stadtrechten ausgestattete Siedlung",
                "STR": "ein landgebundenes Verkehrsbauwerk, das dem Bewegen von Fahrzeugen und Fußgängern zwischen zwei Orten und/oder Positionen dient",
                "LDS": "ein Teil der Erdoberfläche, der sich durch seine einzigartigen physischen und kulturellen Merkmale von der Umgebung abhebt",

                "ORG": "in koordinierter Zusammenschluss von Menschen und Ressourcen, der dem Zweck dient, das Gemeinwohl im Arbeitsfeld der Organisation zu verbessern",
                "UN": "Gesellschaft, die in Produktion oder Handel tätig ist oder Dienstleistungen erbringt",
                "INN": "Einrichtung, Organisationselement, Behörde, Anstalt",
                "GRT": "Organ, dessen Aufgabe es ist, vorgetragene Fälle anzuhören und über sie unter Beachtung der Rechtslage zu entscheiden",
                "MRK": "Ware mit einem bestimmten geschützten Namen",

                "GS": "Regel, die ein Gesetzgeber in einem bestimmten Verfahren erlässt und die die jeweilig Untergebenen zu befolgen haben",
                "VO": "gesetzesähnliche Vorschrift, die von einer Verwaltungsbehörde erlassen wird",
                "EUN": "Die Europäischen Normen sind Regeln, die von einem der drei europäischen Komitees für Standardisierung ratifiziert worden sind",

                "VS": "Anweisung, die man befolgen muss",
                "VT": "rechtlich bindende Vereinbarung zwischen mindestens zwei verschiedenen Partnern",

                "RS": "Menge und/oder Art der erfolgten Gerichtsurteile",
                "LIT": ""
                }
    hyper_dict = {"PER": ["Person", "Mensch"],
                  "RR": ["Person", "Jurist"],
                  "AN": ["Person", "Vertreter"],

                  "LD": ["Ort", "Staat", "Verwaltungsgebiet"],
                  "ST": ["Ort", "Ortschaft", "Siedlung"],
                  "STR": ["Ort", "Verkehrsweg"],
                  "LDS": ["Ort", "Region"],

                  "ORG": ["Organisation", "Einrichtung"],
                  "UN": ["Organisation", "Gesellschaft", "juristische Person", "Rechtsform"],
                  "INN": ["Organisation"],
                  "GRT": ["Organisation", "Organ", "Institution"],
                  "MRK": ["Organisation"],

                  "GS": ["Rechtsvorschrift"],
                  "VO": ["Rechtsvorschrift", "Ordnung", "Rechtsnorm"],
                  "EUN": ["Rechtsvorschrift", "Europäische Vorschrift"],

                  "VS": ["Regulation", "Regel"],
                  "VT": ["Regulation", "Vereinbarung"],

                  "RS": ["Rechtsprechung"],
                  "LIT": ["Rechtsliteratur"]
                  }
    cls_label_dict = {
        "PER": "Person",
        "RR": "Richter",
        "AN": "Anwalt",

        "LD": "Land",
        "ST": "Stadt",
        "STR": "Straße",
        "LDS": "Landschaft",

        "ORG": "Organisation",
        "UN": "Unternehmen",
        "INN": "Institution",
        "GRT": "Gericht",
        "MRK": "Marke",

        "GS": "Gesetz",
        "VO": "Verordnung",
        "EUN": "EU Norm",

        "VS": "Vorschrift",
        "VT": "Vertrag",

        "RS": "Rechtsprechung",
        "LIT": "Rechtsliteratur"
    }

    print('train', sum([Counter(l) for l in train_labels], Counter()))
    logging.log(logging.INFO, "Load test dataset")
    train_ds = TTRSepDataset(contexts=small_train_token, labels=small_train_labels,
                             hypernyms=hyper_dict, definitions=def_dict, cls_labels=cls_label_dict,
                             tokenizer=tok)

    print('test', sum([Counter(l) for l in val_labels], Counter()))
    logging.log(logging.INFO, "Load val dataset")
    val_ds = TTRSepDataset(contexts=small_val_token, labels=small_val_labels,
                           hypernyms=hyper_dict, definitions=def_dict, cls_labels=cls_label_dict,
                           tokenizer=tok)


    def compute_metrics(pred, detailed_metrics=False):
        labels = [y for x in pred.label_ids for y in x]
        preds = [y for x in pred.predictions.argmax(-1) for y in x]
        average = None if detailed_metrics else 'micro'
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2],
                                                                         average=average)
        support = support if support is not None else len(labels)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'support': support
        }


    training_args = TrainingArguments(
        output_dir=str(output_path),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy='epoch',
        num_train_epochs=4,
        per_device_train_batch_size=2,
    )

    data_collator = DataCollatorForTokenClassificationWithSeparateSequencesAndOffsetMapping(tok, max_length=512)
    trainer = Trainer(
        model=model,
        args=training_args,
        # prediction_loss_only=True,
        data_collator=data_collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     # prediction_loss_only=True,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     compute_metrics=compute_metrics,
    #     # data_collator=
    #     # tokenizer=
    # )
    logging.log(logging.INFO, "Start training")
    output = trainer.train()
    print(f'Training output: {output}')
    model.config.label2id = train_ds.tag2idx
    model.config.id2label = indx2ner = {id: val for val, id in train_ds.tag2idx.items()}
    trainer.save_model()
    preds = trainer.predict(test_dataset=val_ds)
    # print(preds)
    # print(preds.predictions.tolist())
    # trainer.eval_dataset=val_ds
    # eval_data = trainer.evaluate()
    eval_data = compute_metrics(preds, True)
    print(eval_data)
    # for i in indx2ner.keys():
    #     print(indx2ner[i], eval_data['support'][i], eval_data['f1'][i])
