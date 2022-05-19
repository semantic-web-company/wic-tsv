from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


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