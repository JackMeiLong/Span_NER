import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

class EntityModel(nn.Module):
    def __init__(self, bert_dir, data_configs, model_configs, num_labels):
        super(EntityModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(bert_dir)
        self.bert_config = AutoConfig.from_pretrained(bert_dir)

        self.width_embedding = nn.Embedding(data_configs['max_span_length'] + 1, data_configs['width_embedding_dim'])

        self.dropout = nn.Dropout(model_configs['hidden_dropout_prob'])

        self.final_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * self.bert_config.hidden_size + data_configs['width_embedding_dim'], data_configs['width_embedding_dim']),
            nn.Dropout(model_configs['hidden_dropout_prob']),
            nn.ReLU(),
            nn.Linear(data_configs['width_embedding_dim'], data_configs['width_embedding_dim']),
            nn.Dropout(model_configs['hidden_dropout_prob']), 
            nn.Linear(data_configs['width_embedding_dim'], num_labels)    
        )
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, input_ids, attention_mask, token_type_ids, spans, spans_mask, spans_ner_label):
        # [batch_size, max_seq_length, 768]
        output = self.bert_model(input_ids, attention_mask, None, return_dict=True)
        sequence_output = self.dropout(output['last_hidden_state'])
        # [batch_size, max_spans]
        spans_start = spans[:, :, 0]
        spans_start_idx = torch.zeros(spans_start.size(0)*spans_start.size(1), input_ids.size(1)).to(input_ids.device)
        spans_start_idx[range(spans_start_idx.size(0)), spans_start.view(-1)] = 1
        # [batch_size, max_spans, max_seq_length]
        spans_start_idx = spans_start_idx.view(spans_start.size(0), spans_start.size(1), -1)
        # [batch_size, max_spans, 768]
        spans_start_rep = torch.einsum('ijk, ikl -> ijl', spans_start_idx, sequence_output)

        # [batch_size, max_spans]
        spans_end = spans[:, :, 1]
        spans_end_idx = torch.zeros(spans_end.size(0)*spans_end.size(1), input_ids.size(1)).to(input_ids.device)
        spans_end_idx[range(spans_end_idx.size(0)), spans_end.view(-1)] = 1
        # [batch_size, max_spans, max_seq_length]
        spans_end_idx = spans_end_idx.view(spans_end.size(0), spans_end.size(1), -1)
        # [batch_size, max_spans, 768]
        spans_end_rep = torch.einsum('ijk, ikl -> ijl', spans_end_idx, sequence_output)

        # [batch_size, max_spans]
        spans_width = spans[:, :, 2]
        # [batch_size, max_spans, width_embedding_dim]
        spans_width_rep = self.width_embedding(spans_width)

        # [batch_size, max_spans, 2*768+width_embedding_dim]
        spans_rep = torch.cat((spans_start_rep, spans_end_rep, spans_width_rep), dim=-1)
        # [batch_size, max_spans, num_labels]
        logits = self.final_layer(spans_rep)

        active_loss = spans_mask.view(-1) == 1
        # [batch_size*max_spans, num_labels]
        active_logits = logits.view(-1, logits.shape[-1])

        results = (logits, )

        if spans_ner_label is not None:
            active_labels = torch.where(
                active_loss, spans_ner_label.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(spans_ner_label)
            )
            loss = self.loss_fct(active_logits, active_labels)
            results += (loss,)

        return results

    def get_metrics(self, y_true, y_pred):
        precision = torch.tensor(precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),average='macro'))
        recall = torch.tensor(recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),average='macro'))
        f1 = torch.tensor(f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(),average='macro'))
        conf_matrix = torch.tensor(confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy()))
        return (precision, recall, f1)
