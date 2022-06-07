import os
import numpy as np
import json
import copy
import logging
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger('root')

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

def get_labelmap(label_list):
    label2id = {}
    id2label = {}
    for i, label in enumerate(label_list):
        label2id[label] = i+1
        id2label[i+1] = label
    return label2id, id2label

def read_dataset(json_file):
    gold_docs = [json.loads(line) for line in open(json_file)]
    return gold_docs

def convert_dataset_to_samples(dataset, max_span_length, ner_label2id, mode):
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_ner_label = 0

    for c, doc in enumerate(dataset):
        sent_start = 0
        pre_length = 0
        for k, sent in enumerate(doc['sentences']):
            sample = {'doc_key': doc.get('doc_key'), 'tokens': sent, 'sent_length': len(sent)}
            max_len = max(max_len, len(sent))
            max_ner = max(max_ner, len(doc['ner'][k]))

            sent_ner = {}
            num_ner += len(doc['ner'][k])
            for ner in doc['ner'][k]:
                sent_ner[(ner[0], ner[1])] = ner[2]

            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []
            for i in range(len(sent)):
                for j in range(i, min(len(sent), i + max_span_length)):
                    sample['spans'].append((i, j, j - i + 1))
                    span2id[(i, j)] = len(sample['spans']) - 1
                    if (i + sent_start, j + sent_start) not in sent_ner:
                        sample['spans_label'].append(0)
                    else:
                        num_ner_label += 1
                        sample['spans_label'].append(ner_label2id[sent_ner[(i + sent_start, j + sent_start)]])

            pre_length += len(sent)
            sent_start = pre_length - 1

            samples.append(sample)

    print('Extracted {} samples, with {} NER labels'.format(len(samples), num_ner))
    print('max_len {}, max_ner_per_sent {}'.format(max_len, max_ner))
    print('num_ner_label {}'.format(num_ner_label))

    return samples

def get_input_tensor_bacths(samples_list, training=True):
    tokens_tensor_list = []
    bert_spans_tensor_list = []
    spans_ner_label_tensor_list = []
    sentence_length = []

    max_tokens = 0
    max_spans = 0
    for sample in samples_list:
        tokens = sample['tokens']
        spans = sample['spans']
        spans_ner_label = sample['spans_label']

        tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = get_input_tensors(tokens, spans, spans_ner_label)
        tokens_tensor_list.append(tokens_tensor)
        bert_spans_tensor_list.append(bert_spans_tensor)
        spans_ner_label_tensor_list.append(spans_ner_label_tensor)
        assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
        if (tokens_tensor.shape[1] > max_tokens):
            max_tokens = tokens_tensor.shape[1]
        if (bert_spans_tensor.shape[1] > max_spans):
            max_spans = bert_spans_tensor.shape[1]

    # apply padding and concatenate tensors
    final_tokens_tensor = None
    final_attention_mask = None
    final_bert_spans_tensor = None
    final_spans_ner_label_tensor = None
    final_spans_mask_tensor = None
    for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor in zip(tokens_tensor_list, bert_spans_tensor_list,
                                                                        spans_ner_label_tensor_list):
        # padding for tokens
        num_tokens = tokens_tensor.shape[1]
        tokens_pad_length = max_tokens - num_tokens
        attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
        if tokens_pad_length > 0:
            pad = torch.full([1, tokens_pad_length], tokenizer.pad_token_id, dtype=torch.long)
            tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
            attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
            attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

        # padding for spans
        num_spans = bert_spans_tensor.shape[1]
        spans_pad_length = max_spans - num_spans
        spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
        if spans_pad_length > 0:
            pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
            bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
            mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
            spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
            spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)

        # update final outputs
        if final_tokens_tensor is None:
            final_tokens_tensor = tokens_tensor
            final_attention_mask = attention_tensor
            final_bert_spans_tensor = bert_spans_tensor
            final_spans_ner_label_tensor = spans_ner_label_tensor
            final_spans_mask_tensor = spans_mask_tensor
        else:
            final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
            final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
            final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
            final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
            final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
    #[batch_size, max_tokens]/[batch_size, max_tokens]/[batch_size, max_spans, 3]
    return final_tokens_tensor, final_attention_mask, final_bert_spans_tensor, final_spans_mask_tensor, final_spans_ner_label_tensor

def get_input_tensors(tokens, spans, spans_ner_label):
    start2idx = []
    end2idx = []

    bert_tokens = []
    bert_tokens.append(tokenizer.cls_token)
    for token in tokens:
        start2idx.append(len(bert_tokens))
        sub_tokens = tokenizer.tokenize(token)
        bert_tokens += sub_tokens
        end2idx.append(len(bert_tokens) - 1)
    bert_tokens.append(tokenizer.sep_token)

    indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])

    bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
    bert_spans_tensor = torch.tensor([bert_spans])

    spans_ner_label_tensor = torch.tensor([spans_ner_label])

    return tokens_tensor, bert_spans_tensor, spans_ner_label_tensor
