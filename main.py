import logging
import random
import yaml
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from process import *
from utils import *
from model import EntityModel
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataset(samples_list, batch_size, id2label):
    #[,max_tokens], [,max_tokens],[,max_spans,3]
    input_ids, attention_mask, spans, spans_mask, spans_ner_label = get_input_tensor_bacths(samples_list)
    #
    print(input_ids.size())
    print(spans_ner_label.size())
    temp = dict(Counter(spans_ner_label.view(-1).cpu().numpy()))
    print(temp)
    print({id2label.get(k,'Other'):v for k,v in temp.items()})
    
    tensor_dataset = TensorDataset(input_ids, attention_mask, spans, spans_mask, spans_ner_label)
    batch_dataset = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return batch_dataset


def train(model, train_configs, train_dataset, dev_dataset):
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0])
    else:
        model = model.to('cpu')

    params = [k for k, v in model.named_parameters() if v.requires_grad == True]
    print(params)

    optim_grouped_parameters = [
        {'params': [v for k, v in model.named_parameters()
                    if 'bert' in k]},
        {'params': [v for k, v in model.named_parameters()
                    if 'bert' not in k], 'lr': train_configs['task_learning_rate']}]

    optimizer = AdamW(optim_grouped_parameters, lr=train_configs['learning_rate'])
    total_steps = len(train_dataset) * train_configs['max_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * train_configs['warm_up_ratio']), total_steps)

    for epoch in range(train_configs['max_epochs']):
        model.train()
        loss_sum = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        for step, batch in enumerate(train_dataset, 1):
            input_ids = batch[0].cuda() if torch.cuda.is_available() else batch[0]
            attention_mask = batch[1].cuda() if torch.cuda.is_available() else batch[1]
            token_type_ids = torch.zeros_like(input_ids).cuda() if torch.cuda.is_available() else torch.zeros_like(input_ids)
            spans = batch[2].cuda() if torch.cuda.is_available() else batch[2]
            spans_mask = batch[3].cuda() if torch.cuda.is_available() else batch[3]
            spans_ner_label = batch[4].cuda() if torch.cuda.is_available() else batch[4]

            logits, loss, metrics = model(input_ids, attention_mask, token_type_ids, spans, spans_mask, spans_ner_label)

            optimizer.zero_grad()
            loss.mean().backward()
            scheduler.step()
            optimizer.step()

            loss_sum += loss.mean().item()
            precision_sum += metrics[0].item()
            recall_sum += metrics[1].item()
            f1_sum += metrics[2].item()
            
            if step % 100 == 0:
                print('step : {}, loss : {}, precision : {}, recall : {}, f1 : {}'.format(step, round(loss_sum / step, 4),
                                                                                        round(precision_sum / step, 4),
                                                                                        round(recall_sum / step, 4),
                                                                                        round(f1_sum / step, 4)))
            
        model.eval()
        dev_loss_sum = 0
        dev_precision_sum = 0
        dev_recall_sum = 0
        dev_f1_sum = 0

        for dev_step, batch in enumerate(dev_dataset, 1):
            input_ids = batch[0].cuda() if torch.cuda.is_available() else batch[0]
            attention_mask = batch[1].cuda() if torch.cuda.is_available() else batch[1]
            token_type_ids = torch.zeros_like(input_ids).cuda() if torch.cuda.is_available() else torch.zeros_like(input_ids)
            spans = batch[2].cuda() if torch.cuda.is_available() else batch[2]
            spans_mask = batch[3].cuda() if torch.cuda.is_available() else batch[3]
            spans_ner_label = batch[4].cuda() if torch.cuda.is_available() else batch[4]

            with torch.no_grad():
                _, loss, metrics = model(input_ids, attention_mask, token_type_ids, spans, spans_mask, spans_ner_label)

            dev_loss_sum += loss.mean().item()
            dev_precision_sum += metrics[0].item()
            dev_recall_sum += metrics[1].item()
            dev_f1_sum += metrics[2].item()

        print('epoch : {}, loss : {}, precision : {}, recall : {}, f1 : {}, dev_loss : {}, dev_precision : {}, dev_recall : {}, dev_f1 : {}'.format(
                    epoch,
                    round(loss_sum / step, 4),
                    round(precision_sum / step, 4), round(recall_sum / step, 4), round(f1_sum / step, 4),
                    round(dev_loss_sum / dev_step, 4),
                    round(dev_precision_sum / dev_step, 4), round(dev_recall_sum / dev_step, 4), round(dev_f1_sum / dev_step, 4)))

    torch.save(model.state_dict(), './model_save/net_params.pkl')
    

if __name__ == '__main__':
    set_seed(44)

    ner_label2id = get_labelmap(task_ner_labels['scierc'])

    configs_file = open('configs.yml', 'r', encoding="utf-8")
    configs_data = configs_file.read()
    configs_file.close()
    configs = yaml.load(configs_data, Loader=yaml.Loader)

    data_configs = configs['data_configs']
    model_configs = configs['model_configs']
    train_configs = configs['train_configs']

    train_data_path = './scierc_dataset/processed_data/json/train.json'
    train_dataset = read_dataset(train_data_path)
    train_samples = convert_dataset_to_samples(train_dataset, data_configs['max_span_length'], ner_label2id[0], 'train')
    train_batch_dataset = create_dataset(train_samples, train_configs['batch_size'], ner_label2id[1])

    dev_data_path = './scierc_dataset/processed_data/json/dev.json'
    dev_dataset = read_dataset(dev_data_path)
    dev_samples = convert_dataset_to_samples(dev_dataset, data_configs['max_span_length'], ner_label2id[0], 'dev')
    dev_batch_dataset = create_dataset(dev_samples, train_configs['batch_size'], ner_label2id[1])
    
    test_data_path = './scierc_dataset/processed_data/json/test.json'
    test_dataset = read_dataset(test_data_path)
    test_samples = convert_dataset_to_samples(test_dataset, data_configs['max_span_length'], ner_label2id[0], 'test')
    test_batch_dataset = create_dataset(test_samples, train_configs['batch_size'], ner_label2id[1])
    
    #model = EntityModel(model_configs['bert_dir'], data_configs, model_configs, len(ner_label2id[0])+1)
    #train(model, train_configs, dev_batch_dataset, dev_batch_dataset)
