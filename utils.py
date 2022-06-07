import numpy as np
import torch

def evaluate(model, batchs):
    n_pred = 0
    n_total = 0


    for dev_step, batch in enumerate(batchs, 1):
        input_ids = batch[0].cuda() if torch.cuda.is_available() else batch[0]
        attention_mask = batch[1].cuda() if torch.cuda.is_available() else batch[1]
        token_type_ids = torch.zeros_like(input_ids).cuda() if torch.cuda.is_available() else torch.zeros_like(
            input_ids)
        spans = batch[2].cuda() if torch.cuda.is_available() else batch[2]
        spans_mask = batch[3].cuda() if torch.cuda.is_available() else batch[3]
        spans_ner_label = batch[4].cuda() if torch.cuda.is_available() else batch[4]

        with torch.no_grad():
            logits, loss, metrics = model(input_ids, attention_mask, token_type_ids, spans, spans_mask, spans_ner_label)

        # [batch_size*max_spans, num_labels]
        active_logits = logits.view(-1, logits.shape[-1])


    pass