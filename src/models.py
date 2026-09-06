import torch
import torch.nn as nn
from transformers import AutoModel

class AutoModelForMultiTaskTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels_dict, task_weights=None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        
        self.classifiers = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_labels)
            for task, num_labels in num_labels_dict.items()
        })
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.task_weights = task_weights or {task: 1.0 for task in num_labels_dict}

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits, losses = {}, {}

        for task, classifier in self.classifiers.items():
            task_logits = classifier(sequence_output)
            logits[task] = task_logits

            if labels is not None and task in labels:
                loss = self.loss_fct(
                    task_logits.view(-1, task_logits.size(-1)),
                    labels[task].view(-1)
                )
                losses[task] = loss

        if labels is not None:
            total_loss, total_weight = 0, 0
            for task, loss in losses.items():
                w = self.task_weights.get(task, 1.0)
                total_loss += w * loss
                total_weight += w

            return {
                "loss": total_loss / total_weight,
                "logits": logits
            }

        return {"logits": logits}
