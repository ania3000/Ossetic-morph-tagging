from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from src.utils import make_last_subtoken_mask

class UDDataset(Dataset):
    def __init__(self, data, tokenizer, min_count=1, tags=None):
        self.data = data
        self.tokenizer = tokenizer
        self.raw_labels = [item["labels"] for item in data if "labels" in item]
        
        if tags is None:
            tag_counts = Counter([tag for elem in data for tag in elem["labels"]])
            self.tags_ = ["<PAD>", "<UNK>"] + [x for x, count in tag_counts.items() if count >= min_count]
        else:
            self.tags_ = tags
            
        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.unk_index = 1
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        
        answer = {
            "input_ids": tokenization["input_ids"], 
            "mask": last_subtoken_mask
        }
        
        if "labels" in item:
            labels = [self.tag_indexes_.get(tag, self.unk_index) for tag in item["labels"]]
            zero_labels = np.array([self.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
            answer["labels"] = zero_labels
            
        return answer
      
class MultiTaskUDDataset(Dataset):
    def __init__(self, data, tokenizer, min_count=1, tags=None):
        self.data = data
        self.tokenizer = tokenizer
        self.ignore_index = -100
        self.unk_index = 0
        self.tasks = list(data[0]["labels"].keys())
        self.raw_labels = [item["raw_labels"] for item in data]

        self.tags_ = {}
        self.tag_indexes_ = {}
        for task in self.tasks:
            if tags is None or task not in tags:
                tag_counts = Counter([label for item in data for label in item["labels"][task]])
                task_tags = ["<UNK>"] + [x for x, count in tag_counts.items() if count >= min_count]
            else:
                task_tags = tags[task]

            self.tags_[task] = task_tags
            self.tag_indexes_[task] = {tag: i for i, tag in enumerate(task_tags)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        input_ids = tokenization["input_ids"]

        answer = {"input_ids": input_ids}
        if "labels" in item:
            labels_out = {}
            for task in self.tasks:
                labels = [self.tag_indexes_[task].get(label, self.unk_index) for label in item["labels"][task]]
                zero_labels = np.array([self.ignore_index] * len(input_ids), dtype=int)
                zero_labels[last_subtoken_mask] = labels
                labels_out[task] = zero_labels
            answer["labels"] = labels_out

        return answer

class MultiTaskDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels_dict = {task_name: [] for task_name in features[0]["labels"]}
        
        for feature in features:
            for task_name, task_labels in feature.pop("labels").items():
                labels_dict[task_name].append(task_labels)

        batch = super().__call__(features)

        batch_labels = {}
        max_length = batch["input_ids"].shape[1]
        for task_name, task_labels in labels_dict.items():
            padded_task_labels = []
            for label in task_labels:
                label = np.array(label)
                padding_length = max_length - label.shape[0]
                padded_label = np.pad(label, (0, padding_length), constant_values=-100) if padding_length > 0 else label
                padded_task_labels.append(padded_label)
            batch_labels[task_name] = torch.tensor(np.array(padded_task_labels), dtype=torch.long)
        
        batch["labels"] = batch_labels
        return batch

class LemmatizationDataset(Dataset):
    def __init__(self, data, tokenizer, min_count=1, tags=None):
        self.data = data
        self.tokenizer = tokenizer
        self.raw_labels = [item["labels"] for item in data if "labels" in item]
        self.raw_words = [item["words"] for item in data if "words" in item]
        self.raw_lemmas = [item["lemmas"] for item in data if "lemmas" in item]

        if tags is None:
            tag_counts = Counter([tag for elem in data for tag in elem["labels"]])
            self.tags_ = ["<PAD>", "<UNK>"] + [x for x, count in tag_counts.items() if count >= min_count]
        else:
            self.tags_ = tags

        self.tag_indexes_ = {tag: i for i, tag in enumerate(self.tags_)}
        self.unk_index = 1
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        tokenization = self.tokenizer(item["words"], is_split_into_words=True)
        last_subtoken_mask = make_last_subtoken_mask(tokenization.word_ids())
        
        answer = {"input_ids": tokenization["input_ids"]}
        if "labels" in item:
            labels = [self.tag_indexes_.get(tag, self.unk_index) for tag in item["labels"]]
            zero_labels = np.array([self.ignore_index] * len(tokenization["input_ids"]), dtype=int)
            zero_labels[last_subtoken_mask] = labels
            answer["labels"] = zero_labels
        return answer
