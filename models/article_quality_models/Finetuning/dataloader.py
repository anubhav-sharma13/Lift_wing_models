import torch
import json
import os, sys
import linecache
from torch.utils.data import DataLoader, TensorDataset, Dataset

# required to access the python modules present in project directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# now we can import the all modules present in project folder
from utils import load_jsonl_file


class TextDataset(Dataset):
    def __init__(self, tokenizer, filename, dataset_count, max_seq_len, label_encoder, inference):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_encoder = label_encoder
        self.inference = inference
        self.dataset = load_jsonl_file(filename, valid_field=['title', 'label'])
        if dataset_count>0:
            self.dataset = load_jsonl_file(filename, valid_field=['title', 'label'])[:dataset_count]
    
    def preprocess(self, text):
        tokenzier_args = {'text': text, 'truncation': True, 'padding': 'max_length', 
                                    'max_length': self.max_seq_len, 'return_attention_mask': True,
                                    "return_tensors":"pt"}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'][0], tokenized_data['attention_mask'][0]

    def __getitem__(self, idx):
        data_instance = self.dataset[idx]
        input_ids, attention_mask = self.preprocess(data_instance['title'])
        if self.inference:
            return input_ids, attention_mask
        label = self.label_encoder.transform([data_instance['label']])[0]
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.dataset)

def get_dataset_loaders(tokenizer, filename, dataset_count, label_encoder, batch_size=8, num_threads=0, max_seq_len=100, inference=False):
    dataset = TextDataset(tokenizer, filename, dataset_count, max_seq_len, label_encoder, inference)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads)
    return input_dataloader
