#bert model input features
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import string
import os
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, IterableDataset
import linecache

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, input_length, guid):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        #self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_length = input_length

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

def space_tokenizer(text):
  return [x for x in text.split()]

def title_tokenizer(title_text):
  exclude = set(string.punctuation)
  processed_title_text = ''.join(ch if ch not in exclude else ' ' for ch in title_text)
  return space_tokenizer(processed_title_text)

class MultiLabelTextProcessor(DataProcessor):
    def __init__(self, data_dir, logger, max_tokens=512):
        self.data_dir = data_dir
        self.labels = self.get_labels()
        self.logger = logger
        self.max_token_seq = max_tokens
    
    def get_train_examples(self, data_dir, file_name):
        self.logger.info(" training data - looking at {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(os.path.join(data_dir, file_name))
        
    def get_dev_examples(self, data_dir, file_name):
        self.logger.info("validation data - looking at {}".format(os.path.join(data_dir, file_name)))
        return self._create_examples(os.path.join(data_dir, file_name))
    
    def get_labels(self):
        ene_ids = set()
        file_path = os.path.join(self.data_dir, 'enwiki.human_labeled_revisions.20k_2015.json')
        with open(file_path, 'r', encoding='utf-8') as ene_file:
            for line in ene_file.readlines():
                line = str(line).strip()
                #workaround for escape characters
                if "\'" in line:
                    line = line.replace("\'", "\\'")
                data = json.loads(line)
                ene_id = data.get('ENE_id').strip()
                ene_ids.add(ene_id)
        return list(ene_ids)

    def get_test_examples(self, data_dir, data_file_name):
        self.logger.info("inference data - looking at {}".format(os.path.join(data_dir, data_file_name)))
        return self._create_examples(os.path.join(data_dir, data_file_name), labels_available=False)

    def _create_examples(self, file_path, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        empty_counter=0
        with open(file_path, 'r', encoding='utf-8') as dfile:
            for line in dfile.readlines():
                json_data = json.loads(line.strip())
                guid = int(json_data['page_id'].strip())
                text_a = ' '.join(title_tokenizer(json_data['title']))
                text_b = ' '.join(space_tokenizer(json_data['text'])[:self.max_token_seq])
                if text_b.strip()=='' or len(text_b.strip())==0:
                    text_b=None
                    empty_counter+=1
                if labels_available:
                    labels = json_data['actual_classes']
                else:
                    #empty label
                    labels = []
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels))
        if empty_counter!=0:
            self.logger.info(" %s empty articles"%empty_counter)
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example_to_features(example_row):
    """Loads a data file into a list of `InputBatch`s."""
    
    example, label_encoder, max_seq_length, tokenizer, multilabel_binarizer = example_row
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 4)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.eos_token]
    #segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += [tokenizer.sep_token] + tokens_b + [tokenizer.eos_token]
        #segment_ids += [0] * (len(tokens_b) + 2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    input_length = len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    #segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    #assert len(segment_ids) == max_seq_length
    
    label_ids = label_encoder.transform(example.labels)
    if len(label_ids)==0:
        label_ids = [[-1]]
    else:
        #label_ids = label_ids[0]
        label_ids = label_ids
        if multilabel_binarizer is not None:
            label_ids = multilabel_binarizer.transform([label_ids])[0]
    
    #return InputFeatures(input_ids=input_ids,
    #                          input_mask=input_mask,
    #                          segment_ids=None,
    #                          label_ids=label_ids,
    #                          input_length=input_length,
    #                          guid=example.guid)
    return [
              input_ids,
              input_mask,
              label_ids,
              input_length,
              example.guid,
           ]

def convert_examples_to_features(examples, label_encoder, max_seq_length, tokenizer, multilabelbinarizer=None):

    examples = [(example, label_encoder, max_seq_length, tokenizer, multilabelbinarizer) for example in examples]# if len(example.labels)<=1]

    host_cpu_count = cpu_count()
    if host_cpu_count < 4:
        process_count = 1
    else:
        process_count = 2 


    #speedup the tokenizer process
    with Pool(process_count) as p:
        # features = list(tqdm(p.imap(convert_single_example_to_features, examples, chunksize=100), total=len(examples)))
        features = list(p.imap(convert_single_example_to_features, examples, chunksize=1000))

    return features

class TextIterableDataset(IterableDataset):
    def __init__(self, filename):
        self.filename = filename

    def preprocess(self, json_data):

        all_input_page_ids = torch.tensor(json_data['page_id'], dtype=torch.int)
        all_input_ids = torch.tensor(json_data['input_ids'], dtype=torch.long)
        all_input_mask = torch.tensor(json_data['input_mask'], dtype=torch.long)
        all_label_ids = torch.tensor(json_data['label_ids'], dtype=torch.float)
        all_input_lengths = torch.tensor(json_data['input_length'], dtype=torch.int)

        #datasets = TensorDataset(all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths)

        return all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths

    def line_mapper(self, line):
        
        json_data = json.loads(line.strip())
        datasets = self.preprocess(json_data)

        return datasets


    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.line_mapper, file_itr)
        return mapped_itr

def get_iterable_dataset_loaders(filename, batch_size=8):
    dataset = TextIterableDataset(filename)
    input_dataloader = DataLoader(dataset, batch_size=batch_size)
    return input_dataloader

class TextDataset(Dataset):
    def __init__(self, filename, total_entries):
        self.filename = filename
        self.total_entries = total_entries

    def preprocess(self, json_data):
        all_input_page_ids = torch.tensor(json_data['page_id'], dtype=torch.int)
        all_input_ids = torch.tensor(json_data['input_ids'], dtype=torch.long)
        all_input_mask = torch.tensor(json_data['input_mask'], dtype=torch.long)
        all_label_ids = torch.tensor(json_data['label_ids'], dtype=torch.float)
        all_input_lengths = torch.tensor(json_data['input_length'], dtype=torch.int)

        #datasets = TensorDataset(all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths)

        return all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths

    def line_mapper(self, line):
        json_data = json.loads(line.strip())
        datasets = self.preprocess(json_data)

        return datasets

    def __getitem__(self, idx):
        line = linecache.getline(self.filename, idx+1)
        return self.line_mapper(line)

    def __len__(self):
        return self.total_entries

def get_dataset_loaders(filename, count, batch_size=8, num_threads=0):
    dataset = TextDataset(filename, count)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads)
    return input_dataloader

def get_datasets(input_features):
    all_input_page_ids = torch.tensor([f.guid for f in input_features], dtype=torch.int)
    all_input_ids = torch.tensor([f.input_ids for f in input_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in input_features], dtype=torch.long)
    #all_segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in input_features], dtype=torch.float)
    all_input_lengths = torch.tensor([f.input_length for f in input_features], dtype=torch.int)
    dataset = TensorDataset(all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths)

    return dataset

def get_datasetloaders(input_features, batch_size=8):
    all_input_page_ids = torch.tensor([f[4] for f in input_features], dtype=torch.int)
    all_input_ids = torch.tensor([f[0] for f in input_features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in input_features], dtype=torch.long)
    #all_segment_ids = torch.tensor([f.segment_ids for f in input_features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in input_features], dtype=torch.float)
    all_input_lengths = torch.tensor([f[3] for f in input_features], dtype=torch.long)
    dataset = TensorDataset(all_input_page_ids, all_input_ids, all_input_mask, all_label_ids, all_input_lengths)

    #data_sampler = RandomSampler(dataset)
    input_dataloader = DataLoader(dataset, batch_size=batch_size)
    return input_dataloader
