import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelWithLMHead, BertForMaskedLM
from transformers import PretrainedConfig
import copy
import numpy as np
import pickle

tokenizer = AutoTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')
tokenize = tokenizer.tokenize
token_to_id = tokenizer.convert_tokens_to_ids

# read the data
with open('./data/test.txt') as f:
    data = f.readlines()

# tokenize data and convert to id's
tokenized_data = [token_to_id(tokenize(s)) for s in data]
mask_id = token_to_id("[MASK]")

dataloader = []

# iterate over all the sentences and mask one word at the time
for sentence in tokenized_data:
    # initialize tensors
    length = len(sentence)

    for i in range(len(sentence)):
        # replace one of the words by mask id
        sentence_copy = copy.deepcopy(sentence)
        sentence_copy[i] = mask_id
        batch_tensor = torch.Tensor(sentence_copy)

        # construct batch data: sentences and labels
        batch_tensor = batch_tensor.int()
        batch_labels = torch.Tensor(sentence).int()
        dataloader.append((batch_tensor, batch_labels, i))

# save the dataloader
with open("data/dataloader.p", "wb") as f:
    pickle.dump(dataloader, f)








