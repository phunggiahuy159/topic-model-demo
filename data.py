import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
import numpy as np

# Fetch the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# print(newsgroups.data[0])
vectorizer = CountVectorizer(max_features=2000, stop_words='english', binary=False)
corpus_bow = vectorizer.fit_transform(newsgroups.data).toarray()
vocab_size = corpus_bow.shape[0]
print(vocab_size)

# Convert to PyTorch Dataset
class BoWDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]
dataset = BoWDataset(corpus_bow)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
