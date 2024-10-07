
import numpy as no
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import lxml

cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv(
    "train_NLP.csv",
    header=None,
    names=cols,
    engine="python",
    encoding= "latin1"
)
test_data = pd.read_csv(
    "test_NLP.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)

data = train_data.copy()
#Limpieza
data.drop(["id","date","query","user"],
          axis = 1,
          inplace = True)

def clean_tweet(tweet):
  tweet = BeautifulSoup(tweet, "lxml").get_text()
  tweet = re.sub(r"@[A-Za-z0-9]+", " ", tweet)
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", " ", tweet)
  tweet = re.sub(r"[^A-Za-z.!?']", " ", tweet)
  tweet = re.sub(r" +", " ", tweet)
  return tweet

data_clean = [clean_tweet(tweet) for tweet in data.text]

class SkipGramaData(Dataset):
  def __init__(self, corpus, window_size = 2):
    super().__init__()
    self.window = window_size
    self.corpus = corpus
    
    self.vocab = list(set([token.lower() for sentence in self.corpus for token in sentence.split()]))
    #diccionario que va a tener como llave la palabra y como indice el idx
    self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}

    self.data = self.gen_dataset()
  
  def gen_dataset(self):
    data = []
    for sentence in self.corpus:
      text = sentence.lower().split()
      for center_idx, center_word in enumerate(text):
        for offset in range(-self.window, self.window + 1):
          context_idx = center_idx + offset
          if context_idx < 0 or context_idx >= len(text) or context_idx == center_idx: continue
          context_word = text[context_idx]
          data.append((self.word2idx[center_word], self.word2idx[context_word]))
    return data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]

class SkipGram(nn.Module):
  def __init__(self, vocab_size, embedding_size):
    super().__init__()
    self.embed_layer = nn.Linear(vocab_size, embedding_size, bias = False)
    self.output_layer = nn.Linear(embedding_size, vocab_size)
    #self.embed = nn.Embedding(vocab_size, embedding_size)

  def forward(self, x):
    embed = self.embed_layer(x)
    output = self.output_layer(embed)
    return output

def train_skipgram(model, loss_fn, optimizer, data_loader, device, epochs=100):
  #model.train()
  model_train = model.to(device)
  for epoch in range(epochs):
    total_loss=0
    for center, context in data_loader:
      center_vector = torch.zeros(len(data_loader.dataset.vocab))
      center_vector[center] = 1.0
      center_vector = center_vector.unsqueeze(0).to(device)
      scores = model(center_vector).to(device)

      loss = loss_fn(scores, torch.tensor([context]).to(device))

      total_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print(f"Epoch {epoch}, Loss: {total_loss/len(data_loader)}")
    

dataset = SkipGramaData(data_clean[:150])
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

model = SkipGram(len(dataset.vocab), embedding_size=300)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_skipgram(model, loss_fn, optimizer, data_loader, device, epochs=1000)

def plot_embedings(embeddings, word2idx):
  num_samples = len(word2idx)
  tsne = TSNE(n_components=2, random_state=0)
  vectors = tsne.fit_transform(embeddings)
  plt.figure(figsize=(20, 10))
  for word, idx in word2idx.items():
    plt.scatter(vectors[idx, 0], vectors[idx, 1])
    plt.annotate(word, xy= (vectors[idx, 0], vectors[idx, 1]), xytext=(5, 2), 
                 textcoords="offset points", ha="right",va = "bottom")
  plt.title("Word Embeddings Visualized using t-SNE")
  plt.xlabel("t-SNE Dimension 1")
  plt.ylabel("t-SNE Dimension 2")
  plt.grid(True)
  plt.show()
  
embeddings = model.embed_layer.weight.detach().cpu().numpy().T
plot_embedings(embeddings, dataset.word2idx)