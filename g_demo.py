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
vectorizer = CountVectorizer(max_features=2000, stop_words='english', binary=False)
corpus_bow = vectorizer.fit_transform(newsgroups.data).toarray()
vocab_size = corpus_bow.shape[1]

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
import os

class ETM(nn.Module):
    def __init__(self, vocab_size, num_topics, embedding_dim, hidden_dim, pretrained_embeddings=None):
        super(ETM, self).__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics

        # Encoder
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization layer
        self.fc_mu = nn.Linear(hidden_dim, num_topics)
        self.fc_log_sigma = nn.Linear(hidden_dim, num_topics)

        # Decoder
        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embedding_dim))
        if pretrained_embeddings is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))  # Normalize before ReLU
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, theta):
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.weight.T), dim=-1)
        return torch.matmul(theta, beta)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)
        recon_x = self.decode(theta)
        return recon_x, mu, log_sigma, theta

def load_glove_embeddings(glove_path, vocab, embedding_dim):
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove_dict[word] = vector

    embedding_matrix = np.random.uniform(-0.05, 0.05, (len(vocab), embedding_dim))
    for i, word in enumerate(vocab):
        if word in glove_dict:
            embedding_matrix[i] = glove_dict[word]

    return torch.tensor(embedding_matrix, dtype=torch.float32)

num_topics = 10
hidden_dim = 300
# Initialize model and optimizer
device = 'cpu'
# Path to GloVe file
glove_path = "D:/code/topic_modeling/glove.6B.300d.txt"

# Load GloVe embeddings
vocab = vectorizer.get_feature_names_out()
embedding_dim = 300  # GloVe embedding dimension
glove_embeddings = load_glove_embeddings(glove_path, vocab, embedding_dim)


import wandb

# Initialize W&B
wandb.init(project="etm-topic-modeling", name="ETM-with-Glove", config={
    "num_topics": num_topics,
    "hidden_dim": hidden_dim,
    "embedding_dim": embedding_dim,
    "batch_size": 32,
    "learning_rate": 0.001,
    "num_epochs": 50,
})

def train_etm(model, dataloader, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    model.train()
    config = wandb.config

    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_div = 0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for batch in tepoch:
                batch = batch.to(device)
                optimizer.zero_grad()

                # Forward pass
                recon_x, mu, log_sigma, theta = model(batch)

                # Compute losses
                recon_loss = -torch.sum(batch * torch.log(recon_x + 1e-9)) / batch.size(0)
                kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp()) / batch.size(0)
                loss = recon_loss + kl_div

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update metrics
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_div += kl_div.item()

                tepoch.set_postfix(loss=total_loss / len(dataloader))

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "total_loss": total_loss / len(dataloader),
            "reconstruction_loss": total_recon_loss / len(dataloader),
            "kl_divergence": total_kl_div / len(dataloader),
        })

    print("Training complete!")
    wandb.finish()

# Train the model


def etm_inference(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    topic_proportions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, _, _, theta = model(batch)
            topic_proportions.append(theta.cpu().numpy())

    return np.vstack(topic_proportions)
    
# Model hyperparameters

# Initialize ETM with pre-trained GloVe embeddings
etm = ETM(vocab_size, num_topics, embedding_dim, hidden_dim, pretrained_embeddings=glove_embeddings)

# Train the model
# optimizer = torch.optim.Adam(etm.parameters(), lr=0.001)
# train_etm(etm, dataloader, optimizer, num_epochs=50, device=device)

# Inference
inference_loader = DataLoader(dataset, batch_size=32, shuffle=False)
topic_proportions = etm_inference(etm, inference_loader, device=device)

# Print the topic proportions for the first document
print("Topic proportions for the first document:", topic_proportions[0])
def get_topic_word_distributions(model):
    with torch.no_grad():
        beta = F.softmax(torch.matmul(model.topic_embeddings, model.word_embeddings.weight.T), dim=-1)
    return beta.cpu().numpy()

# Extract word distributions for each topic
beta = get_topic_word_distributions(etm)

# Display top words for each topic
vocab = vectorizer.get_feature_names_out()
num_top_words = 10  # Number of top words to display for each topic
for topic_idx, topic in enumerate(beta):
    top_words = [vocab[i] for i in topic.argsort()[-num_top_words:][::-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
# Topic proportions for all documents
optimizer = torch.optim.Adam(etm.parameters(), lr=wandb.config.learning_rate)
train_etm(etm, dataloader, optimizer, num_epochs=wandb.config.num_epochs, device=device)

# Log topic proportions after training
doc_topic_proportions = etm_inference(etm, inference_loader, device=device)
wandb.log({"topic_proportions": doc_topic_proportions})
# doc_topic_proportions = etm_inference(etm, inference_loader, device=device)


# Display topic proportions for the first document
# print("Document 1 Topic Proportions:", doc_topic_proportions[0])
