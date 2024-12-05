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
class ETM(nn.Module):
    def __init__(self, vocab_size, num_topics, embedding_dim, hidden_dim):
        super(ETM, self).__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        # Encoder
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_topics)
        self.fc_log_sigma = nn.Linear(hidden_dim, num_topics)

        # Decoder
        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, embedding_dim))
        self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon

    def decode(self, theta):
        beta = F.softmax(torch.matmul(self.topic_embeddings, self.word_embeddings.T), dim=-1)
        return torch.matmul(theta, beta)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        theta = F.softmax(self.reparameterize(mu, log_sigma), dim=-1)
        recon_x = self.decode(theta)
        return recon_x, mu, log_sigma, theta

def train_etm(model, dataloader, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
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
                total_loss += loss.item()

                tepoch.set_postfix(loss=total_loss / len(dataloader))
    print("Training complete!")
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
num_topics = 10
embedding_dim = 150
hidden_dim = 256

# Initialize model and optimizer
device = 'cpu'
etm = ETM(vocab_size, num_topics, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(etm.parameters(), lr=0.001)
# Train the model
train_etm(etm, dataloader, optimizer, num_epochs=10, device=device)
# Inference
inference_loader = DataLoader(dataset, batch_size=64, shuffle=False)
topic_proportions = etm_inference(etm, inference_loader, device=device)

# Print the topic proportions for the first document
print("Topic proportions for the first document:", topic_proportions[0])
def get_topic_word_distributions(model):
    """
    Extract word distributions for each topic.
    Returns a matrix (num_topics x vocab_size) where each row represents
    the word probabilities for a topic.
    """
    with torch.no_grad():
        beta = F.softmax(torch.matmul(model.topic_embeddings, model.word_embeddings.T), dim=-1)
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
doc_topic_proportions = etm_inference(etm, inference_loader, device=device)

# Display topic proportions for the first document
print("Document 1 Topic Proportions:", doc_topic_proportions[0])
