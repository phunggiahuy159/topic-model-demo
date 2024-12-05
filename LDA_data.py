import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from tqdm import tqdm

# Fetch the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
vectorizer = CountVectorizer(max_features=2000, stop_words='english', binary=False)
corpus_bow = vectorizer.fit_transform(newsgroups.data).toarray()
vocab_size = corpus_bow.shape[1]

# PyTorch Dataset
class BoWDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = BoWDataset(corpus_bow)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# LDA with Variational Inference
class LDA_VI:
    def __init__(self, num_topics, vocab_size, alpha=0.1, eta=0.1, max_iter=100, tol=1e-4):
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.eta = torch.tensor(eta, dtype=torch.float32)
        self.max_iter = max_iter
        self.tol = tol

        # Global topic-word distribution
        self.beta = torch.rand((num_topics, vocab_size), requires_grad=False)
        self.beta = self.beta / self.beta.sum(dim=1, keepdim=True)  # Normalize to probabilities

    def e_step(self, batch):
        """
        E-step: Update gamma (document-topic distributions) and phi (word-topic assignments).
        """
        batch_size, vocab_size = batch.shape

        # Initialize gamma for the batch
        gamma = torch.ones((batch_size, self.num_topics)) * self.alpha
        gamma.requires_grad = False  # No gradient, variational parameter

        # Compute phi for the batch
        phi = torch.ones((batch_size, vocab_size, self.num_topics)) / self.num_topics

        for _ in range(self.max_iter):
            # Update phi
            log_theta = torch.digamma(gamma) - torch.digamma(gamma.sum(dim=1, keepdim=True))
            log_beta = torch.log(self.beta + 1e-10)  # Add small value to avoid log(0)
            new_phi = torch.exp(log_theta.unsqueeze(1) + log_beta.T.unsqueeze(0))
            new_phi /= new_phi.sum(dim=2, keepdim=True)

            # Update gamma
            new_gamma = self.alpha + torch.einsum('bnk,bn->bk', new_phi, batch)

            # Check for convergence
            if torch.norm(new_phi - phi) < self.tol and torch.norm(new_gamma - gamma) < self.tol:
                break

            phi = new_phi
            gamma = new_gamma

        return gamma, phi

    def m_step(self, data_loader):
        """
        M-step: Update beta (topic-word distributions) using aggregated phi values.
        """
        beta = torch.zeros((self.num_topics, self.vocab_size))

        for batch in data_loader:
            _, phi = self.e_step(batch)
            beta += torch.einsum('bnk,bn->kv', phi, batch)

        # Add Dirichlet prior and normalize
        self.beta = beta + self.eta
        self.beta /= self.beta.sum(dim=1, keepdim=True)

    def train(self, data_loader):
        """
        Train the LDA model using variational inference.
        """
        for iteration in tqdm(range(self.max_iter), desc="Training LDA"):
            self.m_step(data_loader)

    def get_document_topic_distribution(self, data_loader):
        """
        Compute document-topic distribution (theta) for all documents.
        """
        thetas = []
        for batch in data_loader:
            gamma, _ = self.e_step(batch)
            theta = gamma / gamma.sum(dim=1, keepdim=True)
            thetas.append(theta)
        return torch.cat(thetas, dim=0)

    def get_topic_word_distribution(self):
        """
        Return topic-word distribution (beta).
        """
        return self.beta

# Initialize and train the LDA model
num_topics = 10
lda = LDA_VI(num_topics=num_topics, vocab_size=vocab_size, alpha=0.1, eta=0.1, max_iter=10)
lda.train(dataloader)

# Get document-topic and topic-word distributions
document_topic_dist = lda.get_document_topic_distribution(dataloader)
topic_word_dist = lda.get_topic_word_distribution()

print("Document-Topic Distributions (Theta):")
print(document_topic_dist)

print("Topic-Word Distributions (Beta):")
print(topic_word_dist)
# Function to get the top words for each topic
def get_top_words_per_topic(beta, vectorizer, top_n=10):
    """
    Extract the top words for each topic.
    Args:
        beta: Topic-word distribution (K x V matrix).
        vectorizer: The CountVectorizer used for the corpus (provides the vocabulary).
        top_n: Number of top words to extract per topic.
    Returns:
        A list of lists, where each inner list contains the top words for a topic.
    """
    vocab = np.array(vectorizer.get_feature_names_out())
    top_words = []
    for topic_idx, topic_dist in enumerate(beta):
        top_word_indices = topic_dist.argsort()[-top_n:][::-1]  # Indices of top N words
        top_words.append(vocab[top_word_indices])
    return top_words

# Retrieve top words for each topic
top_words = get_top_words_per_topic(topic_word_dist.detach().numpy(), vectorizer, top_n=10)

# Display the results
for topic_idx, words in enumerate(top_words):
    print(f"Topic {topic_idx + 1}: {', '.join(words)}")
