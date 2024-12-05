import numpy as np
from scipy.special import digamma, gammaln

class LDA_VI:
    def __init__(self, num_topics, alpha, eta, max_iter=100, tol=1e-4):
        self.num_topics = num_topics  # Number of topics (K)
        self.alpha = alpha            # Dirichlet prior for document-topic distribution
        self.eta = eta                # Dirichlet prior for topic-word distribution
        self.max_iter = max_iter      # Maximum number of iterations
        self.tol = tol                # Convergence tolerance

    def initialize(self, documents, vocab_size):
        """
        Initialize parameters for variational inference.
        """
        self.vocab_size = vocab_size
        self.documents = documents
        self.num_docs = len(documents)

        # Variational parameters
        self.gamma = np.random.gamma(100., 1. / 100., (self.num_docs, self.num_topics))  # Document-topic parameters
        self.phi = []  # Word-topic responsibilities for each document

        # Initialize topic-word distribution (global parameter)
        self.beta = np.random.gamma(100., 1. / 100., (self.num_topics, self.vocab_size))

    def e_step(self):
        """
        E-step: Update phi (word-topic responsibilities) and gamma (document-topic distributions).
        """
        for d, doc in enumerate(self.documents):
            word_ids, word_counts = doc  # word_ids: unique words, word_counts: their frequencies
            num_words = len(word_ids)

            # Initialize phi for this document
            phi_d = np.random.dirichlet(np.ones(self.num_topics), num_words)

            # Update phi and gamma iteratively
            gamma_d = self.alpha + np.sum(phi_d * word_counts[:, None], axis=0)
            for _ in range(self.max_iter):
                # Compute the expected log of theta and beta
                log_theta_d = digamma(gamma_d) - digamma(np.sum(gamma_d))
                log_beta = digamma(self.beta[:, word_ids]) - digamma(np.sum(self.beta, axis=1)[:, None])

                # Update phi
                new_phi_d = np.exp(log_theta_d + log_beta.T)
                new_phi_d /= np.sum(new_phi_d, axis=1, keepdims=True)

                # Update gamma
                new_gamma_d = self.alpha + np.sum(new_phi_d * word_counts[:, None], axis=0)

                # Check for convergence
                if np.linalg.norm(new_phi_d - phi_d) < self.tol and np.linalg.norm(new_gamma_d - gamma_d) < self.tol:
                    break

                phi_d = new_phi_d
                gamma_d = new_gamma_d

            # Save updated parameters for the document
            self.phi.append(phi_d)
            self.gamma[d] = gamma_d

    def m_step(self):
        """
        M-step: Update beta (topic-word distributions).
        """
        # Reset beta
        self.beta = np.zeros((self.num_topics, self.vocab_size))

        for d, doc in enumerate(self.documents):
            word_ids, word_counts = doc
            self.beta[:, word_ids] += np.dot(self.phi[d].T, word_counts)

        # Add Dirichlet prior
        self.beta += self.eta

    def train(self, documents, vocab_size):
        """
        Train the LDA model using variational inference.
        """
        self.initialize(documents, vocab_size)

        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}/{self.max_iter}")
            
            # E-step: Update phi and gamma
            self.e_step()
            
            # M-step: Update beta
            self.m_step()

            # Compute ELBO for convergence (optional)
            # ELBO computation would involve calculating the full joint probability
            # and variational entropy, which can be added for monitoring.

        # Normalize beta to obtain probabilities
        self.beta /= np.sum(self.beta, axis=1, keepdims=True)

    def get_document_topic_distribution(self):
        """
        Return the document-topic distribution (theta).
        """
        theta = self.gamma / np.sum(self.gamma, axis=1, keepdims=True)
        return theta

    def get_topic_word_distribution(self):
        """
        Return the topic-word distribution (phi).
        """
        return self.beta

# Toy Example
# Vocabulary: {0: "apple", 1: "banana", 2: "orange"}
documents = [
    (np.array([0, 1]), np.array([2, 1])),  # "apple apple banana"
    (np.array([1, 2]), np.array([2, 1]))   # "banana banana orange"
]

lda = LDA_VI(num_topics=2, alpha=0.1, eta=0.1, max_iter=20, tol=1e-4)
lda.train(documents, vocab_size=3)

# Print results
print("Document-Topic Distributions (Theta):")
print(lda.get_document_topic_distribution())

print("Topic-Word Distributions (Beta):")
print(lda.get_topic_word_distribution())
