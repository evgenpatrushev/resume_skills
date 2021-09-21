from collections import Counter
import numpy as np
from scipy import spatial
import pickle

rng = np.random.default_rng(123)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_batch(data, size, prob):
    x = rng.choice(data, size, p=prob)
    return x[:, 0], x[:, 1]


class Embedding:
    """
    Word embedding model.

    Parameters
    ----------
    N: int
        Number of unique words in the vocabulary
    D: int
        Dimension of the word vector embedding
    """

    def __init__(self, N, D):
        self.N = N
        self.D = D

        self.ctx = None  # Used to store values for backpropagation

        self.U = None
        self.V = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        init weight matrices U and V, dimension (D, N) and (N, D) respectively
        """
        self.ctx = None
        self.U = np.random.normal(0, np.sqrt(6. / (self.D + self.N)), (self.D, self.N))
        self.V = np.random.normal(0, np.sqrt(6. / (self.D + self.N)), (self.N, self.D))

    def one_hot(self, x):
        """
        Given a vector returns a matrix with rows corresponding to one-hot encoding.

        Parameters
        ----------
        x: array
            M-dimensional vector containing integers from [0, N]

        Returns
        -------
        one_hot: array
            (N, M) matrix, each column is N-dimensional one-hot encoding of elements from x
        """

        one_hot = np.zeros((self.N, x.shape[0]))
        for i, x_i in enumerate(x):
            one_hot[x_i, i] = 1

        assert one_hot.shape == (self.N, x.shape[0]), 'Incorrect one-hot embedding shape'
        return one_hot

    def softmax(self, x, axis):
        """
        Parameters
        ----------
        x: array
            A non-empty matrix of any dimension
        axis: int
            Dimension on which softmax is performed

        Returns
        -------
        y: array
            Matrix of same dimension as x with softmax applied to 'axis' dimension
        """

        # numerically stable version of softmax

        # if axis==0 we apply by rows, if axis==1 we apply by columns
        y = np.zeros(x.shape)
        if axis == 0:
            for i in range(x.shape[0]):
                x_i = np.array(x[i])
                x_i = x_i - np.max(x_i)
                y[i] = np.exp(x_i) / np.sum(np.exp(x_i))

        elif axis == 1:
            for j in range(x.shape[1]):
                x_j = np.array(x[:, j])
                x_j = x_j - np.max(x_j)
                y[:, j] = np.exp(x_j) / np.sum(np.exp(x_j))

        assert x.shape == y.shape, 'Output should have the same shape is input'
        return y

    def loss(self, y, prob):
        """
        Parameters
        ----------
        y: array
            (N, M) matrix of M samples where columns are one-hot vectors for true values
        prob: array
            (N, M) column of M samples where columns are probability vectors after softmax

        Returns
        -------
        loss: int
            Cross-entropy loss
        """

        prob = np.clip(prob, 1e-8, None)

        loss = -1 / y.shape[1] * (y * np.log(prob)).sum()

        assert isinstance(loss, float), 'Loss must be a scalar'
        return loss

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: array
            M-dimensional vector containing integers from [0, N]
        y: array
            Output words, same dimension and type as 'x'
        learning_rate: float
            A positive scalar determining the update rate

        Returns
        -------
        loss: float
        d_U: array
        d_V: array
        """

        # Input transformation

        x = self.one_hot(x)
        y = self.one_hot(y)

        # Forward propagation

        embedding = self.U[:, np.argmax(x, axis=0)]
        logits = np.dot(self.V, embedding)
        prob = self.softmax(logits, axis=1)

        assert embedding.shape == (self.D, x.shape[1])
        assert logits.shape == (self.N, x.shape[1])
        assert prob.shape == (self.N, x.shape[1])

        # Save values for backpropagation
        self.ctx = (embedding, logits, prob, x, y)

        # Loss calculation
        loss = self.loss(y, prob)

        return loss

    def backward(self):
        """
        Returns gradient of U and V.

        Returns
        -------
        d_V: array
        d_U: array
        """

        embedding, logits, prob, x, y = self.ctx

        d_V = np.dot((prob - y), embedding.T)
        d_U = np.dot(np.dot((prob - y).T, self.V).T, x.T)

        assert d_V.shape == (self.N, self.D)
        assert d_U.shape == (self.D, self.N)

        return {'V': d_V, 'U': d_U}


class Optimizer:
    """
    Stochastic gradient descent with momentum optimizer.

    Parameters
    ----------
    model: object
    learning_rate: float
    momentum: float (optional)(default: 0)
    """

    def __init__(self, model, learning_rate, momentum=0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.previous = None  # Previous gradients

    def _init_previous(self, grad):
        # Initialize previous gradients to zero
        self.previous = {k: np.zeros_like(v) for k, v in grad.items()}

    def step(self, grad):
        if self.previous is None:
            self._init_previous(grad)

        for name, dw in grad.items():
            dw_prev = self.previous[name]
            w = getattr(self.model, name)

            #             Given weight w, previous gradients dw_prev and current
            #             gradients dw, performs an update of weight w.

            dw_new = self.momentum * dw_prev + self.learning_rate * dw
            w_new = w - dw_new

            self.previous[name] = dw_new
            setattr(self.model, name, w_new)


def get_loss(model, old, variable, epsilon, x, y, i, j):
    np.random.seed(123)
    model.reset_parameters()  # reset weights

    delta = np.zeros_like(old)
    delta[i, j] = epsilon

    setattr(model, variable, old + delta)  # change one weight by a small amount
    loss = model.forward(x, y)

    return loss


def test_embeddings(emb_matrix, word_to_ind, ind_to_word, triplets=None):
    if triplets is None:
        triplets = [['machine', 'learning', 'learning'], ['machine', 'learning', 'python'],
                    ['deep', 'learning', 'javascript'], ]

    for triplet in triplets:
        a, b, d = triplet

        """
        Returns

        Example: Paris (a) is to France (b) as _____ (c) is to Germany (d)

        -------
        result: array
            The embedding vector for word (c): w_a - w_b + w_d
        """

        w_a, w_b, w_d = emb_matrix[word_to_ind[a]], emb_matrix[word_to_ind[b]], emb_matrix[word_to_ind[d]]
        result = w_a - w_b + w_d

        distances = [spatial.distance.cosine(x, result) for x in emb_matrix]
        candidates = [ind_to_word[i] for i in np.argsort(distances)]
        candidates = [x for x in candidates if x not in [a, b, d]][:5]

        print(f'`{a}` is to `{b}` as [{", ".join(candidates)}] is to `{d}`')


def create_embeddings(df, run_test=True, test_triplets=None):
    def get_window(sentence, window_size):
        """
        Iterate over all the sentences
        Take all the words from [i - window_size) to (i + window_size] and save them to pairs

        Parameters
        ----------
        sentence: list
            List of sentences. Sentence is a list of words (str)
        window_size: int
            Positive scalar

        Returns
        -------
        pairs: list
            A list of tuple (word index, word index from its context)
        """

        pairs = []
        for ind_w in range(len(sentence)):
            window_ind = [i for i in range(ind_w - window_size, ind_w + window_size + 1)
                          if i != ind_w and 0 <= i <= len(sentence) - 1]
            for win_ind_i in window_ind:
                pairs.append((word_to_ind[sentence[ind_w]], word_to_ind[sentence[win_ind_i]]))

        return pairs

    # create vocabulary
    vocabulary = []
    for resume in df['Resume']:
        for sentence in resume:
            for w in sentence:
                vocabulary.append(w)

    # use only 500 most common words (hyperparameter)
    vocabulary, counts = zip(*Counter(vocabulary).most_common(200))

    corpus = []  # list of all sentences
    for resume in df['Resume']:
        for sentence in resume:
            corpus.append(sentence)

    # use "unk" as unknown words for non-vocabulary words
    corpus = [[w if w in vocabulary else 'unk' for w in sentence] for sentence in corpus]
    vocabulary += ('unk',)  # Add "unk" to vocabulary
    counts += (sum([w == 'unk' for s in corpus for w in s]),)  # Add count for "unk"

    VOCABULARY_SIZE = len(vocabulary)
    EMBEDDING_DIM = 64  # (hyperparameter)

    print('vocabulary len', VOCABULARY_SIZE)

    # Dictionaries for converting words to index, index to words and index to frequency of this word
    word_to_ind = {w: vocabulary.index(w) for w in vocabulary}
    ind_to_word = {i: vocabulary[i] for i in range(len(vocabulary))}, '}'
    ind_to_freq = {i: counts[i] for i in range(len(vocabulary))}

    data = []
    for x in corpus:
        data += get_window(x, window_size=3)
    data = np.array(data)

    print('First 5 pairs:', data[:5].tolist())
    print('First 5 pairs words:', [[ind_to_word[index] for index in window] for window in data[:5].tolist()])
    print('Total number of pairs:', data.shape[0])

    probabilities = [1 - np.sqrt(1e-3 / ind_to_freq[x]) for x in data[:, 0]]
    probabilities /= np.sum(probabilities)

    # training embeddings ------------------------------------------------------------------------------------

    model = Embedding(N=VOCABULARY_SIZE, D=EMBEDDING_DIM)
    optim = Optimizer(model, learning_rate=1e-3, momentum=0.5)

    losses = []

    MAX_ITERATIONS = 30000
    PRINT_EVERY = 1000
    BATCH_SIZE = 1000

    for i in range(MAX_ITERATIONS):
        x, y = get_batch(data, BATCH_SIZE, probabilities)

        loss = model.forward(x, y)
        grad = model.backward()
        optim.step(grad)

        assert not np.isnan(loss)

        losses.append(loss)

        if (i + 1) % PRINT_EVERY == 0:
            print(f'Iteration: {i + 1}, Avg. training loss: {np.mean(losses[-PRINT_EVERY:]):.4f}')

    emb_matrix = model.U.T
    np.save('emb_matrix', emb_matrix)

    save_obj(word_to_ind, 'word_to_ind')
    save_obj(ind_to_word, 'ind_to_word')
    save_obj(ind_to_freq, 'ind_to_freq')
    save_obj(vocabulary, 'vocabulary')

    if run_test:
        test_embeddings(emb_matrix, word_to_ind, ind_to_word, test_triplets)

    return emb_matrix, word_to_ind, ind_to_word, ind_to_freq, vocabulary


if __name__ == '__main__':
    embedding_matrix = np.load('emb_matrix.npy')
