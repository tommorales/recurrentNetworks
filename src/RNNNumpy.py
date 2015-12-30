__author__ = 'tmorales'

import numpy as np

from activationFuncion import softmax

class RNNNumpy:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables.
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Random initialize the network parmeters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0.
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The output at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step
        for t in np.arange(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self,x ):
        # Perform foreard propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)


    def calculate_total_lost(self, x, y):
        L = 0
        # For each sentence ...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "corrent" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_lost(x, y)/N


    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumalate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards ...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - s[t] ** 2)
            # Backpropagation through time (for at most self.bptt_truncation step)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation setp t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdW[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdU, dLdW]


    def gradiente_check(self, x, y, h=0.001, error_threshold=0.01):
        pass


    # Gradients of the parameters **********************************************************************

    # Step 1: calculate the gradients and performs the updates for one batch.

    def numpy_sdg_step(self, x, y, learning_rate):
        # calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x,y)
        # Change paremeters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    # Step 2: An outer loop that iterates throught the training set and adjusts the learning rate.
    # Outer: SGD Loop
    # - model: The RNN model instance.
    # - X_train: The training data set.
    # - y_train: The training data labels.
    # - learning_rate: Initial learning rate for SGD.
    # - nepoch: Number of times to iterate throught the complete dataset.
    # - evaluate_lost_after: Evaluate the loss after this many epochs.
    def train_with_sgd(self, model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                pass

            #
            #  Here
            #


    # ***************************************************************************************************