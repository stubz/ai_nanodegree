import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('inf') # we want the minimum BIC
        best_model = self.base_model(self.n_constant)
        # we try different number of hidden components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                n_params = n_components * n_components + 2 * n_components * len(self.X[0]) - 1
                # evaluate the log-likelihood on the test data
                logL = model.score(self.X, self.lengths)
                BIC = -2.0*logL + np.log(len(self.X))*n_params
                if BIC < best_score:
                    best_score = BIC
                    best_model = model
            except:
                pass
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float('inf') # we want the minimum BIC
        best_model = self.base_model(self.n_constant)

        def remove_key(d, key):
            # create a new dictionary that excludes one word
            new_dict = dict(d)
            del new_dict[key]
            return new_dict

        rest_words = remove_key(self.hwords, self.this_word)
        # we try different number of hidden components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(self.X, self.lengths)
                n_params = n_components * n_components + 2 * n_components * len(self.X[0]) - 1
                # evaluate the log-likelihood on the test data
                logL = model.score(self.X, self.lengths)
                rest_scores = [model.score(*rest_words[word]) for word in rest_words]
                DIC = logL - np.mean(rest_scores)
                if DIC < best_score:
                    best_score = DIC
                    best_model = model
            except:
                pass
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    model = SelectorCV(sequences, Xlengths, word,
                    min_n_components=2, max_n_components=15, random_state = 14).select()
    '''



    def select(self):
        best_score = float('-inf')
        best_model = self.base_model(self.n_constant)
        n_splits = min(len(self.lengths), 5) # the number of K-Fold
        split_method = KFold(n_splits = n_splits)
        # we try different number of hidden components
        for n_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # calculate the log-likelihood for each K-Fold
                logL_sum = 0
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                    # fit a model on the training data
                    model = GaussianHMM(n_components=n_components, random_state=self.random_state, n_iter=1000).fit(X_train, length_train)
                    # evaluate the log-likelihood on the test data
                    logL = model.score(X_test, length_test)
                    logL_sum += logL

                # Because we have K-Fold, we divide the sum of the log-likelihood of each K-fold
                # by the number of splits

                if logL_sum / float(n_splits) > best_score:
                    best_score = logL_sum
                    best_model = model

            except:
                best_model = GaussianHMM(n_components=n_components, n_iter=1000).fit(self.X, self.lengths)
        return best_model
