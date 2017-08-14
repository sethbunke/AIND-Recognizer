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

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        # TODO implement model selection based on DIC scores
        print('Now running DIC Selector for word= ',self.this_word)

        best_model = GaussianHMM()
        best_score = float("-inf")

            # Incrementally test models with one more component each time
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                print('Processing component ', n, 'building model')
                # Create model using n components and fit with data for this word then score it
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                print('Model built..now to score it')
                score_ThisWord = model.score(self.X, self.lengths)  # score the model on this word
                print('This word score=', score_ThisWord,'Next calcualte how the model score on all the other words...')

                #Score the same model for all other words
                AvgScore_OtherWords = self.GetScoreforOtherWords(model)
                print('Other words  score=', AvgScore_OtherWords)

                #DIC is the difference between score_this word and aver score other words
                DIC_score = score_ThisWord - AvgScore_OtherWords
                print('DIC Score and current best_score', DIC_score, best_score)

                # Store best DIC score (highest) and corresponding best model
                if DIC_score > best_score:
                    best_score = BIC_score
                    best_model = model
                    print('New best_score', best_score)

            except:
                pass
                #print('Error')

        # Return model
        return best_model

    def GetScoreforOtherWords(self, model):
        #Loop round array of words processing all words expect the current one
        print('Now caculating average score for the model across the ther words...')
        word_count = 0
        total_score = 0
        for word, features in self.hwords.items(): #list of words and their corresponding data sets
            if self.this_word != word:
                X, lengths = features
                try:
                    score_word = model.score(X, lengths)  # score the model on word
                    total_score = total_score + score_word
                    word_count = word_count + 1
                except: #catch errors that may be generated
                    pass
        average_score = total_score/word_count
        return average_score



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        print('my test')
        best_score = float('-inf')
        best_model = GaussianHMM()
        n_splits = 3

        if len(self.sequences) < n_splits:
            n_splits = len(self.sequences)

        split_method = KFold(n_splits) 
        #split_method = KFold()        
        #num_hidden_states = self.max_n_components - self.min_n_components
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                #X, lengths = cv_train_idx.get_word_Xlengths(self.words)
                X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(X_train, lengths_train)
                try:
                    log_L = model.score(X_test, lengths_test)
                    if log_L > best_score:
                        best_score = log_L
                        best_model = model
                except ValueError:
                    pass
                    #print('exception in SelectorCV')
                
        return best_model
