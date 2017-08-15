import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    #raise NotImplementedError
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    hwords = test_set.get_all_Xlengths()
    try:
        for word_id in range(0, len(test_set.get_all_sequences())):
            words_prob = {}
            best_score = float('-Inf')
            guess_word = None
            X, lengths = hwords[word_id]

            for word, model in models.items():
                try:
                    score = model.score(X, lengths)
                    words_prob[word] = score

                    if score > best_score:
                        guess_word = word
                        best_score = score
                except:
                    pass
        
            probabilities.append(words_prob)
            guesses.append(guess_word)

    except:
        print('outer exception')
        pass
    # return probabilities, guesses
    return probabilities, guesses
