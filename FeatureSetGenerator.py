
from nltk.tokenize import word_tokenize


# Generate labelled featureset (All labels are true).
def baseline(content, labels, verbose=False):
    labeled_featuresets = []
    l = len(content)
    for i in range(len(content)):
        __showProgress(i, l, verbose)
        tokens = word_tokenize(content[i])

        # Construct feature set.
        feature_set = {}
        for t in tokens:
            feature_set[t] = True
        labeled_featuresets.append((feature_set, labels[i]))
    return labeled_featuresets


# Generate labelled featureset.
def labeled(content, labels, preprocesser, verbose=False):
    labeled_featuresets = []
    l = len(content)
    for i in range(len(content)):
        __showProgress(i, l, verbose)
        tokens = preprocesser.preprocess_sentence(content[i])

        # Construct feature set.
        feature_set = {}
        for t in tokens:
            feature_set[t] = t in preprocesser.corpus
        labeled_featuresets.append((feature_set, labels[i]))
    return labeled_featuresets


# Generate feature set with no labels.
def non_labeled(content, preprocesser, verbose=False):
    labeled_featuresets = []
    l = len(content)
    for i in range(l):
        __showProgress(i, l, verbose)
        tokens = preprocesser.preprocess_sentence(content[i])

        # Construct feature set.
        feature_set = {}
        for t in tokens:
            feature_set[t] = t in preprocesser.corpus
        labeled_featuresets.append(feature_set)
    return labeled_featuresets


# Displays feedback for the user while featuresets are generated.
def __showProgress(i, l, verbose=False):
    if verbose and (i % 5000 == 0 or i == l-1):
        print(str(round(i * 100 / l, 3)) + "%")
