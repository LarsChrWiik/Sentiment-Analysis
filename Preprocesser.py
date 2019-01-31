# By Lars Wiik (LW17793).

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import SnowballStemmer
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import ngrams

class Preprocesser:

    corpus = []
    words_frequency = []

    tokenizer = ToktokTokenizer()
    stopWords = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    def __init__(self, data):
        new_corpus = self.preprocess_multiple_sentences(data, has_corpus = False)
        self.words_frequency = FreqDist(new_corpus)
        self.corpus = list(set(new_corpus))

        # Write all words to file. Used for WordCould.
        # self.write_to_file(self.corpus)

    def preprocess_multiple_sentences(self, data_raw, has_corpus = False):
        new_corpus = []
        for sentence in data_raw:
            new_corpus += self.preprocess_sentence(sentence, has_corpus)
        return new_corpus

    def preprocess_sentence(self, sentence, has_corpus = True):

        # Tokenize.
        sentence_words = self.tokenizer.tokenize(sentence)

        # Convert to lowercase.
        sentence_words = [x.lower() for x in sentence_words]

        # Remove stop-words.
        #sentence_words = [x for x in sentence_words if x not in self.stopWords]

        # Remove stopwords in phrases that contain non-stop words.
        """
        without_stop_words = [x for x in sentence_words if x not in self.stopWords]
        if len(without_stop_words) > 0:
            sentence_words = [x for x in sentence_words if x not in self.stopWords]
        """

        # Remove additional words.
        #sentence_words = self.remove_additional_words(sentence_words)

        # Remove outliers.

        #if has_corpus:
            #sentence_words = self.remove_outliers(self.corpus, self.words_frequency, 1.0, 0.3)

        # Remove words that starts with numbers.
        #sentence_words = [x for x in sentence_words if not self.starts_with_number(x)]

        # Remove single characters. (mostly punctuation)
        sentence_words = [x for x in sentence_words if len(x) > 1]

        # Remove starting and ending exceptions.
        sentence_words = [x for x in sentence_words if not self.starts_or_ends_with_exceptions(x)]

        # Stemming
        sentence_words = [self.stemmer.stem(x) for x in sentence_words]

        # Pos-tag.
        """
        pos = pos_tag(sentence_words)
        for i in range(len(sentence_words)):
            sentence_words[i] += "_" + pos[i][1]
        """

        # Capture negation.
        if "not" in sentence_words:
            for i in range(len(sentence_words)):
                if sentence_words[i] not in self.stopWords:
                    sentence_words[i] = sentence_words[i] + "_NEG"

        # Capture very.
        if "very" in sentence_words:
            for i in range(len(sentence_words)):
                if sentence_words[i] not in self.stopWords:
                    sentence_words[i] = sentence_words[i] + "_VERY"

        # Lemmatization
        #sentence_words = [self.lemmatizer.lemmatize(x) for x in sentence_words]

        # Add n-grams.
        sentence_words += self.get_ngrams(
            ' '.join(sentence_words),
            2
        )

        return sentence_words

    # Used to generate word-cloud.
    def write_to_file(self, array):
        with open("corpus.txt", "w") as file:
            for w in array:
                file.write(w + "\n")

    # Generate n-grams.
    def get_ngrams(self, text, n):
        t = word_tokenize(text)
        t = [x.lower() for x in t]
        n_grams = ngrams(t, n)
        return [' '.join(grams) for grams in n_grams]

    # Uses words_frequency to remove outliers.
    def remove_outliers(self, corpus, words_freq, upper, lower):
        counts = [words_freq[x] for x in words_freq]
        counts.sort()
        indl = int(len(counts)*lower)
        if indl < 0:
            indl = 0
        indu = int(len(counts) * upper)
        if indu >= len(counts):
            indu = len(counts) - 1

        lower_bound = counts[indl]
        upper_bound = counts[indu]
        all_words_new = []
        for x in corpus:
            if words_freq[x] >= lower_bound and words_freq[x] <= upper_bound:
                all_words_new.append(x)

        return all_words_new

    # Self made stop words.
    def remove_additional_words(self, all_words):
        additional_stop_words = [
            ',', "'", '.', '`', 'n', "the", "a", "...", ":",
            ' ', 'movi', 'film', "one", "of", "and", "to", "it", "is", "in", "that", "with",
            "for", "but", "of th", "an", "this", "on", "at", "be", "by", "from", "his", "than", "s", "t", "are"
        ]
        return [x for x in all_words if x not in additional_stop_words]

    # Checks if a word starts with a number or not.
    @staticmethod
    def starts_with_number(text):
        char = text[0]
        try:
            int(char)
            return True
        except:
            return False

    # Checks if a word starts of ends with a special character.
    @staticmethod
    def starts_or_ends_with_exceptions(text):
        start_char = text[0]
        end_char = text[len(text)-1]

        exceptions = [
            "-", "+", "=", "/", "\ "[0], "*",
            ".", ",", ":", ";",
            "`", "&", "!", "?", "$", " ", "\'", "#"
        ]

        try:
            if start_char in exceptions or end_char in exceptions:
                return True
            elif len(text) > 1 and text[:2] in ["a-", "a.", "a.s"]:
                return True
            return False
        except:
            return False
