from imports import *

class PreProcess(object):
    """
    This class helps achieve all the Pre processing that needs to be done on the
    medical text, and finally returns the features that we would use to extract
    concept information.

    :params in constructor class

    path - The path where the training data (i.e the medical reports in .txt) resides.
    """

    def __init__(self, path):
        self.path = path

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.tagger = nltk.data.load(nltk.tag._POS_TAGGER)
        self.porterStemmer = nltk.stem.porter.PorterStemmer()
        self.snowBallStemmer = nltk.stem.snowball.EnglishStemmer()
        self.lancasterStemmer = nltk.stem.lancaster.LancasterStemmer()
        self.wordnetLemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

        self.vocabularyWords = []
        self.vocabularyPortStem = []
        self.vocabularyLancStem = []
        self.vocabularySnowStem = []
        self.vocabularyWordnetLem = []
        ## TODO Try other models of  word representation - Word2Vec

        self.feats = []
        self.docs = []

        for filename in glob.glob(os.path.join(path, '*.txt')):
            f = open(filename,'r')
            doc = f.read()
            self.docs.append(self.wordTokenize(self.sentTokenize(str(doc))))

        # for doc in docs:
        #     print doc
        #     exit(0)
        self.makeFeatureList()

        return

    def sentTokenize(self, doc):
        """
        A method that returns a list of all sentences found in a string(Medical
        Report)

        :params

        doc - A string that represents one medical report in the training data
        """

        try:
            return self.sent_tokenizer.tokenize(doc)
        except Exception as e:
            print "Failing at Sentence tokenization.."
            return False

    def wordTokenize(self, sents):
        """
        A method that returns a word level split of all the sentences passed to it

        :params

        sents = list of sentences(Output from sentTokenize())
        """

        doc = []
        try:
            for sent in sents:
                words = self.word_tokenizer.tokenize(sent)
                doc.append(words)
                for word in words:
                    if not word in self.vocabularyWords:  ## TODO test by using only lower case words
                        self.vocabularyWords.append(word)  ## Creating a list of all the unique words present
                                                            #  in the dataset. This helps us represent features
                                                            #  like words with an appropriate IR model

                    temp = self.porterStemmer.stem(word)
                    if not temp in self.vocabularyPortStem:
                        self.vocabularyPortStem.append(temp)  ## Creating a list of all the unique words obtained
                                                               # by using Porter Stemming algorithm

                    temp = self.lancasterStemmer.stem(word)
                    if not temp in self.vocabularyLancStem:
                        self.vocabularyLancStem.append(temp)  ## Creating a list of all the unique words obtained
                                                               # by using Lancaster Stemming algorithm

                    temp = self.wordnetLemmatizer.lemmatize(word)
                    if not temp in self.vocabularyWordnetLem:
                        self.vocabularyWordnetLem.append(temp)  ## Creating a list of all the unique words obtained
                                                                 # by using Wordnet Lemmatizer

        except Exception as e:
            print "Failing at Word tokenization.."
            return False
        return doc

    def makeFeatureList(self):
        """
        This method runs all the neccessary modules to generate the final feature
        set we would like to pass to a ML algorithm

        There are three steps here. We calcuate the word features, sentence features
        and ngram features in that order. Finally we merge all the vectors to obtain
        the feature set.
        """

        try:
            word_vecs = self.calcWordFeats()
            sent_vecs = self.calcSentenceFeats()
            ngram_vecs = self.calcNgramFeats()

        except Exception as e:
            print "Failing at making Feature list..."
            return False

        return


    def calcWordFeats(self):
        """
        This method returns a vector containing all the word features extracted per
        word
        """

        word_vecs = []
        for doc in self.docs:
            for sent in doc:
                word_vec = []
                for i in range(len(sent)):

                    word_vec.append(self.vocabularyWords.index(sent[i]))  ## Adding Word as a feature

                    word_vec.append(len(sent[i]))                         ## Adding length of word as a Feature

                    t = self.evalRegex(sent[i])                                ## Adding regex match as feature
                    if t:
                        word_vec.append(t)
                    else:
                        print "No match in regex lib"
                        exit(0)

                    word_vec.append(self.vocabularyPortStem.index(self.porterStemmer.stem(sent[i]))) ## Porter stemming

                    word_vec.append(self.vocabularyLancStem.index(self.lancasterStemmer.stem(sent[i]))) ## Lancaster stemming

                    ## Wordnet at sentence level
                    ## word_vec.append(self.vocabularyWordnetLem.index(self.wordnetLemmatizer.lemmatize(sent[i])))
                    ## TODO pass pos as a parameter to lemmatizer.

                    ## TODO - Wordshape classifer - Stanford CoreNLP library.

                word_vecs.append(word_vec)

        return word_vecs


    def evalRegex(self, word):

        if re.match(r"^[A-Z][a-zA-Z]+", word):
            return INITCAP
        elif re.match(r"[A-Z]+", word):
            return ALLCAPS
        elif re.match(r"[A-Za-z]*[A-Z]+[a-z]+[a-zA-Z]*", word):
            return CAPSMIX
        elif re.match(r"(?=[^aeiouAEIOU])(?=[a-zA-Z])", word):
            return NOVOWELS
        elif re.match(r"[A-Za-z0-9]*[0-9]+[a-zA-Z]+[a-zA-Z0-9]*", word):
            return HASDIGIT
        elif re.match(r"^[-]?[0-9]$", word):
            return SINGLEDIGIT
        elif re.match(r"^[-]?[0-9]{2}$",word):
            return DOUBLEDIGIT
        elif re.match(r"^[-]?[0-9]{4}$", word):
            return FOURDIGITS
        elif re.match(r"^[-]?[0-9]{5}$", word):
            return FIVEDIGITS
        elif(r"^[0-9]+$", word):
            return NATURALNUM
        elif re.match(r"^[+-]?(?:\d+\.?\d+|\d*\.\d+|\d+\/\d+)$", word):
            return REALNUM
        elif re.match(r"[a-zA-Z0-9]+", word):
            return ALPHANUM
        elif re.match(r"[a-zA-Z0-9]*[-]+[a-zA-Z0-9]*", word):           # TODO - Check if dash is '-'(hyphen)
            return HASDASH
        elif re.match(r"^[.]$", word):
            return PUNCTUATION
        # TODO - Regex for phone nums, find structure
        elif re.match(r"(?=[a-zA-Z0-9]*[-])(?=[a-zA-Z]*[0-9])(?=[a-zA-Z0-9][a-zA-Z])", word):
            return HASDASHNUMALPHA
        elif re.match(r"^[-/.]$", word):
            return DATESEPARATOR
        else:
            return False

    def calcSentenceFeats(self):

        sent_vecs = []
        for doc in self.docs:
            for sent in doc:
                sent_vec = []
                pos = self.tagger.tag(self.word_tokenizer.tokenize(sent))
                for item in pos:
                    sent_vec.append(item[1])
                    sent_vec.append(self.vocabularyWordnetLem.index(self.wordnetLemmatizer.lemmatize(item[0], item[1])))
                    ## TODO - Formatted text
                sent_vecs.append(sent_vec)
        return sent_vecs

    def calcNgramFeats(self):

        ngram_vecs = []
        for doc in self.docs:
            for sent in doc:
                ngram_vec = []
                for i in range(len(sent)):
                    if i == 0:
                        ngram_vec.append(-1)
                    else:
                        ngram_vec.append(self.vocabularyWords.index(sent[i-1]))
                    if i == len(sent) - 1:
                        ngram_vec.append(-1)
                    else:
                        ngram_vec.append(self.vocabularyWords.index(sent[i + 1]))

                ngram_vecs.append(ngram_vec)

        return ngram_vecs







PreProcess("data/concept_assertion_relation_training_data/beth/txt")
