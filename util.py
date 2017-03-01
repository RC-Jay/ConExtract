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
        self.vocabularyWordShapes = []
        self.vocabularyPOSTags = []
        # TODO Try other models of  word representation - Word2Vec

        if LOAD_DOCS:
            print "Retrieving pickled docs and vocabularies"

            f = open("pickled/docs.pkl", "rb")
            self.docs = pickle.load(f)
            f.close()

            f = open("pickled/words.pkl", "rb")
            self.vocabularyWords = pickle.load(f)
            f.close()

            f = open("pickled/portStem.pkl", "rb")
            self.vocabularyPortStem = pickle.load(f)
            f.close()

            f = open("pickled/lancStem.pkl", "rb")
            self.vocabularyLancStem = pickle.load(f)
            f.close()

            f = open("pickled/wnLem.pkl", "rb")
            self.vocabularyWordnetLem = pickle.load(f)
            f.close()

        else:
            self.docs = []

            print "Initialization of docs..."
            filelist = glob.glob(os.path.join(path, '*.txt'))
            for filename in sorted(filelist, cmp=locale.strcoll):
                print "Processing file - " + str(filename)
                f = open(filename,'r')
                doc = f.read()
                self.docs.append(self.wordTokenize(self.sentTokenize(str(doc))))

            print "Pickling all the docs and vocabularies"

            f = open("pickled/docs.pkl", "wb")
            pickle.dump(self.docs, f)
            f.close()

            f = open("pickled/words.pkl", "wb")
            pickle.dump(self.vocabularyWords, f)
            f.close()

            f = open("pickled/portStem.pkl", "wb")
            pickle.dump(self.vocabularyPortStem, f)
            f.close()

            f = open("pickled/lancStem.pkl", "wb")
            pickle.dump(self.vocabularyLancStem, f)
            f.close()

            f = open("pickled/wnLem.pkl", "wb")
            pickle.dump(self.vocabularyWordnetLem, f)
            f.close()

        if LOAD_FEATS:
            f = open("pickled/feats.pkl", 'rb')
            self.feats = pickle.load(f)
            f.close()
        else:
            self.feats = []
            print "Making feature list"
            self.makeFeatureList()

            print "Pickling all the feats"
            f = open("pickled/feats.pkl", "wb")
            pickle.dump(self.feats, f)
            f.close()

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
            print str(e.message)
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
                pos = self.tagger.tag(words)
                doc.append(words)
                for word, tag in pos:
                    if not word in self.vocabularyWords:         # TODO test by using only lower case words
                        self.vocabularyWords.append(word)        # Creating a list of all the unique words present
                                                                 # in the dataset. This helps us represent features
                                                                 # like words with an appropriate IR model

                    temp = self.porterStemmer.stem(word)
                    if not temp in self.vocabularyPortStem:
                        self.vocabularyPortStem.append(temp)     # Creating a list of all the unique words obtained
                                                                 # by using Porter Stemming algorithm

                    temp = self.lancasterStemmer.stem(word)
                    if not temp in self.vocabularyLancStem:
                        self.vocabularyLancStem.append(temp)     # Creating a list of all the unique words obtained
                                                                 # by using Lancaster Stemming algorithm

                    temp = self.wordnetLemmatizer.lemmatize(word, pwn.penn_to_wn(tag))
                    if not temp in self.vocabularyWordnetLem:
                        self.vocabularyWordnetLem.append(temp)   # Creating a list of all the unique words obtained
                                                                 # by using Wordnet Lemmatizer

        except Exception as e:
            print "Failing at Word tokenization.."
            print str(e.message)
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
            print "Failing to create Feature list..."
            print str(e.message)
            return False

        return


    def calcWordFeats(self):
        """
        This method returns a vector containing all the word features extracted per
        word
        """
        try:
            print "Extracting word level features.."
            word_vecs = []
            for doc in self.docs:
                for sent in doc:

                    for i in range(len(sent)):
                        word_vec = []
                        word_vec.append(self.vocabularyWords.index(sent[i]))  # Adding Word as a feature

                        word_vec.append(len(sent[i]))                         # Adding length of word as a Feature

                        t = self.evalRegex(sent[i])                           # Adding regex match as feature
                        if t:
                            word_vec.append(t)
                        else:
                            print "No match in regex lib"
                            exit(0)

                        word_vec.append(self.vocabularyPortStem.index(self.porterStemmer.stem(sent[i]))) # Porter stemming

                        word_vec.append(self.vocabularyLancStem.index(self.lancasterStemmer.stem(sent[i]))) # Lancaster stemming

                        # TODO - Wordshape classifer - Stanford CoreNLP library.
                        wordShapes = ws.getWordShapes(sent[i])
                        for shape in wordShapes:
                            if shape in self.vocabularyWordShapes:
                                word_vec.append(self.vocabularyWordShapes.index(shape))
                            else:
                                self.vocabularyWordShapes.append(shape)
                                word_vec.append(self.vocabularyWordShapes.index(shape))
                        print str(sent[i]) + ": " + str(word_vec)
                        word_vecs.append(word_vec)


            return word_vecs
        except Exception as e:
            print "Failed to calculate word features"
            print str(e.message)
            return False


    def evalRegex(self, word):

        try:
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
            elif re.match(r"[a-zA-Z0-9]*[-]+[a-zA-Z0-9]*", word):           # TODO - Try by using + instead of *
                return HASDASH
            elif re.match(r"^[.]$", word):
                return PUNCTUATION
            elif re.match(r"^[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$", word):
                return PHONE1
            elif re.match(r"^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$", word):
                return PHONE2
            elif re.match(r"(?=[a-zA-Z0-9]*[-])(?=[a-zA-Z]*[0-9])(?=[a-zA-Z0-9][a-zA-Z])", word):
                return HASDASHNUMALPHA
            elif re.match(r"^[-/]$", word):
                return DATESEPARATOR
            else:
                return False
        except Exception as e:
            print "Failed to evaluate regex of word - " + str(word)
            print str(e.message)
            return False

        '''
        "INITCAP" : r"^[A-Z].*$",
		"ALLCAPS" : r"^[A-Z]+$",
		"CAPSMIX" : r"^[A-Za-z]+$",
		"HASDIGIT" : r"^.*[0-9].*$",
		"SINGLEDIGIT" : r"^[0-9]$",
		"DOUBLEDIGIT" : r"^[0-9][0-9]$",
		"FOURDIGITS" : r"^[0-9][0-9][0-9][0-9]$",
		"NATURALNUM" : r"^[0-9]+$",
		"REALNUM" : r"^[0-9]+.[0-9]+$",
		"ALPHANUM" : r"^[0-9A-Za-z]+$",
		"HASDASH" : r"^.*-.*$",
		"PUNCTUATION" : r"^[^A-Za-z0-9]+$",
		"PHONE1" : r"^[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
		"PHONE2" : r"^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
		"FIVEDIGIT" : r"^[0-9][0-9][0-9][0-9][0-9]",
		"NOVOWELS" : r"^[^AaEeIiOoUu]+$",
		"HASDASHNUMALPHA" : r"^.*[A-z].*-.*[0-9].*$ | *.[0-9].*-.*[0-9].*$",
		"DATESEPERATOR" : r"^[-/]$",
        '''

    def calcSentenceFeats(self):

        try:
            print "Extracting sentence level features.."
            sent_vecs = []
            for doc in self.docs:
                for j, sent in enumerate(doc):

                    pos = self.tagger.tag(sent)
                    for word, tag in pos:
                        sent_vec = []
                        if tag in self.vocabularyPOSTags:
                            sent_vec.append(self.vocabularyPOSTags.index(tag))
                        else:
                            self.vocabularyPOSTags.append(tag)
                            sent_vec.append(self.vocabularyPOSTags.index(tag))
                        sent_vec.append(self.vocabularyWordnetLem.index(self.wordnetLemmatizer.lemmatize(word, \
                                                                                          pwn.penn_to_wn(tag))))
                        right = " ".join([w for w in sent[j:]])
                        if self.is_test_result(right):
                            sent_vec.append(1)
                        else:
                            sent_vec.append(0)
                        print word + ": " + str(sent_vec)
                        sent_vecs.append(sent_vec)

            return sent_vecs

        except Exception as e:
            print "Failed to calculate sentence features"
            print str(e.message)
            return False

    def is_test_result(self, context):
        # note: make spaces optional?
        regex = r"^[A-Za-z]+( )*(-|--|:|was|of|\*|>|<|more than|less than)( )*[0-9]+(%)*"
        if not re.search(regex, context):
            return False
        return True

    def calcNgramFeats(self):

        # TODO - Add more features of prev and next words.
        try:
            print "Extracting ngram level features.."
            ngram_vecs = []
            for doc in self.docs:
                for sent in doc:

                    for i in range(len(sent)):
                        ngram_vec = []
                        if i == 0:
                            ngram_vec.append(-1)
                        else:
                            ngram_vec.append(self.vocabularyWords.index(sent[i-1]))
                        if i == len(sent) - 1:
                            ngram_vec.append(-1)
                        else:
                            ngram_vec.append(self.vocabularyWords.index(sent[i + 1]))

                        print sent[i] + ": " + str(ngram_vec)
                        ngram_vecs.append(ngram_vec)

            return ngram_vecs
        except Exception as e:
            print "Failed to calculate ngram features"
            print str(e.message)
            return False







PreProcess("data/concept_assertion_relation_training_data/beth/txt")
