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
        self.rootPath = path

        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.tagger = nltk.data.load(nltk.tag._POS_TAGGER)
        self.porterStemmer = nltk.stem.porter.PorterStemmer()
        self.snowBallStemmer = nltk.stem.snowball.EnglishStemmer()
        self.lancasterStemmer = nltk.stem.lancaster.LancasterStemmer()
        self.wordnetLemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        self.sampler = ClusterCentroids(random_state=41)

        # self.vocabularyWords = []
        # self.vocabularyPortStem = []
        # self.vocabularyLancStem = []
        # self.vocabularySnowStem = []
        # self.vocabularyWordnetLem = []
        self.vocabularyWordShapes = []
        self.vocabularyPOSTags = []

        if LOAD_DOCS:
            print "Retrieving pickled docs and vocabularies"

            f = open("pickled/docs.pkl", "rb")
            self.docs = pickle.load(f)
            f.close()

            f = open("pickled/labels.pkl", "rb")
            self.labels = pickle.load(f)
            f.close()

            # f = open("pickled/words.pkl", "rb")
            # self.vocabularyWords = pickle.load(f)
            # f.close()
            #
            # f = open("pickled/portStem.pkl", "rb")
            # self.vocabularyPortStem = pickle.load(f)
            # f.close()
            #
            # f = open("pickled/lancStem.pkl", "rb")
            # self.vocabularyLancStem = pickle.load(f)
            # f.close()
            #
            # f = open("pickled/wnLem.pkl", "rb")
            # self.vocabularyWordnetLem = pickle.load(f)
            # f.close()

            assert len(self.labels) == len(self.docs)

        else:
            self.docs = []
            self.labels = []

            print "Initialization of docs..."
            t = dt.now()
            filelist = glob.glob(os.path.join(self.rootPath + "txt",
                                              '*.txt'))  # List of all files that contains the individual reports in txt
            conlist = glob.glob(os.path.join(self.rootPath + "concept",
                                             '*.con'))  # Synchronised list of annotated classes for the medical reports
            for filename, conname in zip(sorted(filelist, cmp=locale.strcoll), sorted(conlist, cmp=locale.strcoll)):
                print "Processing file - " + str(filename) + " -- " + str(conname)
                f = open(filename, 'r')
                doc = f.readlines()
                f.close()

                f = open(conname, 'r')
                con = f.readlines()
                f.close()

                temp = self.wordTokenize(doc, con)
                self.docs.append(temp[0])
                self.labels.append(temp[1])
            print "Time Taken = " + str(dt.now() - t)

            print "Pickling all the docs and vocabularies"

            f = open("pickled/docs.pkl", "wb")
            pickle.dump(self.docs, f)
            f.close()

            f = open("pickled/labels.pkl", "wb")
            pickle.dump(self.labels, f)
            f.close()

            # f = open("pickled/words.pkl", "wb")
            # pickle.dump(self.vocabularyWords, f)
            # f.close()
            #
            # f = open("pickled/portStem.pkl", "wb")
            # pickle.dump(self.vocabularyPortStem, f)
            # f.close()
            #
            # f = open("pickled/lancStem.pkl", "wb")
            # pickle.dump(self.vocabularyLancStem, f)
            # f.close()
            #
            # f = open("pickled/wnLem.pkl", "wb")
            # pickle.dump(self.vocabularyWordnetLem, f)
            # f.close()

        if WORD2VEC:
            print "Retrieving word2vec models"

            f = open("pickled/wv.pkl", "rb")
            self.wvModel = pickle.load(f)
            f.close()

            f = open("pickled/wvPort.pkl", "rb")
            self.wvModelPorter = pickle.load(f)
            f.close()

            f = open("pickled/wvLanc.pkl", "rb")
            self.wvModellancaster = pickle.load(f)
            f.close()

            f = open("pickled/wvWnLem.pkl", "rb")
            self.wvModelWnLemmatizer = pickle.load(f)
            f.close()

        else:

            print "Creating Word2Vec models.."
            t = dt.now()

            self.makeWord2VecModels()
            print "Time Taken = " + str(dt.now() - t)

            f = open("pickled/wv.pkl", "wb")
            pickle.dump(self.wvModel, f)
            f.close()

            f = open("pickled/wvPort.pkl", "wb")
            pickle.dump(self.wvModelPorter, f)
            f.close()

            f = open("pickled/wvLanc.pkl", "wb")
            pickle.dump(self.wvModellancaster, f)
            f.close()

            f = open("pickled/wvWnLem.pkl", "wb")
            pickle.dump(self.wvModelWnLemmatizer, f)
            f.close()


        if LOAD_FEATS:

            f = open("pickled/wordShape.pkl", 'rb')
            self.vocabularyWordShapes = pickle.load(f)
            f.close()

            f = open("pickled/posTags.pkl", 'rb')
            self.vocabularyPOSTags = pickle.load(f)
            f.close()

            f = open("pickled/x_feats.pkl", 'rb')
            self.X = pickle.load(f)
            f.close()

            f = open("pickled/y_feats.pkl", 'rb')
            self.y = pickle.load(f)
            f.close()

            assert len(self.X) == len(self.y)


        else:
            print "Making feature list"
            t = dt.now()
            self.X = self.makeFeatureList()

            self.y = self.flatten(self.labels)
            print "Time Taken = " + str(dt.now() - t)

            print "Pickling all the feats"

            f = open("pickled/wordShape.pkl", "wb")
            pickle.dump(self.vocabularyWordShapes, f)
            f.close()

            f = open("pickled/posTags.pkl", "wb")
            pickle.dump(self.vocabularyPOSTags, f)
            f.close()

            f = open("pickled/x_feats.pkl", "wb")
            pickle.dump(self.X, f)
            f.close()

            f = open("pickled/y_feats.pkl", "wb")
            pickle.dump(self.y, f)
            f.close()

        assert len(self.X) == len(self.y)



        print "Length of dataset = ",len(self.X)
        print "Dimensions of feature space = ", len(self.X[0])

        return

    def getFeatures(self, one_hot = False):

        if one_hot:
            return self.expandFeatures(), self.y
        else:
            return self.X, self.y

    def cleanData(self):
        # TODO - Clean the dataset synchronously

        pass

    def flatten(self, list):
        '''
        Method to flatten the list of labels to 1-D
        :param list: Contains a 3 level list of all corresponding classes of training corpora
        :return: Flattened list of classes (self.y)
        '''
        result = []
        for doc in list:
            for sent in doc:
                for word in sent:
                    result.append(word)

        return result

    def expandFeatures(self):

        X = []
        '''
        self.featureNames = ["word", "lenWord", "MitRe", "portStem", "lancStem", "wordShape", "POS", "wordNetLem", \
                             "isTest", "prevWord", "nextWord"]
        '''
        for i, dp in enumerate(self.X):
            #print "Expanding data point - " + str(i + 1)
            expFeat = []

            expFeat += list(dp[0:10])

            expFeat.append(int(dp[10]))

            expFeat += list(np.eye(18, dtype=np.int32)[int(dp[11]) - 101])

            expFeat += list(dp[12:22])

            expFeat += list(dp[22:32])

            expFeat += list(dp[32:37])

            expFeat += list(np.eye(len(self.vocabularyPOSTags), dtype=np.int32)[int(dp[10])])

            expFeat += list(dp[38:48])

            expFeat.append(int(dp[48]))

            expFeat += list(dp[49:59])  # TODO - Add all features of prev word in the mix

            expFeat += list(dp[59:69])

            X.append(expFeat)

        return X

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

    def wordTokenize(self, sents, con):
        """
        A method that returns a word level split of all the sentences passed to it

        :params

        sents = list of sentences(Output from sentTokenize())
        """

        doc = []
        # labels = [[None] * len(sent) for sent in sents]
        labels = []
        try:
            for i, sent in enumerate(sents):
                words = self.word_tokenizer.tokenize(sent)
                labels.append([0] * len(words))
                #pos = self.tagger.tag(words)
                doc.append(words)
                # for word, tag in pos:
                #     if not word in self.vocabularyWords:  # TODO test by using only lower case words
                #         self.vocabularyWords.append(word)  # Creating a list of all the unique words present
                #         # in the dataset. This helps us represent features
                #         # like words with an appropriate IR model
                #
                #     temp = self.porterStemmer.stem(word)
                #     if not temp in self.vocabularyPortStem:
                #         self.vocabularyPortStem.append(temp)  # Creating a list of all the unique words obtained
                #         # by using Porter Stemming algorithm
                #
                #     temp = self.lancasterStemmer.stem(word)
                #     if not temp in self.vocabularyLancStem:
                #         self.vocabularyLancStem.append(temp)  # Creating a list of all the unique words obtained
                #         # by using Lancaster Stemming algorithm
                #
                #     temp = self.wordnetLemmatizer.lemmatize(word, pwn.penn_to_wn(tag))
                #     if not temp in self.vocabularyWordnetLem:
                #         self.vocabularyWordnetLem.append(temp)  # Creating a list of all the unique words obtained
                #         # by using Wordnet Lemmatizer
            for line in con:
                c, t = line.split('||')
                t = t[3:-2]
                d = re.search(r'\"(.+?)\"', c)
                d = d.group(1)
                c = c.split()
                start = c[-2].split(':')
                end = c[-1].split(':')
                assert "concept spans one line", start[0] == end[0]
                l = int(start[0]) - 1
                start = (int(start[0]), int(start[1]))
                end = (int(end[0]), int(end[1]))

                if d == " ".join(doc[start[0] - 1][start[1]:end[1] + 1]).lower():
                    for i in range(start[1], end[1] + 1):
                        if t.lower() == 'problem':
                            labels[start[0] - 1][i] = 1
                        elif t.lower() == 'treatment':
                            labels[start[0] - 1][i] = 2
                        elif t.lower() == 'test':
                            labels[start[0] - 1][i] = 3
                        else:
                            print "Error in mapping classes"
                            raise Exception

        except Exception as e:
            print "Failing at Word tokenization.."
            print str(e.args), str(e.message)
            return False

        assert len(doc) == len(labels)

        return (doc, labels)

    def makeWord2VecModels(self, dim=WED):

        sents = list()
        pSents = list()
        lSents = list()
        wnSents = list()
        for doc in self.docs:
            for sent in doc:
                sents.append(sent)

                pSent = list()
                lSent = list()
                wnSent = list()

                for word, pos in self.tagger.tag(sent):
                    pSent.append(self.porterStemmer.stem(word))
                    lSent.append(self.lancasterStemmer.stem(word))
                    wnSent.append(self.wordnetLemmatizer.lemmatize(word, pwn.penn_to_wn(pos)))

                pSents.append(pSent)
                lSents.append(lSent)
                wnSents.append(wnSent)
        print "Word Model"
        self.wvModel = Word2Vec(sents, size=dim, window=5, min_count=5, workers=4)
        print "Porter Model"
        self.wvModelPorter = Word2Vec(pSents, size=dim, window=5, min_count=5, workers=4)
        print "Lancaster Model"
        self.wvModellancaster = Word2Vec(lSents, size=dim, window=5, min_count=5, workers=4)
        print "Wordnet Model"
        self.wvModelWnLemmatizer = Word2Vec(wnSents, size=dim, window=5, min_count=5, workers=4)

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

            assert len(ngram_vecs) == len(sent_vecs) == len(word_vecs)

            return np.hstack((word_vecs, sent_vecs, ngram_vecs))

        except Exception as e:
            print "Failing to create Feature list..."
            print str(e.message)
            exit(0)

    def calcWordFeats(self):
        """
        This method returns a vector containing all the word features extracted per
        word
        """
        try:
            print "Extracting word level features.."
            word_vecs = list()
            for doc in self.docs:
                for sent in doc:

                    for i in range(len(sent)):
                        word_vec = list()
                        try:
                            word_vec += list(self.wvModel.wv[sent[i]])  # Adding Word as a feature
                        except Exception as e:
                            word_vec += [-1] * WED

                        word_vec.append(len(sent[i]))  # Adding length of word as a Feature

                        t = self.evalRegex(sent[i])  # Adding regex match as feature
                        if t:
                            word_vec.append(t)
                        else:
                            print "No match in regex lib"
                            exit(0)

                        try:
                            word_vec += list(self.wvModelPorter.wv[self.porterStemmer.stem(sent[i])])  # Porter stemming
                        except Exception as e:
                            word_vec += [-1] * WED
                        try:
                            word_vec += list(
                                self.wvModellancaster.wv[self.lancasterStemmer.stem(sent[i])])  # Lancaster stemming
                        except Exception as e:
                            word_vec += [-1] * WED

                        wordShapes = ws.getWordShapes(sent[i])
                        for shape in wordShapes:
                            if shape in self.vocabularyWordShapes:
                                word_vec.append(self.vocabularyWordShapes.index(shape))
                            else:
                                self.vocabularyWordShapes.append(shape)
                                word_vec.append(self.vocabularyWordShapes.index(shape))
                        # print str(sent[i]) + ": " + str(word_vec)
                        word_vecs.append(word_vec)

            return word_vecs
        except Exception as e:
            print "Failed to calculate word features"
            print str(e.message)
            exit(0)

    def evalRegex(self, word):
        '''
        This method is used to evaluate the regex class of each word
        :param word:
        :return: Corresponding ReGex class
        '''

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
            elif re.match(r"^[-]?[0-9]{2}$", word):
                return DOUBLEDIGIT
            elif re.match(r"^[-]?[0-9]{4}$", word):
                return FOURDIGITS
            elif re.match(r"^[-]?[0-9]{5}$", word):
                return FIVEDIGITS
            elif (r"^[0-9]+$", word):
                return NATURALNUM
            elif re.match(r"^[+-]?(?:\d+\.?\d+|\d*\.\d+|\d+\/\d+)$", word):
                return REALNUM
            elif re.match(r"[a-zA-Z0-9]+", word):
                return ALPHANUM
            elif re.match(r"[a-zA-Z0-9]*[-]+[a-zA-Z0-9]*", word):  # TODO - Try by using + instead of *
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
        '''
        Calculate the features of each word at the sentence level. Involves POS tag, Wordnet lemmatizer and a hard coded condition checker.
        :return: Vector containing the sentence level feature of all words in sample.
        '''

        try:
            print "Extracting sentence level features.."
            sent_vecs = list()
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
                        try:
                            sent_vec += list(self.wvModelWnLemmatizer.wv[self.wordnetLemmatizer.lemmatize(word)])
                        except Exception as e:
                            sent_vec += [-1] * WED

                        right = " ".join([w for w in sent[j:]])
                        if self.is_test_result(right):
                            sent_vec.append(1)
                        else:
                            sent_vec.append(0)
                        # print word + ": " + str(sent_vec)
                        sent_vecs.append(sent_vec)

            return sent_vecs

        except Exception as e:
            print "Failed to calculate sentence features"
            print str(e.message)
            exit()

    def is_test_result(self, context):
        # note: make spaces optional?
        regex = r"^[A-Za-z]+( )*(-|--|:|was|of|\*|>|<|more than|less than)( )*[0-9]+(%)*"
        if not re.search(regex, context):
            return False
        return True

    def calcNgramFeats(self):
        '''
        Calclate ngram feature i.e next and previous word of all words in sample.
        :return: A vector containing ngram features of all words.
        '''

        # TODO - Add more features of prev and next words.
        try:
            print "Extracting ngram level features.."
            ngram_vecs = []
            for doc in self.docs:
                for sent in doc:

                    for i, word in enumerate(sent):
                        ngram_vec = []
                        if i == 0:
                            ngram_vec += [-1] * WED
                        else:
                            try:
                                ngram_vec += list(self.wvModel.wv[word])
                            except Exception as e:
                                ngram_vec += [-1] * WED
                        if i == len(sent) - 1:
                            ngram_vec += [-1] * WED
                        else:
                            try:
                                ngram_vec += list(self.wvModel.wv[word])
                            except Exception as e:
                                ngram_vec += [-1] * WED

                        # print sent[i] + ": " + str(ngram_vec)
                        ngram_vecs.append(ngram_vec)

            return ngram_vecs
        except Exception as e:
            print "Failed to calculate ngram features"
            print str(e.message)
            exit(0)


#data = PreProcess("data/concept_assertion_relation_training_data/beth/")
