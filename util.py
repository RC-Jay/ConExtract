from imports import *

class PreProcess(object):

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
        return self.docs

    def sentTokenize(self, doc):
        try:
            return self.sent_tokenizer.tokenize(doc)
        except Exception as e:
            print "Failing at Sentence tokenization.."
            return False

    def wordTokenize(self, sents):
        doc = []
        try:
            for sent in sents:
                words = self.word_tokenizer.tokenize(sent)
                doc.append(words)
                for word in words:
                    if not word in self.vocabularyWords:             ## TODO test by using only lower case words
                        self.vocabularyWords.append(word)

                    temp = self.porterStemmer.stem(word)
                    if not temp in self.vocabularyPortStem:
                        self.vocabularyPortStem.append(temp)

                    temp = self.lancasterStemmer.stem(word)
                    if not temp in self.vocabularyLancStem:
                        self.vocabularyLancStem.append(temp)

                    temp = self.wordnetLemmatizer.lemmatize(word)
                    if not temp in self.vocabularyWordnetLem:
                        self.vocabularyWordnetLem.append(temp)

        except Exception as e:
            print "Failing at Word tokenization.."
            return False
        return doc

    def makeFeatureList(self):
        try:
            word_vec = calcWordFeats()
        except Exception as e:
            print "Failing at making Feature list..."
            return False

        return

    def calcWordFeats(self):

        word_vec = []
        for doc in self.docs:
            for sent in doc:
                for i in range(len(sent)):
                    word_vec.append(self.vocabularyWords.index(sent[i]))  ## Adding Word as a feature
                    word_vec.append(len(sent[i]))                         ## Adding length of word as a Feature
                    t = evalRegex(sent[i])                                ## Adding regex match as feature
                    if t:
                        word_vec.append(t)
                    else:
                        print "No match in regex lib"
                        exit(0)
                    word_vec.append(self.evalRegex(sent[i]))
                    word_vec.append(self.vocabularyPortStem.index(self.porterStemmer.stem(sent[i]))) ## Porter stemming
                    word_vec.append(self.vocabularyLancStem.index(self.lancasterStemmer.stem(sent[i]))) ## Lancaster stemming

                    ## Wordnet at sentence level
                    ## word_vec.append(self.vocabularyWordnetLem.index(self.wordnetLemmatizer.lemmatize(sent[i]))) ## Porter stemming
                    ## TODO pass pos as a parameter to lemmatizer.




    def evalRegex(self, word):

        if re.match(r"^[A-Z][a-zA-Z]+", word):
            return INITCAP
        elif re.match(r"[A-Z]+", word):
            return ALLCAPS
        elif re.match(r"[A-Za-z]*[A-Z]+[a-z]+[a-zA-Z]*", word):
            retun CAPSMIX
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




PreProcess("data/concept_assertion_relation_training_data/beth/txt")
