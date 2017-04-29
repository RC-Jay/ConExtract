from imports import *

word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
tagger = nltk.data.load(nltk.tag._POS_TAGGER)
porterStemmer = nltk.stem.porter.PorterStemmer()
snowBallStemmer = nltk.stem.snowball.EnglishStemmer()
lancasterStemmer = nltk.stem.lancaster.LancasterStemmer()
wordnetLemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

f = open("pickled/retVecs.pkl", "rb")
retVecDic = pickle.load(f)
ret_vec = retVecDic['overSampledROS']
f.close()

print retVecDic.keys()
f = open("pickled/wv.pkl", "rb")
wvModel = pickle.load(f)
f.close()

f = open("pickled/wvPort.pkl", "rb")
wvModelPorter = pickle.load(f)
f.close()

f = open("pickled/wvLanc.pkl", "rb")
wvModellancaster = pickle.load(f)
f.close()

f = open("pickled/wvWnLem.pkl", "rb")
wvModelWnLemmatizer = pickle.load(f)
f.close()

f = open("pickled/wordShape.pkl", 'rb')
vocabularyWordShapes = pickle.load(f)
f.close()

f = open("pickled/posTags.pkl", 'rb')
vocabularyPOSTags = pickle.load(f)
f.close()


def evalRegex(word):
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

def is_test_result(context):
    # note: make spaces optional?
    regex = r"^[A-Za-z]+( )*(-|--|:|was|of|\*|>|<|more than|less than)( )*[0-9]+(%)*"
    if not re.search(regex, context):
        return False
    return True

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "Usage: python driver.py `ehr_file`"
        exit()

    print "Extracting features from Doc"

    f = open(sys.argv[1])
    doc = f.readlines()
    print len(doc), doc[len(doc)-1]
    f.close()

    docFeats = list()
    gWords = list()
    for sent in doc:
        print sent

        words = word_tokenizer.tokenize(sent)
        pos = tagger.tag(words)
        j = 0
        for i, (word, tag) in enumerate(pos):
            gWords.append(word)
            wordFeats = list()

            try:
                wordFeats += list(wvModel.wv[word])
            except Exception as e:
                "Exception at wordVectors"
                wordFeats += [-1] * WED

            wordFeats.append(len(word))

            t = evalRegex(word)  # Adding regex match as feature
            if t:
                wordFeats.append(t)
            else:
                print "No match in regex lib"
                exit(0)

            try:
                wordFeats += list(wvModelPorter.wv[porterStemmer.stem(word)])  # Porter stemming
            except Exception as e:
                wordFeats += [-1] * WED

            try:
                wordFeats += list(
                    wvModellancaster.wv[lancasterStemmer.stem(word)])  # Lancaster stemming
            except Exception as e:
                wordFeats += [-1] * WED

            wordShapes = ws.getWordShapes(word)
            for shape in wordShapes:
                if shape in vocabularyWordShapes:
                    wordFeats.append(vocabularyWordShapes.index(shape))
                else:
                    "Couldn't index word shape"
                    exit()

            if tag in vocabularyPOSTags:
                wordFeats.append(vocabularyPOSTags.index(tag))
            else:
                print "Couldn't index POS tag: " + tag
                exit(0)

            try:
                wordFeats += list(wvModelWnLemmatizer.wv[wordnetLemmatizer.lemmatize(word)])
            except Exception as e:
                wordFeats += [-1] * WED

            right = " ".join([w for w in sent[j:]])
            if is_test_result(right):
                wordFeats.append(1)
            else:
                wordFeats.append(0)

            if i == 0:
                wordFeats += [-1] * WED
            else:
                try:
                    wordFeats += list(wvModel.wv[sent[i - 1]])
                except Exception as e:
                    wordFeats += [-1] * WED

            if i == len(sent) - 1:
                wordFeats += [-1] * WED
            else:
                try:
                    wordFeats += list(wvModel.wv[sent[i + 1]])
                except Exception as e:
                    wordFeats += [-1] * WED
            j += 1
            docFeats.append(np.array(wordFeats))
            #print i, word, tag

    print "End of for loop"
    tagged = svm.libsvm.predict(np.array(docFeats), *ret_vec)

    for i, id in enumerate(tagged):
        if id != 0:
            print gWords[i], id









