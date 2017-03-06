from imports import *
from util import *

class SVM(object):

    def __init__(self, X, y, model=None):
        self.X = np.array(X).astype(float)
        # self.X = sk.preprocessing.normalize(self.X) -- Makes it far worse.
        self.y = np.array(y).astype(float)

        clf = svm.NuSVC(kernel='rbf',nu=0.01)
        cv = sk.cross_validation.KFold(len(self.X), n_folds=5, shuffle=True, random_state=None)

        print "Cross Validation with NuSVC"
        i=1
        for traincv, testcv in cv:
            print "==============================================================================="
            print "Fold no. " + str(i)
            clf.fit(self.X[traincv], self.y[traincv])
            #print clf.score(self.X[testcv], self.y[testcv])

            y_true = self.y[testcv]
            y_pred = clf.predict(self.X[testcv])
            target_names = ['None', 'Problem', 'Treatment', 'Test']
            print sk.metrics.classification_report(y_true, y_pred, target_names=target_names)
            print "==============================================================================="
            i+=1

        print "Cross Validation with libsvm"
        i=1
        for traincv, testcv in cv:
            print "=============================================================================="
            print "Fold no. " + str(i)

            ret_vec = svm.libsvm.fit(self.X[traincv], self.y[traincv], 0)
            #print ret_vec

            y_true = self.y[testcv]
            y_pred = svm.libsvm.predict(self.X[testcv], *ret_vec)
            target_names = ['None', 'Problem', 'Treatment', 'Test']
            print sk.metrics.classification_report(y_true, y_pred, target_names=target_names)
            print "=============================================================================="
            i+=1

data = PreProcess("data/concept_assertion_relation_training_data/beth/")
X,y = data.getFeatures()
SVM(X,y)
