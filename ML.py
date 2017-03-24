from imports import *
from util import *

class ML(object):

    def __init__(self, X, y, sample = None):
        self.X = np.array(X).astype(float)
        # self.X = sk.preprocessing.normalize(self.X) -- Makes it far worse.
        self.y = np.array(y).astype(float)

        if sample == 'cluster_centroids':
            print "Cluster Centroid Sampler activated"
            sampler = ClusterCentroids(random_state=41)
        elif sample == 'condensed_nn':
            print "Condensed Nearest Neighbour activated"
            sampler = CondensedNearestNeighbour(random_state=41)
        elif sample == 'random_over_sampler':
            print "Random Over sampler Sampler activated"
            sampler = RandomOverSampler(random_state=41)
        else:
            return

        print "Sampling the dataset..."
        t1 = dt.now()
        self.X, self.y = sampler.fit_sample(self.X, self.y)
        print "Time Taken = " + str(dt.now() - t1)
        print "Sampled Datashape =" + str(Counter(self.y))

        return
        #print self.X[0]

    def NuSVC(self, ker='rbf', k=5, rand_state = None, verbose=True, target_names = None):

        clf = svm.NuSVC(kernel=ker,nu=0.01, verbose=verbose)
        cv = sk.cross_validation.KFold(len(self.X), n_folds=k, shuffle=True, random_state=rand_state)

        print "Cross Validation with NuSVC"
        i=1
        for traincv, testcv in cv:
            t = dt.now()
            print "==============================================================================="
            print "Fold no. " + str(i)
            clf.fit(self.X[traincv], self.y[traincv])
            print "Time Taken = " + str(dt.now() - t)
            #print clf.score(self.X[testcv], self.y[testcv])

            self.classReport(self.y[testcv], clf.predict(self.X[testcv]), target_names)
            i += 1

        return

    def libsvm(self, k=5, random_state = None, target_names=None, CV=False):

        if CV:

            cv = sk.cross_validation.KFold(len(self.X), n_folds=k, shuffle=True, random_state=random_state)

            print "Cross Validation with libsvm"
            i = 1
            for traincv, testcv in cv:
                t = dt.now()
                print "=============================================================================="
                print "Fold no. " + str(i)

                ret_vec = svm.libsvm.fit(self.X[traincv], self.y[traincv], 0)
                print "Time Taken = " + str(dt.now() - t)

                self.classReport(self.y[testcv],  svm.libsvm.predict(self.X[testcv], *ret_vec), target_names)
                i += 1
        else:
            print "Splitting data at 0.3"
            print "Training with libsvm"
            t= dt.now()
            X_train, X_test, y_train, y_test = sk.cross_validation.train_test_split(self.X, self.y, test_size=0.3, random_state=42)
            ret_vec = svm.libsvm.fit(X_train, y_train, 0)
            print "Time Taken = " + str(dt.now() - t)

            y_pred = svm.libsvm.predict(X_test, *ret_vec)
            self.classReport(y_test, y_pred, target_names)
            #self.plotPRcurve(y_test, y_pred, target_names)

        return ret_vec

    def classReport(self, y_true, y_pred, target_names):

        print "==============================================================================="
        print "Confusion Matrix: "
        print sk.metrics.confusion_matrix(y_true, y_pred)
        print
        print "Accuracy is: " + str(sk.metrics.accuracy_score(y_true, y_pred))
        print
        print "Classification report: "
        print sk.metrics.classification_report(y_true, y_pred, target_names=target_names)
        print "==============================================================================="

        return

    def plotPRcurve(self, y_test, y_score, target_names=None):

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'black'])
        lw = 2

        # Binarize the output
        y_test = sk.preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3])
        y_score = sk.preprocessing.label_binarize(y_score, classes=[0, 1, 2, 3])
        n_classes = y_test.shape[1]

        # Compute Precision-Recall and plot curve
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = sk.metrics.precision_recall_curve(y_test[:, i],
                                                                y_score[:, i])
            average_precision[i] = sk.metrics.average_precision_score(y_test[:, i], y_score[:, i])

        # Compute micro-average ROC curve and ROC area
        precision["micro"], recall["micro"], _ = sk.metrics.precision_recall_curve(y_test.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = sk.metrics.average_precision_score(y_test, y_score,
                                                             average="micro")

        # Plot Precision-Recall curve
        plt.clf()
        plt.plot(recall[0], precision[0], lw=lw, color='navy',
                 label='Precision-Recall curve')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
        # plt.legend(loc="lower left")
        # plt.show()

        # Plot Precision-Recall curve for each class
        #plt.clf()
        plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        for i, color in zip(range(n_classes), colors):
            plt.plot(recall[i], precision[i], color=color, lw=lw,
                     label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(target_names[i] if target_names else i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('P-R curve')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)
        plt.show()

        return

# dp = PreProcess("data/concept_assertion_relation_training_data/beth/")
# X, y = dp.getFeatures()
# ml = ML(X,y)
# ml.NuSVC(target_names=['None', 'Problem', 'Treatment', 'Test'])



