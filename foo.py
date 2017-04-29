from imports import *
from util import *
from ML import *

dp = PreProcess("data/concept_assertion_relation_training_data/all/")
X, y = dp.getFeatures()

print "Original Data shape = " + str(Counter(y))

retVecs = {}
print "Unsampled data with libsvm"
print "==============================================================="
ml = ML(X,y)
retVecs['unsampled'] = ml.libsvm(target_names=['None', 'Problem', 'Treatment', 'Test'])

print "Undersampled data using Clustered Centroids with libsvm"
print "==============================================================="
ml = ML(X,y, sample='cluster_centroids')
retVecs['underSampledCC'] = ml.libsvm(target_names=['None', 'Problem', 'Treatment', 'Test'])

print "Undersampled data using Condensed Nearest Neighbours with libsvm"
print "==============================================================="
ml = ML(X,y, sample='condensed_nn')
retVecs['underSampledCNN'] = ml.libsvm(target_names=['None', 'Problem', 'Treatment', 'Test'])

print "Oversampled data using Random Over Sampler with libsvm"
print "==============================================================="
ml = ML(X,y, sample='random_over_sampler')
retVecs['overSampledROS'] = ml.libsvm(target_names=['None', 'Problem', 'Treatment', 'Test'])

print "Pickling return vectors"
f = open('pickled/retVecs.pkl', 'wb')
pickle.dump(retVecs, f)
f.close()





