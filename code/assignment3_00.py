import cPickle as pickle
import urllib
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import idx2numpy
import copy 
import matplotlib.pyplot as plt

def parse_data(data):
    """Parses raw bank note dataset into feature vectors and labels"""
    labels = np.array([int(d.strip().split(',')[-1:][0]) for d in data])
    feature_vects = np.array([np.array([float(e) for e in d.strip().split(',')[:-1]]) for d in data])
    return feature_vects, labels

def parse_iris_data(data_iris):
    labels = []
    data_iris = data_iris[:-1]
    feature_vects = np.array([np.array([float(e) for e in d.strip().split(',')[:-1]]) for d in data_iris])
    for d in data_iris:
        lbl = d[:-1].split(',')[-1:][0]
        if lbl == 'Iris-setosa':
            labels.append(0.)
        elif lbl == 'Iris-versicolor':
            labels.append(1.)
        elif lbl == 'Iris-virginica':
            labels.append(2.)
    labels = np.array(labels)
    return feature_vects, labels

def sigmoid(theta, x):
    p = -np.dot(theta, x)
    if p > 20 : return .0001
    elif p < -20 : return .999
    return 1 / (1 + np.exp(p))

def error_delta(theta, x, y):
    return np.sum(np.array([(sigmoid(theta, x[i]) - y[i]) * x[i] for i in xrange(x.shape[0])]), axis=0)

def likelihood(theta, x, y):
    return np.sum([(y[i] * np.log(sigmoid(theta, x[i]) + .001)) + ((1 - y[i]) * np.log(1 - sigmoid(theta, x[i]) + .001)) for i in xrange(x.shape[0])])

def log_reg_pred(theta, x):
    """Classifies observations in x according to classifier defined by theta"""
    preds = np.array([sigmoid(theta, x_) for x_ in x])
    less = preds < .5
    great = preds > .5
    preds[less] = 0
    preds[great] = 1
    return preds

def log_reg_trainer(x_train, y_train, learn_rate=.01):
    """Trains a logistic regression classifier on input training data.
    Basically finds values of theta."""
    theta = np.ones(x_train.shape[1])
    preds = log_reg_pred(theta, x_train)
    acc = accuracy_score(y_train, preds)
    prev_acc = acc
    delta = 1
    while np.abs(delta) > .0001:
        theta -= error_delta(theta, x_train, y_train) * learn_rate
        preds = log_reg_pred(theta, x_train)
        acc = accuracy_score(y_train, preds)
        delta = acc - prev_acc
        prev_acc = acc
    return theta

def log_reg(x_bn, y_bn, nfolds=4, degree=1):
    """Performs logistic regression experiments on Bank Note dataset for 2 class discrimination."""
    kf = KFold(y_bn.shape[0], n_folds=nfolds, shuffle=True)
    poly = PolynomialFeatures(degree)
    x_bn = poly.fit_transform(x_bn)
    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    for train_ids, test_ids in kf:
        theta = log_reg_trainer(x_bn[train_ids], y_bn[train_ids])
        y_pred = log_reg_pred(theta, x_bn[test_ids])
        avg_accuracy += accuracy_score(y_bn[test_ids], y_pred)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_bn[test_ids], y_pred)
        conf_mat = confusion_matrix(y_bn[test_ids], y_pred)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat
    
    return avg_accuracy / nfolds, avg_precision / nfolds, avg_recall / nfolds, avg_fscore / nfolds, avg_conf_mat / nfolds


def get_mnist_testset():
    x_train = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    x_train = np.array([x.flatten() for x in x_train])
    y_train = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    return x_train, y_train

def get_mnist_trainset():
    x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    x_train = np.array([x.flatten() for x in x_train])
    y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    return x_train, y_train

def merge_mnist_set(trainset, testset):
    """Merges two sets and returns a superset containing the two sets appended."""
    return np.append(trainset, testset, axis=0)

def log_reg_pred_kclass(thetas, x):
    return np.array([np.argmax([np.dot(theta, x_elem) for theta in thetas]) for x_elem in x])

def log_reg_pred_kclass_single(thetas, x):
    return np.argmax([np.dot(theta, x) for theta in thetas])

def hyp_kclass(thetas, j, x):
    return np.exp(np.dot(thetas[j], x)) / (np.sum([np.exp(np.dot(theta, x)) for theta in thetas]) + .001)

def error_delta_kclass(thetas, j, x, y):
    mem = 0
    if j == y: mem = 1
    return (hyp_kclass(thetas, j, x) - mem) * x


def log_reg_trainer_kclass(x_train, y_train, learn_rate=.01, min_accuracy=.64):
    """Trains a k-class logistic regression classifier on input training data.
    Basically finds values of theta."""
    thetas = np.array([np.ones(x_train.shape[1])] * len(set(y_train)))
    acc = 0
    n_iter = 1
    while acc < min_accuracy:
        i = 0
        for x in x_train:
            for j in xrange(thetas.shape[0]):
                err_delta = error_delta_kclass(thetas, j, x, y_train[i])
                thetas[j] -= err_delta * learn_rate
            i += 1
        preds = log_reg_pred_kclass(thetas, x_train)
        acc = accuracy_score(y_train, preds)
        n_iter += 1
    return thetas


def softmax(arr, j):
    if arr[j] < 10:
        res = np.exp(arr[j]) / np.sum(np.exp(arr))
    else:
        arr -= arr.max()
        res = np.exp(arr[j]) / np.sum(np.exp(arr))
    return res

def computeZs(w, x_train):
    z = []
    for i in xrange(x_train.shape[0]):
        z.append(np.array([sigmoid(w[j], x_train[i]) for j in xrange(w.shape[0])]))
    poly = PolynomialFeatures(1)
    z = poly.fit_transform(z)
    return np.array(z)

def computeYs(z, v):
    prods = np.dot(z, v.T)
    return np.array([[softmax(prod, i) for i in xrange(prods.shape[1])] for prod in prods])
    
def neural_likelihood(x_train, y_train, y_preds):
    k = len(set(y_train))
    m = x_train.shape[0]
    l = 1. * -np.sum([np.sum([np.log(y_preds[i][j] + .001) for j in xrange(k) if y_train[j] == j]) for i in xrange(m)])
    return l


def neural_predict_probs(x, w, v):
    z = computeZs(w, x)
    y = computeYs(z, v)
    return y

def neural_predict(x, w, v):
    preds_train = neural_predict_probs(x, w, v)
    return np.argmax(preds_train, axis=1)

def neural_train_test(x_mnist, y_mnist, hidden_units=200, momentum=.0001, learn_rate=.0005, degree=1, nfolds=4, limit=None):
    if limit is not None:
        x_mnist = x_mnist[:limit]
        y_mnist = y_mnist[:limit]
    n = x_mnist.shape[1]
    h = hidden_units
    k = len(set(y_mnist))
    v = np.ones([k, (1 + h)])

    prev_delta_v = np.zeros([k, (1 + h)])
    prev_delta_w = np.zeros([h, (1 + n)])

    w = np.zeros([h, (1 + n)])
    kf = KFold(y_mnist.shape[0], n_folds=nfolds, shuffle=True)
    poly = PolynomialFeatures(degree)
    x_mnist = poly.fit_transform(x_mnist)
    z = []

    accs_iters = []; precs_iters = []; recs_iters = []; fscores_iters = []

    flg = True
    iter = 1
    for train_ids, test_ids in kf:
        if flg == False:
            break
        flg = False
        x_train = x_mnist[train_ids]
        y_train = y_mnist[train_ids]
        x_test = x_mnist[test_ids]
        y_test = y_mnist[test_ids]
        y_oneHot = np.zeros([y_train.shape[0], 3]) #Converting to one-hot encoding
        for i in xrange(y_oneHot.shape[0]):
            y_oneHot[i][y_train[i]] = 1

        prev_l = 0; l = 5
        
        while True:
            z = computeZs(w, x_train)
            y = computeYs(z, v)
            diffs = y - y_oneHot
            delta_v = np.array([np.sum([d[i] * z[i] for i in xrange(d.shape[0])], axis=0) for d in diffs.T])
            v -= delta_v * learn_rate + prev_delta_v * momentum
            prev_delta_v = copy.deepcopy(delta_v)

            diffs_w_0 = np.dot(diffs, v)
            z_coeffs = z * (1 - z)
            diffs_w_1 = np.array([[x_train[i] * z_coeffs[i][j] for i in xrange(x_train.shape[0])] for j in xrange(h)])
            delta_w = np.array([np.sum([diffs_w_0[i,j] * diffs_w_1[j][i] for i in xrange(x_train.shape[0])], axis=0) for j in xrange(h)])
            w -= delta_w * learn_rate + prev_delta_w * momentum
            prev_delta_w = copy.deepcopy(delta_w)

            preds = neural_predict_probs(x_train, w, v)
            l = neural_likelihood(x_train, y_train, preds)
            preds_train = neural_predict(x_train, w, v)
            preds_test = neural_predict(x_test, w, v)
            acc_train = accuracy_score(y_train, preds_train)

            acc_test = accuracy_score(y_test, preds_test)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_test, preds_test)

            accs_iters.append(acc_test)
            precs_iters.append(precision1); recs_iters.append(recall1); fscores_iters.append(fscore1)

            print 'Iteration', iter, 'Likelihood =', l, 'Training Accuracy =', acc_train, 'Testing Accuracy =', acc_test

            if np.abs(prev_l - l) < .1 or iter > 100 or acc_test > .92:
                break
            prev_l = l
            iter += 1


        y_pred = neural_predict(x_test, w, v)

        acc = accuracy_score(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_test, y_pred)

        accs_iters = np.array(accs_iters)
        precs_iters = np.array(precs_iters)
        recs_iters = np.array(recs_iters)
        fscores_iters = np.array(fscores_iters)

    return acc, precision1, recall1, fscore1, conf_mat, accs_iters, precs_iters, recs_iters, fscores_iters, iter


def log_reg_kclass(x, y, nfolds=4, degree=1, limit=None):
    """Performs logistic regression experiments on Iris dataset for k class discrimination."""
    #print 'Training k Class classifier on Iris dataset'
    if limit is not None:
        print 'Considering only', limit, ' datapoints'
        x = x[:limit]
        y = y[:limit]

    #x /= x.max(axis=0)

    poly = PolynomialFeatures(degree)
    x = poly.fit_transform(x)
    num_classes = len(set(y))
    avg_accuracy =  0.; avg_precision = np.zeros(num_classes); avg_recall = np.zeros(num_classes); avg_fscore = np.zeros(num_classes); avg_conf_mat = np.zeros([num_classes, num_classes])
    kf = KFold(y.shape[0], n_folds=nfolds, shuffle=True)
    
    for train_ids, test_ids in kf:
        thetas = log_reg_trainer_kclass(x[train_ids], y[train_ids])
        y_pred = log_reg_pred_kclass(thetas, x[test_ids])
        acc = accuracy_score(y[test_ids], y_pred)
        avg_accuracy += acc
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y[test_ids], y_pred)
        conf_mat = confusion_matrix(y[test_ids], y_pred)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat
        lol=0
    
    return avg_accuracy / nfolds, avg_precision / nfolds, avg_recall / nfolds, avg_fscore / nfolds, avg_conf_mat / nfolds


if __name__ == "__main__":
    print 'Performing 2 class logistic regression experiments on Bank Note Dataset'
    f1 = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')
    data_bn = f1.readlines() #Bank Note Dataset
    #pickle.dump(data_bn, open('data_bn.dat', 'wb'))

    #data_bn = pickle.load(open('data_bn.dat', 'rb'))
    x_bn, y_bn = parse_data(data_bn)

    max_degree = 10
    accs = []; precs = []; recs = []; fscores = []; confs = []
    for deg in xrange(1, max_degree+1):
        avg_accuracy, avg_precision, avg_recall, avg_fscore, avg_conf_mat = log_reg(x_bn, y_bn, degree=deg)
        accs.append(avg_accuracy)
        precs.append(avg_precision)
        recs.append(avg_recall)
        fscores.append(avg_fscore)
        confs.append(avg_conf_mat)

    accs = np.array(accs); precs = np.array(precs); recs = np.array(recs); fscores = np.array(fscores)

    acc_gr, = plt.plot(range(1, max_degree+1), accs, '-o')
    plt.title('Accuracy vs Degree (Degree of Dimensions)\nMax. Accuracy at degree = ' + str(np.argmax(accs) + 1) +'\nAccuracy = ' + str(np.max(accs)))
    plt.xlabel('Degree')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, max_degree+1))
    plt.show()

    prec0, = plt.plot(range(1, max_degree+1), precs[:, 0], '-o')
    prec1, = plt.plot(range(1, max_degree+1), precs[:, 1], '-o')
    plt.title('Precision vs Degree (Degree of Dimensions)')
    plt.xlabel('Degree')
    plt.ylabel('Precision')
    plt.legend([prec0, prec1], ['Class 0', 'Class 1'])
    plt.xticks(range(1, max_degree+1))
    plt.show()

    rec0, = plt.plot(range(1, max_degree+1), recs[:, 0], '-o')
    rec1, = plt.plot(range(1, max_degree+1), recs[:, 1], '-o')
    plt.title('Recall vs Degree (Degree of Dimensions)')
    plt.xlabel('Degree')
    plt.ylabel('Recall')
    plt.legend([rec0, rec1], ['Class 0', 'Class 1'])
    plt.xticks(range(1, max_degree+1))
    plt.show()

    fs0, = plt.plot(range(1, max_degree+1), fscores[:, 0], '-o')
    fs1, = plt.plot(range(1, max_degree+1), fscores[:, 1], '-o')
    plt.title('F-Score vs Degree (Degree of Dimensions)')
    plt.xlabel('Degree')
    plt.ylabel('F-Score')
    plt.legend([fs0, fs1], ['Class 0', 'Class 1'])
    plt.xticks(range(1, max_degree+1))
    plt.show()

    print '\nPerforming experiments on k class classifier on Iris Dataset'
    f2 = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    data_iris = f2.readlines() #Iris Dataset
    #pickle.dump(data_iris, open('data_iris.dat', 'wb'))

    #data_iris = pickle.load(open('data_iris.dat', 'rb'))
    x_ir, y_ir = parse_iris_data(data_iris)

    accs = []; precs = []; recs = []; fscores = []; confs = []
    max_degree = 2
    for i in xrange(1, max_degree+1):
        avg_accuracy, avg_precision, avg_recall, avg_fscore, avg_conf_mat = log_reg_kclass(x_ir, y_ir, degree=i)
        accs.append(avg_accuracy)
        precs.append(avg_precision)
        recs.append(avg_recall)
        fscores.append(avg_fscore)
        confs.append(avg_conf_mat)

    accs = np.array(accs); precs = np.array(precs); recs = np.array(recs); fscores = np.array(fscores)
    accs_amax = np.argmax(accs)

    print 'Best Degree =', accs_amax+1
    print 'Best Accuracy =', accs[accs_amax]
    print 'Best Precisions =', precs[accs_amax]
    print 'Best Recalls =', recs[accs_amax]
    print 'Best F-Scores =', fscores[accs_amax]


    print '\nPerforming Neural Network experiments on MNIST dataset'

    x_mnist_train, y_mnist_train = get_mnist_trainset()
    x_mnist_test, y_mnist_test = get_mnist_testset()
    x_mnist = merge_mnist_set(x_mnist_train, x_mnist_test)
    y_mnist = merge_mnist_set(y_mnist_train, y_mnist_test)
    ids = y_mnist < 3
    y_mnist = y_mnist[ids]
    x_mnist = x_mnist[ids]

    shuffled_indices = np.arange(y_mnist.shape[0])
    np.random.shuffle(shuffled_indices)
    x_mnist = x_mnist[shuffled_indices]
    y_mnist = y_mnist[shuffled_indices]


    avg_accuracy, avg_precision, avg_recall, avg_fscore, avg_conf_mat, accs, precs, recs, fscores, max_degree = neural_train_test(x_mnist, y_mnist, hidden_units=125, momentum=.0001, learn_rate=.0005, limit=1000)
    print 'Accuracy =', avg_accuracy, '\nPrecision (3 Classes) =', avg_precision, '\nRecall (3 Classes) =', avg_recall, '\nF-Score (3 Classes) =', avg_fscore, '\nConfusion Matrix (3 Classes) =\n', avg_conf_mat

    acc_gr, = plt.plot(range(1, max_degree+1), accs, '-o')
    plt.title('Accuracy vs Number of Iterations\nMax. Accuracy at Iteration = ' + str(np.argmax(accs) + 1) +'\nAccuracy = ' + str(np.max(accs)))
    plt.xlabel('#Iterations')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, max_degree+1))
    plt.show()

    prec0, = plt.plot(range(1, max_degree+1), precs[:, 0], '-o')
    prec1, = plt.plot(range(1, max_degree+1), precs[:, 1], '-o')
    prec2, = plt.plot(range(1, max_degree+1), precs[:, 2], '-o')
    plt.title('Precision vs Number of Iterations')
    plt.xlabel('#Iterations')
    plt.ylabel('Precision')
    plt.legend([prec0, prec1, prec2], ['Class 0', 'Class 1', 'Class 2'], loc=4)
    plt.xticks(range(1, max_degree+1))
    plt.show()

    rec0, = plt.plot(range(1, max_degree+1), recs[:, 0], '-o')
    rec1, = plt.plot(range(1, max_degree+1), recs[:, 1], '-o')
    rec2, = plt.plot(range(1, max_degree+1), recs[:, 2], '-o')
    plt.title('Recall vs Number of Iterations')
    plt.xlabel('#Iterations')
    plt.ylabel('Recall')
    plt.legend([rec0, rec1, rec2], ['Class 0', 'Class 1', 'Class 2'], loc=4)
    plt.xticks(range(1, max_degree+1))
    plt.show()

    fs0, = plt.plot(range(1, max_degree+1), fscores[:, 0], '-o')
    fs1, = plt.plot(range(1, max_degree+1), fscores[:, 1], '-o')
    fs2, = plt.plot(range(1, max_degree+1), fscores[:, 1], '-o')
    plt.title('F-Score vs Number of Iterations')
    plt.xlabel('#Iterations')
    plt.ylabel('F-Score')
    plt.legend([fs0, fs1, fs2], ['Class 0', 'Class 1', 'Class 2'], loc=4)
    plt.xticks(range(1, max_degree+1))
    plt.show()