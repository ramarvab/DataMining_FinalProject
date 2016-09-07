import numpy as np
from Classifier import Classifier
from sklearn.cross_validation import KFold
import random
from time import time

class OutputGen(object):
    def __init__(self):
        self.data = np.loadtxt("../cleaned/training.csv", delimiter=',', dtype=int)

    def simple(self, k):
        self.calculate_results("whole","withoutReplacemnt", len(self.data), k, self.data)

    def sample_wo_r(self, count, k):
        data_points = random.sample(range(self.data.shape[0]), count)
        new_data = self.data[data_points,:]
        self.calculate_results("simple","withoutReplacemnt", count, k, new_data)

    def sample_w_r(self, count, k):
        data_points = [random.randint(0, count-1) for _ in range(count)]
        new_data = self.data[data_points,:]
        self.calculate_results("simple","withReplacemnt", count, k, new_data)

    def sample_s_wo_r(self, count, k):
        part_count = int(count/10)
        count = part_count*10
        sample_range = [int(float(len(self.data))*float(i)/10.0) for i in range(11)]
        data_points = []
        for i in range(len(sample_range)-1):
            data_points += random.sample(range(sample_range[i], sample_range[i+1]), part_count)
        new_data = self.data[data_points,:]
        self.calculate_results("stratified","withoutReplacemnt", count, k, new_data)

    def sample_s_w_r(self, count, k):
        part_count = int(count/10)
        count = part_count*10
        sample_range = [int(float(len(self.data))*float(i)/10.0) for i in range(11)]
        data_points = []
        for i in range(len(sample_range)-1):
            data_points += [random.randint(sample_range[i], sample_range[i+1]-1) for _ in range(part_count)]
        new_data = self.data[data_points,:]
        self.calculate_results("stratified","withReplacemnt", count, k, new_data)

    def calculate_results(self, method_type, replacement, count, k, data):
        start_time = time()
        kf = KFold(count, n_folds=k, shuffle=True)
        accuracy = 0
        for train, test in kf:
            conf_table = self.get_conf_table(data[train,:], data[test,:])
            results, acc = self.get_acc(conf_table, data[test,:])
            with open("../output/%s_%s_%s_%s.csv" % (method_type, replacement, count, k), "a") as f:
                np.savetxt(f, np.hstack([results, conf_table]))
            accuracy += acc
        accuracy /= k
        with open("../output/accuracy.csv", "a") as f:
            t = time() - start_time
            f.write("%s, %s, %s, %s, %s, %s\n" % (method_type, replacement, count, k, accuracy, t))

    def get_acc(self, conf_table, test):
        results = np.array(zip([1 if c[1] > c[0] else 0 for c in conf_table], test[:, [(test.shape[1]-1)]].reshape(-1,)))
        correct = 0
        for result in results:
            if result[0] == result[1]:
                correct += 1
        return results, float(correct)/float(len(results))

    '''
    def get_acc(self, conf_table, test):
        prediction = [1 if conf[1] > conf[0] else 0 for conf in conf_table]
        test_class = test[:, [(test.shape[1]-1)]].reshape(-1,)
        results = np.array(zip(prediction, test_class))
        total = len(results)
        correct = 0
        for result in results:
            if result[0] == result[1]:
                correct += 1
        return float(correct)/float(total)
    '''
    def get_conf_table(self, train, test):
        train_data = train[:,range(train.shape[1] - 1)]
        train_class = train[:, [(train.shape[1]-1)]].reshape(-1,)
        test_data = test[:,range(test.shape[1] - 1)]
        return Classifier.confidence_table(train_data, train_class, test_data)


if __name__ == "__main__":
    classifier = OutputGen()
    k = 5
    with open("../input.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                size = int(line)
                print "Starting for", size
                classifier.sample_wo_r(size, k)
                print "Done with simple without replacement"
                classifier.sample_w_r(size, k)
                print "Done with simple with replacement"
                classifier.sample_s_wo_r(size, k)
                print "Done with stratified without replacement"
                classifier.sample_s_w_r(size, k)
                print "Done with Stratified with replacement"
                print



'''
class OutputGen(object):
    def __init__(self):
        self.data = np.loadtxt("../cleaned/training.csv", delimiter=',', dtype=int)

    def simple(self):
        train = self.data
        print self.get_conf_table(train, train)

    def sample_wo_r(self, count, k):
        data_points = random.sample(range(self.data.shape[0]), count)
        new_data = self.data[data_points,:]

        kf = KFold(count, n_folds=k, shuffle=True)
        accuracy = 0
        for train, test in kf:
            accuracy += self.calculate_results("Simple","withoutReplacemnt", count, new_data[train,:], new_data[test,:])
        accuracy /= k
        self.write_acc("ping", accuracy)

    def sample_w_r(self, count, k):
        pass

    def sample_s_wo_r(self, count, k):
        pass

    def sample_s_w_r(self, count, k):
        pass

    def calculate_results(self, method_type, replacement, count, train, test):
        # get confidence table
        print "******"
        conf_table = self.get_conf_table(train, test)

        # store confidence table in file
        print type(conf_table)
        with open("../output/%s_%s_%s.csv" % (method_type, replacement, count), "a") as f:
            np.savetxt(f, conf_table)

        return self.get_acc(conf_table, test)

    def get_acc(self, conf_table, test):
        prediction = [1 if conf[1]>conf[0] else 0 for conf in conf_table]
        test_class = test[:, [(test.shape[1]-1)]].reshape(-1,)
        results = zip(prediction, test_class)
        total = len(results)
        correct = 0
        for result in results:
            if result[0] == result[1]:
                correct += 1
        return float(correct)/float(total)

    def get_conf_table(self, train, test):
        train_data = train[:,range(train.shape[1] - 1)]
        train_class = train[:, [(train.shape[1]-1)]].reshape(-1,)
        test_data = test[:,range(test.shape[1] - 1)]
        return Classifier.confidence_table(train_data, train_class, test_data)

    def write_acc(self, initial_part, acc):
        with open("../output/accuracy.csv", "a") as f:
            f.write(initial_part+", "+str(acc)+"\n")

if __name__ == "__main__":
    classifier = OutputGen()
    # classifier.simple()
    classifier.sample_wo_r(3000, 10)
'''


'''
class Classifier(object):
    def __init__(self):
        self.model = svm.SVC()
        self.data = np.loadtxt("../cleaned/training.csv", delimiter=',', dtype=int)
        self.model.fit(self.data[:,range(self.data.shape[1] - 1)], self.data[:,[(self.data.shape[1]-1)]].reshape(-1,))
        print self.data

    def get_acc(self):
        actual_classes = self.data[:,[(self.data.shape[1]-1)]].reshape(-1,)
        predicted_classes = self.model.predict(self.data[:,range(self.data.shape[1] - 1)])
        results = zip(actual_classes, predicted_classes)
        total = len(results)
        acc = 0
        for result in results:
            if result[1] == 1:
                print 'hi'
            if result[0] == result[1]:
                acc += 1
        print float(acc)/float(total)
        scores = kcv.cross_val_score(self.mo)

if __name__ == "__main__":
    classifier = Classifier()
    print classifier.get_acc()
'''