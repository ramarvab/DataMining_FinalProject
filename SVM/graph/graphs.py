import numpy
import numpy as np
import matplotlib.pyplot as plt

class plot(object):

    def __init__(self, size, k):

        self.size = size
        self.k = k
        self.majors = ["Simple no Repl", "Simple Repl", "Stratified no Repl", "Stratified Repl"]
        self.majors_data = []

        self.points = size
        self.categories = 2
        self.class_reverse_dict = {0: 0, 1: 1}
        self.class_dict = {0: 0, 1: 1}

        self.majors_data.append(numpy.loadtxt("../output/%s_%s_%s_%s.csv" % ("simple", "withoutReplacemnt", size, k), delimiter=' '))
        self.majors_data.append(numpy.loadtxt("../output/%s_%s_%s_%s.csv" % ("simple", "withReplacemnt", size, k), delimiter=' '))
        self.majors_data.append(numpy.loadtxt("../output/%s_%s_%s_%s.csv" % ("stratified", "withoutReplacemnt", size, k), delimiter=' '))
        self.majors_data.append(numpy.loadtxt("../output/%s_%s_%s_%s.csv" % ("stratified", "withReplacemnt", size, k), delimiter=' '))

        self.majors_values = []
        for i in range(len(self.majors_data)):
            self.majors_values.append(self.get_measure_values(self.majors_data[i],i))


    def plot_accuracies(self):
        accuracies = [simple_accuracy for simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data in self.majors_values]

        width = 0.50
        fig = plt.figure(figsize=(10,8))
        plt.bar(range(len(accuracies)), accuracies, width, color="green", align= 'center', label = 'Simple Accuracy')
        plt.xticks(range(len(accuracies)), self.majors)
        plt.xlabel('Method Used')
        plt.ylabel('Accuracy')
        plt.title('Simple Accuracy %s %s\n' % (self.size, self.k))
        plt.ylim([max(min(accuracies)-0.1,0),max(accuracies)+0.1])

        plt.legend()
        #plt.show()
        fig.savefig('generated/Accuracy_%s_%s' % (self.size, self.k), bbox_inches='tight')
        plt.close()

    def plot_roc(self):
        roc_data = [roc_data for simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data in self.majors_values]

        for meth in range(3):
            fig = plt.figure(figsize=(10,8))
            roc_method = roc_data[meth]
            for cat in range(self.categories):
                a = [z for z, m in roc_method[cat]]
                b = [m for z, m in roc_method[cat]]
                plt.plot(b, a, color=np.random.rand(3,1), marker='o', label=self.class_reverse_dict[cat])
            plt.plot([0,1], [0,1], color='red', marker='o',linestyle='--', label='random')
            plt.xlabel('FP rate')
            plt.ylabel('TP rate')
            plt.title('ROC %s %s\n' % (self.size, self.k))
            plt.ylim([0,1])
            plt.xlim([0,1])

            plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))

            # plt.show()
            fig.savefig("generated/ROC_%s_%s" % (self.size, self.k), bbox_inches='tight')
            plt.close()

    def plot_baccuracies(self):
        bacc = [balanced_accuracy for simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data in self.majors_values]

        for meth in range(3):

            accuracies = bacc[meth]
            width = 0.50
            fig = plt.figure(figsize=(10,8))
            plt.bar(range(len(accuracies)), accuracies, width, color="red", align= 'center', label = 'Balanced Accuracy')
            cats = [self.class_reverse_dict[c] for c in range(self.categories)]
            plt.xticks(range(len(accuracies)), cats)
            plt.xlabel('Category')
            plt.ylabel('Balanced Accuracy')
            plt.title('Balanced Accuracy %s %s\n' % (self.size, self.k))
            plt.ylim([max(min(accuracies)-0.1,0),max(accuracies)+0.1])

            plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))

            #plt.show()
            fig.savefig("generated/BACC_%s_%s" % (self.size, self.k), bbox_inches='tight')
            plt.close()

    def plot_f1(self):
        f1 = [f1_score for simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data in self.majors_values]

        for meth in range(3):

            accuracies = f1[meth]
            width = 0.50
            fig = plt.figure(figsize=(10,8))
            plt.bar(range(len(accuracies)), accuracies, width, color="yellow", align= 'center', label = 'F1 Score')
            cats = [self.class_reverse_dict[c] for c in range(self.categories)]
            plt.xticks(range(len(accuracies)), cats)
            plt.xlabel('Category')
            plt.ylabel('f1 Score')
            plt.title('f1 Score %s %s\n' % (self.size ,self.k))
            plt.ylim([max(min(accuracies)-0.1,0),max(accuracies)+0.1])

            plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))

            #plt.show()
            fig.savefig("generated/f1_%s_%s" % (self.size, self.k), bbox_inches='tight')
            plt.close()

    def plot_pr(self):
        pr_data = [pr_data for simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data in self.majors_values]

        for meth in range(3):
            fig = plt.figure(figsize=(10,8))
            pr_method = pr_data[meth]
            for cat in range(self.categories):
                a = [z for z, m in pr_method[cat]]
                b = [m for z, m in pr_method[cat]]
                #a.sort(reverse=True)
                #b.sort()
                #a=[1]+a
                #b=[0]+b
                plt.plot(b, a, color=np.random.rand(3,1), marker='o', label=self.class_reverse_dict[cat])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve %s %s\n' % (self.size, self.k))
            plt.ylim([0,1])
            plt.xlim([0,1])

            plt.legend(loc='best')#, bbox_to_anchor=(1, 0.5))

            # plt.show()
            fig.savefig("generated/PR_%s_%s" % (self.size, self.k), bbox_inches='tight')
            plt.close()

    def get_roc_data(self, cat, acc_data):
        values = sorted(list(set(numpy.array(acc_data[:,2+cat:3+cat]).reshape(-1,).tolist())))

        # X-axis -> False Positive
        # Y-axis -> True Positive
        # 0 - TP
        # 1 - FP
        # 2 - TN
        # 3 - FN
        confusion_matrices = numpy.zeros(shape=(len(values),4), dtype=int)

        maxi = max(values)
        mini = min(values)
        diff = maxi - mini
        values= [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        values = [mini + diff*i for i in values]

        for i in range(len(values)):
            for row in acc_data:
                given_cat = -1
                if row[2+cat] >= values[i]:
                    given_cat = cat
                if row[1] == cat:
                    if given_cat == cat:
                        confusion_matrices[i][0] += 1
                    else:
                        confusion_matrices[i][3] += 1
                else:
                    if given_cat != cat:
                        confusion_matrices[i][2] += 1
                    else:
                        confusion_matrices[i][1] += 1
        roc_points = []
        pr_points = []
        for row in confusion_matrices:
            # sensitivity, specificity = float(row[0])/float(row[0]+row[3]), float(row[2])/float(row[2]+row[1])
            #tp_rate, fp_rate = float(row[0])/float(row[0]+row[3]), float(row[1])/float(row[2]+row[1])

            if (row[0]+row[3]) != 0:
                tp_rate = float(row[0])/float(row[0]+row[3])
            else:
                tp_rate = 0
            if float(row[2]+row[1]) != 0:
                fp_rate = float(row[1])/float(row[2]+row[1])
            else:
                fp_rate = 0


            if float(row[0]+row[1]) != 0:
                precision = float(row[0])/float(row[0]+row[1])
            else:
                precision = 0
            if float(row[0]+row[3]) != 0:
                recall = float(row[0])/float(row[0]+row[3])
            else:
                recall = 0
            if recall == 0:
                precision = 1
            pr_points.append((precision, recall))
            roc_points.append((tp_rate, fp_rate))
        return roc_points, pr_points

    def get_measure_values(self,acc_data, measure):

        # calculating confusion matrix
        confusion_matrix = numpy.zeros(shape=(self.categories, 4), dtype=int)
        # 0 - TP
        # 1 - FP
        # 2 - TN
        # 3 - FN
        for cat in range(self.categories):
            for row in acc_data:
                if row[1] == cat:
                    if row[0] == cat:
                        confusion_matrix[cat][0] += 1
                    else:
                        confusion_matrix[cat][3] += 1
                else:
                    if row[0] != cat:
                        confusion_matrix[cat][2] += 1
                    else:
                        confusion_matrix[cat][1] += 1

        # calculating f1 score
        f1_score = []
        for cat in range(self.categories):
            if float(confusion_matrix[cat][0]+confusion_matrix[cat][1]) != 0:
                precision = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][1])
            else:
                precision = 0
            if float(confusion_matrix[cat][0]+confusion_matrix[cat][3]) != 0:
                recall = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
            else:
                recall = 0
            if (precision+recall) != 0:
                f1 = 2*precision*recall/(precision+recall)
            else:
                f1 = 0
            f1_score.append(f1)

        # calculating balanced accuracy
        balanced_accuracy = []
        for cat in range(self.categories):
            sensitivity = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
            specificity = float(confusion_matrix[cat][2])/float(confusion_matrix[cat][2]+confusion_matrix[cat][1])
            balanced_accuracy.append(float(sensitivity + specificity)/2.0)

        # Calculating simple accuracy using confusion matrix
        simple_accuracy = 0
        for cat in range(self.categories):
            simple_accuracy += confusion_matrix[cat][0]
        simple_accuracy = float(simple_accuracy)/float(acc_data.shape[0])



        # Calculating data for ROC curve
        roc_data = []
        pr_data = []
        for cat in range(self.categories):
            fodfsa = self.get_roc_data(cat, acc_data)
            roc_data.append(fodfsa[0])
            pr_data.append(fodfsa[1])

        return simple_accuracy, balanced_accuracy, f1_score, roc_data, pr_data


if __name__ == "__main__":
    ks = [3]
    with open("../input.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            if line:
                size = int(line)
                for k in ks:
                    p = plot(size, k)

                    p.plot_accuracies()
                    p.plot_baccuracies()
                    p.plot_f1()
                    p.plot_roc()
                    p.plot_pr()
