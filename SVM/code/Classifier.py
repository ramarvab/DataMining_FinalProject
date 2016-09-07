from sklearn import svm


class Classifier(object):
    @staticmethod
    def confidence_table(train_data, train_class, test_data):

        model = svm.SVC(probability=True, kernel = 'linear')
        model.fit(train_data, train_class)
        return model.predict_proba(test_data)