from sklearn.neural_network import BernoulliRBM



x = [[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]]
y = [0,0,0,1,1,1]






'''
count = 72983
sample_range = [int(float(count)*float(i)/10.0) for i in range(11)]
for i in sample_range:
    print i-i%10


x = [[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]]
y = [0,0,0,1,1,1]
train_data = np.array(x)
train_class = np.array(y)
print train_class.shape
print train_data.shape
print "+++++++"
kf = KFold(len(x), n_folds=2, shuffle=True)
for train, test in kf:
    print "*"
    print train_data[train,:]
    print train_data[test,:]


clf = svm.SVC(probability=True, kernel="linear")
clf.fit(x,y)
print clf.predict_proba([[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]])
a = np.array(clf.predict_proba([[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]]))
print clf.predict([[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]])
prediction  = [1 if conf[1]>conf[0] else 0 for conf in a]
print prediction
print "++++++++++++++++++"
print kcv.(clf, x, y, cv=2)

x = [[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]]
y = [0,0,0,1,1,1]
clf = svm.SVC()
clf.fit(x,y)
print clf.predict([[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]])
'''