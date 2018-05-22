from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier


#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	[166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,55,39],
	[171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
	'male', 'female', 'male', 'female', 'male']


'''
	classifying using decision tree 
'''
clf = tree.DecisionTreeClassifier()		#clf short for classifier

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])
print("Prediction using Decision Tree:")
print(prediction)



'''
	classifying using Ensemble methods
'''
clf = RandomForestClassifier(n_estimators=10)		#what is n_estimators?
clf = clf.fit(X, Y)
prediction = clf.predict([[190,70,43]])
print("Prediction using Ensemble:")
print(prediction)


'''
	classifying using Support Vector Machine
'''
clf = svm.SVC()
clf.fit(X, Y)
prediction = clf.predict([[190,70,43]])
print("Prediction using Support Vector Machine:")
print(prediction)


'''
	classifying using Stochastic Gradient Descent
'''
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, Y)
prediction = clf.predict([[190,70,43]])
print("Prediction using Stochastic Gradient Descent:")
print(prediction)