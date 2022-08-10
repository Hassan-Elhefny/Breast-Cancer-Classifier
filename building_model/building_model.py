#import important libraries and dataset of breast cancer
from struct import pack
from sklearn.datasets import load_breast_cancer
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

#load dataset
X, Y = load_breast_cancer(return_X_y=True)

#DM
pca = PCA(n_components=12, random_state=0)
X = pca.fit_transform(X)

#building model
model_clf = ExtraTreesClassifier()

#fit the model
model_clf.fit(X,Y)

#save the model
pickle.dump(model_clf, open('extra_trees_clf.pkl', 'wb'))
