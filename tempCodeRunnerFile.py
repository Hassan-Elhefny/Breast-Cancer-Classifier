#import important libraries
from typing import Sized
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#add header and image to the web page
st.write("""
# Breast Cancer Classifier
""")

#add image
st.image("breast_cancer.jpg")

#show breast cancer dataframe from sklearn
breast_df = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
st.header("Breast Data Frame")
st.write(breast_df)

#plot the distribution using TSNE
st.header("Distribution Of Original Breast Cancer Data")
tsne = TSNE(n_components=2, random_state=0)
tsne_features = tsne.fit_transform(breast_df)
fig, ax = plt.subplots()
sns.scatterplot(x = tsne_features[:,0], y= tsne_features[:,1], c=load_breast_cancer().target)
plt.title("Breast Cancer Before PCA")
plt.xlabel("TSNE Feature 0")
plt.ylabel("TSNE Feature 1")
plt.legend(['malignant', 'benign'])
st.pyplot(fig)

#plot data after pca
st.header("Distribution Of Breast Cancer Data After PCA")
pca = PCA(n_components=12, random_state=0)
new_breast_df = pca.fit_transform(breast_df)
tsne_features_pca = tsne.fit_transform(new_breast_df)
fig, ax = plt.subplots()
sns.scatterplot(x = tsne_features_pca[:,0], y= tsne_features_pca[:,1], c=load_breast_cancer().target)
plt.title("Breast Cancer After PCA")
plt.xlabel("TSNE Feature 0")
plt.ylabel("TSNE Feature 1")
plt.legend(['malignant', 'benign'])
st.pyplot(fig)

#get features values from user
st.sidebar.header("Feature Values")
feature_1 = st.sidebar.slider("Feature 1",-8.631423e+02,3.867178e+03,1.867178e+03)
feature_2 = st.sidebar.slider("Feature 2",-6.715323e+02,7.396209e+02,5.396209e+02)
feature_3 = st.sidebar.slider("Feature 3",-6.629304e+01,3.517681e+02,2.517681e+02)
feature_4 = st.sidebar.slider("Feature 4",-2.757122e+01,3.253408e+01,1.253408e+01)
feature_5 = st.sidebar.slider("Feature 5",-2.382460e+01,3.200049e+01,0.200049e+01)
feature_6 = st.sidebar.slider("Feature 6",-6.010343e+00,6.687090e+00,3.687090e+00)
feature_7 = st.sidebar.slider("Feature 7",-5.439203e+00,6.757358e+00,2.757358e+00)
feature_8 = st.sidebar.slider("Feature 8",-5.439203e+00,4.304806e+00,3.304806e+00)
feature_9 = st.sidebar.slider("Feature 9",-1.612797e+00,2.459715e+00,0.459715e+00)
feature_10 = st.sidebar.slider("Feature 10",-9.832591e-01,2.589520e+00,1.589520e+00)
feature_11 = st.sidebar.slider("Feature 11",-8.430705e-01,1.060031e+00,1.060031e+00)
feature_12 = st.sidebar.slider("Feature 12",-4.198778e-01,5.424232e-01,4.424232e-01)

dic_selected_features = {
    "Feature 1" : feature_1,
    "Feature 2" : feature_2,
    "Feature 3" : feature_3,
    "Feature 4" : feature_4,
    "Feature 5" : feature_5,
    "Feature 6" : feature_6,
    "Feature 7" : feature_7,
    "Feature 8" : feature_8,
    "Feature 9" : feature_9,
    "Feature 10" : feature_10,
    "Feature 11" : feature_11,
    "Feature 12" : feature_12,
}

#build a dataframe of selected features
selected_features_df = pd.DataFrame(dic_selected_features, index=[0])

#show selected features
st.header("Selected Features")
st.write(selected_features_df)


#load saved classifier
extra_trees_clf = pickle.load(open('extra_trees_clf.pkl','rb'))