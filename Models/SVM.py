
#SVM Classifier
from sklearn import svm


# In[5]:

def train_model(features,labels,kernal="linear",c=1,degree=2,coef=0):
    if kernal=="linear":
        #class_weight is balanced bcz we have more positives than negatives and nutrals
        model = svm.LinearSVC(C=c, class_weight="balanced")
    elif kernal=="poly":
        model = svm.SVC(C=c,kernal=kernal,degree=degree,coef=coef)
    model.fit(features,labels)
    return model



def predict(features, model):
    return model.predict(features)


# In[13]:

if __name__ == '__main__':
    import numpy as np
    #[x1, x2, x3,.....xm]
    #xi = [f1 f2 f3 f4 ....fn]
    X = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])
    y = [0,1,0,1,0,1]
    model = train_model(X,y)
    print predict([[2,3],[10,11]],model)


# In[ ]:



