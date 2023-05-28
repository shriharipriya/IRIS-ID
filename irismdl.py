from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
iris=load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['Species'])
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.svm import SVC
svc=SVC()
svc.fit(xtrain,ytrain)
import pickle
pickle.dump(svc,open('irismdl.pkl','wb'))
