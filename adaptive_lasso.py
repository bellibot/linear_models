import sys
import numpy as np
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Lasso,Ridge,LogisticRegression
from sklearn.metrics import balanced_accuracy_score

                                                                          
class AdaptiveLasso:
    def __init__(self,estimator_type,fit_intercept=True,multi_class='ovr',class_weight='balanced',scale=True,random_state=None,solver='saga',max_iter=2000,n_jobs=1):   
            if not(estimator_type=='regressor' or estimator_type=='classifier'):
                print('Illegal estimator_type, must be classifier or regressor')
                sys.exit() 
            self.estimator_type = estimator_type
            self.multi_class = multi_class
            self.class_weight = class_weight
            self.solver = solver 
            self.fit_intercept = fit_intercept
            self.parameters = {}
            self.lambd = None
            self.gamma = None
            self.beta_init = None
            self.beta = None
            self.intercept = None
            self.estimator = None
            self.scale = scale
            self.scaler = None
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.max_iter = max_iter

    
    def set_parameters(self,parameters):
        self.gamma = parameters['gamma']
        self.lambd = parameters['lambda']
        self.beta_init = parameters['beta_init'][1]
        self.parameters['gamma'] = self.gamma
        self.parameters['lambda'] = self.lambd
        self.parameters['beta_init'] = parameters['beta_init'][0] 
        if self.lambd<=0:
            print('Warning: Lambda <= 0')
        if self.gamma<=0:
            print('Warning: Gamma <= 0')
        
    
    def score(self,X,Y):
        pred_Y = self.predict(X)
        if self.estimator_type=='regressor':
            score = np.mean((pred_Y-Y)**2)
        else:
            score = balanced_accuracy_score(Y,pred_Y)    
        return score
            
            
    def predict(self,X):
        if self.scale:
            X = self.scaler.transform(X)
        pred_Y = []
        if self.estimator_type=='regressor':
            for x in X:
                pred = np.dot(self.beta,x) + self.intercept
                pred_Y.append(pred)
        else:
            for x in X:
                if np.dot(self.beta,x) + self.intercept > 0:
                    pred_Y.append(1)
                else:
                    pred_Y.append(0)                
        return np.array(pred_Y)
                            
                
    def fit(self,X,Y):         
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        if self.estimator_type=='regressor':
            recipr_w = np.power(np.abs(self.beta_init),self.gamma)
            for i in range(X.shape[0]):
                X[i,:] = X[i,:]*recipr_w
            estimator = Lasso(alpha=self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,max_iter=self.max_iter)
            estimator.fit(X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_*recipr_w
        else:
            recipr_w = np.power(np.abs(self.beta_init),self.gamma)
            for i in range(X.shape[0]):
                X[i,:] = X[i,:]*recipr_w
            estimator = LogisticRegression(penalty='l1',multi_class=self.multi_class,class_weight=self.class_weight,C=1/self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,solver=self.solver,max_iter=self.max_iter)
            estimator.fit(X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_[0]*recipr_w
        return self


