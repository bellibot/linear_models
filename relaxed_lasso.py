import sys
import numpy as np
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

                                                                          
class RelaxedLasso:
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
            self.phi = None
            self.beta = None
            self.intercept = None
            self.estimator = None
            self.scale = scale
            self.scaler = None
            self.random_state = random_state
            self.n_jobs = n_jobs
            self.max_iter = max_iter

    
    def set_parameters(self,parameters):
        self.phi = parameters['phi']
        self.lambd = parameters['lambda']
        self.parameters['phi'] = self.phi
        self.parameters['lambda'] = self.lambd
        if self.lambd<0:
            print('Warning: Lambda <=0')
        if self.phi<=0 or self.phi>1:
            print('Warning: Phi out of (0,1]')
        
    
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
        return self.estimator.predict(X)
                
                
    def fit(self,X,Y):         
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        if self.estimator_type=='regressor':
            estimator = Lasso(alpha=self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,max_iter=self.max_iter)
            estimator.fit(X,Y)
            zero_indexes = np.where(np.isclose(estimator.coef_,0))[0]
            X[:,zero_indexes]=0
            estimator = Lasso(alpha=self.phi*self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,max_iter=self.max_iter)
            self.estimator = estimator.fit(X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_
        else:
            estimator = LogisticRegression(penalty='l1',multi_class=self.multi_class,class_weight=self.class_weight,C=1/self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,solver=self.solver,max_iter=self.max_iter)
            estimator.fit(X,Y)
            zero_indexes = np.where(np.isclose(estimator.coef_[0],0))[0]
            X[:,zero_indexes]=0
            estimator = LogisticRegression(penalty='l1',multi_class=self.multi_class,class_weight=self.class_weight,C=1/(self.lambd*self.phi),fit_intercept=self.fit_intercept,random_state=self.random_state,solver=self.solver,max_iter=self.max_iter)
            self.estimator = estimator.fit(X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_[0]
        return self


