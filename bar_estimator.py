import sys
import numpy as np
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

                                                                          
class BARestimator:
    def __init__(self,estimator_type,fit_intercept=True,max_iter=5,multi_class='ovr',class_weight='balanced',scale=True,random_state=None,solver='saga',max_opt_iter=2000,n_jobs=1):   
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
            self.n_iters = 0
            self.max_opt_iter = max_opt_iter

    
    def set_parameters(self,parameters):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd
        self.parameters['n_iters'] = 0
        if self.lambd<0:
            print('Warning: Lambda <=0')
  
    
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
            estimator = Ridge(alpha=self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,max_iter=self.max_opt_iter)
            estimator.fit(X,Y)
            recipr_w = np.square(estimator.coef_)
            _X = np.zeros(X.shape)
            for i in range(X.shape[0]):
                _X[i,:] = X[i,:]*recipr_w
            estimator.fit(_X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_*recipr_w
            score = self._score(X,Y,self.beta,self.intercept)
            score_decreased = True
            k=0
            while k<self.max_iter and score_decreased:
                recipr_w = np.square(self.beta)
                _X = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    _X[i,:] = X[i,:]*recipr_w
                estimator.fit(_X,Y)
                beta = estimator.coef_*recipr_w
                intercept = estimator.intercept_
                tmp_score = self._score(X,Y,beta,intercept)
                if tmp_score>score:
                    score_decreased = False
                else:
                    self.beta = beta
                    self.intercept = intercept    
                    self.n_iters += 1
                    self.parameters['n_iters'] +=1
                    k+=1      
        else:
            estimator = LogisticRegression(penalty='l2',multi_class=self.multi_class,class_weight=self.class_weight,C=1/self.lambd,fit_intercept=self.fit_intercept,random_state=self.random_state,solver=self.solver,max_iter=self.max_opt_iter)
            estimator.fit(X,Y)
            recipr_w = np.square(estimator.coef_[0])
            _X = np.zeros(X.shape)
            for i in range(X.shape[0]):
                _X[i,:] = X[i,:]*recipr_w
            estimator.fit(_X,Y)
            self.intercept = estimator.intercept_ 
            self.beta = estimator.coef_[0]*recipr_w
            score = 1 - self._score(X,Y,self.beta,self.intercept)
            score_decreased = True
            k=0
            while k<self.max_iter and score_decreased:
                recipr_w = np.square(self.beta)
                _X = np.zeros(X.shape)
                for i in range(X.shape[0]):
                    _X[i,:] = X[i,:]*recipr_w
                estimator.fit(_X,Y)
                beta = estimator.coef_[0]*recipr_w
                intercept = estimator.intercept_
                tmp_score = 1 - self._score(X,Y,beta,intercept)
                if tmp_score>score:
                    score_decreased = False
                else:
                    self.beta = beta
                    self.intercept = intercept    
                    self.n_iters += 1
                    self.parameters['n_iters'] += 1
                    k+=1
        return self


    def _score(self,X,Y,beta,intercept):
        pred_Y = []
        if self.estimator_type=='regressor':
            for x in X:
                pred = np.dot(beta,x) + intercept
                pred_Y.append(pred)
            pred_Y = np.array(pred_Y)
            score = np.mean((pred_Y-Y)**2)
        else:
            for x in X:
                if np.dot(beta,x) + intercept > 0:
                    pred_Y.append(1)
                else:
                    pred_Y.append(0)
            score = balanced_accuracy_score(Y,pred_Y)
        return score 
        
        
