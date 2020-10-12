import sys, time, joblib
import numpy as np
import global_optimization_library as globoptlib
from sklearn.preprocessing import StandardScaler,minmax_scale     
import matplotlib.pyplot as plt

    
class AATR:
    def __init__(self,dfo_solver,random_state=0,scale=True,tol=10**(-5),n_jobs=1):
        self.legal_shape_types = ['rect','tri']      
        self.dfo_solver = dfo_solver 
        self.estimator_type = 'regressor'    
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scale = scale
        self.parameters = {}
        self.beta = None
        self.intercept = None
        self.shape_template = None
        self.int_coeff = None
        self.A = None
        self.T = None
        self.t0 = None
        self.lambd = None
        self.scaler = None
        self.max_iter = None
        self.train_score_by_iter = []
        self.beta_by_iter = []
        self.shape_template_by_iter = []
        self.best_iter = None


    def set_parameters(self,parameters):
        self.lambd = parameters['lambda']
        if self.lambd<=0:
            print('Illegal Lambda == {} (<=0)'.format(parameters['lambda']))
            print('Terminating...')
            sys.exit()
        self.max_iter = parameters['max_iter']
        self.parameters['shape_type'] = parameters['template'][0]
        self.parameters['n_shapes'] = parameters['template'][1]
        self.shape_template = parameters['template'][2]
        self.A = parameters['template'][3]
        self.T = parameters['template'][4]
        self.t0 = parameters['template'][5]
        self.parameters['lambda'] = parameters['lambda']
        self.parameters['max_iter'] = parameters['max_iter']
        if not(self.parameters['shape_type'] in self.legal_shape_types):
            print('Illegal Shape Type == {}'.format(self.parameters['shape_type']))
            print('Terminating...')
            sys.exit()    
        self.parameters['n_global_search'] = parameters['n_global_search']
        self.parameters['num_workers'] = parameters['num_workers']
        self.parameters['budget'] = parameters['budget']
        if 'begin_t' in parameters.keys() and 'end_t' in parameters.keys():
            self.parameters['begin_t'] = parameters['begin_t']
            self.parameters['end_t'] = parameters['end_t']
        else:
            self.parameters['begin_t'] = -1
            self.parameters['end_t'] = 1 


    def score(self,X,Y):
        pred_Y = self.predict(X)
        return np.mean((pred_Y-Y)**2)
    
            
    def predict(self,X):
        if self.scale:
            X = self.scaler.transform(X)
        pred_Y = []
        for x in X:
            pred = self.int_coeff*np.dot(self.beta,x) + self.intercept
            pred_Y.append(pred)
        return np.array(pred_Y)
                
                
    def fit(self,X,Y):   
        self.parameters['n_points_t'] = X.shape[1]
        self.int_coeff = (self.parameters['end_t']-self.parameters['begin_t'])/X.shape[1]
        if self.scale:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        self._initialize(X,Y)         
        self._run(X,Y)
        return self
        
        
    def _initialize(self,X,Y):
        self.shape_template_by_iter.append(self.shape_template)
        self.beta,self.intercept = self._ridge(X,Y,self.shape_template)
        self.beta_by_iter.append(self.beta)
        self.train_score_by_iter.append(self._score(X,Y,self.beta,self.intercept))
        self.best_iter = 0   
    
        
    def _run(self,X,Y):
        start_time = time.time()
        i = 0
        best_score = self.train_score_by_iter[0]
        while i<self.max_iter:
            shape_template,A,T,t0 = self._compute_template(X,Y,self.beta)
            beta,intercept = self._ridge(X,Y,shape_template)
            score = self._score(X,Y,beta,intercept)
            self.beta_by_iter.append(beta)
            self.shape_template_by_iter.append(shape_template)
            self.train_score_by_iter.append(score)       
            if best_score-score>self.tol:
                i+=1
                best_score = score
                self.best_iter = i
                self.beta = beta
                self.intercept = intercept
                self.shape_template = shape_template
                self.A = A
                self.T = T
                self.t0 = t0
            else:
                i = self.max_iter
        elapsed = round((time.time()-start_time)/60,3)
        print('Run() Time: {} Minutes'.format(elapsed))
        
        
    def _ridge(self,X,Y,shape_template):
        intercept = Y.mean()
        A = np.matmul(self.int_coeff*X.transpose(),self.int_coeff*X) + self.lambd*self.int_coeff*np.eye(X.shape[1])
        b = np.matmul(self.int_coeff*X.transpose(),Y) + self.lambd*self.int_coeff*shape_template
        beta = np.linalg.solve(A,b)
        return beta,intercept
            

    def _compute_template(self,X,Y,beta):
        shape_template,A,T,t0 = globoptlib.optimize(X,Y-Y.mean(),beta,self.parameters,self.dfo_solver,self.random_state,self.n_jobs)
        return shape_template,A,T,t0

               
    def _score(self,X,Y,beta,intercept):
        pred_Y = []
        for x in X:
            pred = self.int_coeff*np.dot(beta,x) + intercept
            pred_Y.append(pred)
        pred_Y = np.array(pred_Y)
        return np.mean((pred_Y-Y)**2)    


