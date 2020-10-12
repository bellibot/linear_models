import sys, time
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

                                                                          
class SmoothLinearModel:
    def __init__(self,estimator_type,solver,pen_flag='d2',fit_intercept=True,scale=True,max_iter=100,class_weight='balanced'):
            if not(estimator_type=='regressor' or estimator_type=='classifier'):
                print('Illegal estimator_type, must be classifier or regressor')
                sys.exit() 
            self.estimator_type = estimator_type
            self.pen_types = ['d1','d2']  
            if pen_flag in self.pen_types:
                self.pen_flag = pen_flag
            else:
                print('Illegal Penalization Flag == {}'.format(pen_flag))
                print('Terminating...')
                sys.exit()        
            self.legal_solvers = ['ipopt']
            if solver in self.legal_solvers:
                self.solver = solver
            else:
                print('Illegal Solver == {}'.format(solver))
                print('Terminating...')
                sys.exit()
            self.fit_intercept = fit_intercept
            self.class_weight = class_weight
            self.parameters = {}
            self.lambd = None
            self.max_iter = max_iter
            self.beta = None
            self.intercept = None
            self.scale = scale
            self.scaler = None

    
    def set_parameters(self,parameters):
        self.lambd = parameters['lambda']
        self.parameters['lambda'] = self.lambd
        if self.lambd<=0:
            print('Warning: Lambda <= 0')
        
    
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
        self._solve_optim(X,Y)
        return self

        
    def _solve_optim(self,X,Y):
        P = X.shape[1]
        N = X.shape[0]
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0,P-1)
        model.beta = pyo.Var(model.P, domain=pyo.Reals)
        if self.fit_intercept:
            model.intercept = pyo.Var(domain=pyo.Reals,initialize=0)
        for p in range(P):
            model.beta[p]= 0
        
        if self.estimator_type=='regressor':
            model.obj = pyo.Objective(expr=self._obj_expr_regr(model,X,Y), sense=pyo.minimize)
        else:
            model.obj = pyo.Objective(expr=self._obj_expr_class(model,X,Y), sense=pyo.minimize)
            
        opt = SolverFactory(self.solver,options={'max_iter':self.max_iter}) 
        opt.solve(model)
        
        beta = []
        for p in range(P):
            beta.append(pyo.value(model.beta[p]))   
        self.beta = np.array(beta)
        if self.fit_intercept:
            self.intercept = pyo.value(model.intercept)
        else:
            self.intercept = np.mean(Y)


    def _obj_expr_regr(self,model,X,Y):
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P)))**2 for j in range(N))
        if self.pen_flag=='d2':
            pen = P*P*sum((model.beta[p] -2*model.beta[p+1] + model.beta[p+2])**2 for p in range(P-2))
        elif self.pen_flag=='d1':
            pen = P*sum((model.beta[p]-model.beta[p+1])**2 for p in range(P-1))
        return obj + self.lambd*pen
        

    def _obj_expr_class(self,model,X,Y):
        N = X.shape[0]
        P = X.shape[1]
        if self.class_weight=='balanced': 
            unique_weights = N/(len(np.unique(Y))*np.bincount(Y))
            weights = np.zeros(N)
            for i in np.unique(Y):
                weights[np.where(Y==i)] = unique_weights[i]
        else:
            weights = np.ones(N)
        if self.fit_intercept:
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.beta[p] for p in range(P))+model.intercept) - pyo.log(1+pyo.exp(model.intercept+sum(X[j,p]*model.beta[p] for p in range(P)))))  for j in range(N))
        else:
            obj = -sum(weights[j]*(Y[j]*(sum(X[j,p]*model.beta[p] for p in range(P))) - pyo.log(1+pyo.exp(sum(X[j,p]*model.beta[p] for p in range(P)))))  for j in range(N))
        if self.pen_flag=='d2':
            pen = P*P*sum((model.beta[p] -2*model.beta[p+1] + model.beta[p+2])**2 for p in range(P-2))
        elif self.pen_flag=='d1':
            pen = P*sum((model.beta[p]-model.beta[p+1])**2 for p in range(P-1))
        return obj + self.lambd*pen
        
                
