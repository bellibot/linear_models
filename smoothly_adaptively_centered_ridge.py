import sys, joblib
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

    
class SACR:
    def __init__(self,estimator_type,solver,fit_intercept=True,solver_max_iter=100,random_state=0,scale=True,class_weight='balanced'):
        if not(estimator_type=='regressor' or estimator_type=='classifier'):
            print('Illegal estimator_type, must be classifier or regressor')
            sys.exit() 
        self.estimator_type = estimator_type
        self.legal_solvers = ['ipopt']
        if solver in self.legal_solvers:
            self.solver = solver
        else:
            print('Illegal OLS Solver == {}'.format(solver))
            print('Terminating...')
            sys.exit()
        self.fit_intercept = fit_intercept
        self.solver_max_iter = solver_max_iter
        self.random_state = random_state
        self.scale = scale
        self.class_weight = class_weight
        self.parameters = {}
        self.beta = None
        self.w = None
        self.g = None
        self.intercept = None
        self.lambd = None
        self.phi = None
        self.scaler = None


    def set_parameters(self,parameters):
        self.parameters['lambda'] = parameters['lambda']
        self.parameters['phi'] = parameters['phi']
        self.lambd = parameters['lambda']
        self.phi = parameters['phi']
        

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
        self._init_ridge(X,Y)
        self._joint_ridge(X,Y)       
        return self


    def _init_ridge(self,X,Y):
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
            
        opt = SolverFactory(self.solver,options={'max_iter':self.solver_max_iter}) 
        opt.solve(model)
        
        beta = []
        for p in range(P):
            beta.append(pyo.value(model.beta[p]))   
        self.g = np.array(beta)
            
                            
    def _joint_ridge(self,X,Y):
        N = X.shape[0]
        P = X.shape[1]
        model = pyo.ConcreteModel()
        model.P = pyo.RangeSet(0,P-1)
        model.beta = pyo.Var(model.P, domain=pyo.Reals)
        model.w = pyo.Var(model.P, domain=pyo.NonNegativeReals)
        if self.fit_intercept:
            model.intercept = pyo.Var(domain=pyo.Reals,initialize=0)
        for p in range(P):
            model.beta[p]= 0
            model.w[p]= 0
                    
        def constraint_w_integral(model):
            return (1/P)*sum(model.w[p] for p in model.P) == 1
        model.constraint_w_integral = pyo.Constraint(rule=constraint_w_integral)
        
     
        if self.estimator_type=='regressor':
            model.obj = pyo.Objective(expr=self._obj_expr_joint_regr(model,X,Y), sense=pyo.minimize)
        else:
            model.obj = pyo.Objective(expr=self._obj_expr_joint_class(model,X,Y), sense=pyo.minimize)
            
        opt = SolverFactory(self.solver,options={'max_iter':self.solver_max_iter}) 
        opt.solve(model)
        
        beta = []
        w = []
        for p in range(P):
            beta.append(pyo.value(model.beta[p]))
            w.append(pyo.value(model.w[p]))   
        self.beta = np.array(beta)
        self.w = np.array(w)
        if self.fit_intercept:
            self.intercept = pyo.value(model.intercept)
        else:
            self.intercept = 0
     

    def _obj_expr_regr(self,model,X,Y):
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P)))**2 for j in range(N))
        pen = sum(model.beta[p]**2 for p in range(P))
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
        pen = sum(model.beta[p]**2 for p in range(P))
        return obj + self.lambd*pen


    def _obj_expr_joint_regr(self,model,X,Y):
        N = X.shape[0]
        P = X.shape[1]
        if self.fit_intercept:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P))-model.intercept)**2 for j in range(N))
        else:
            obj = sum((Y[j]-sum(X[j,p]*model.beta[p] for p in range(P)))**2 for j in range(N))
        pen1 = sum((model.beta[p]-self.g[p]*model.w[p])**2 for p in range(P))
        pen2 = P*P*sum((self.g[p]*model.w[p] -2*self.g[p+1]*model.w[p+1] + self.g[p+2]*model.w[p+2])**2 for p in range(P-2))
        return obj + self.phi*self.lambd*pen1 + (1-self.phi)*self.lambd*pen2


    def _obj_expr_joint_class(self,model,X,Y):
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
        pen1 = sum((model.beta[p]-self.g[p]*model.w[p])**2 for p in range(P))
        pen2 = P*P*sum((self.g[p]*model.w[p] -2*self.g[p+1]*model.w[p+1] + self.g[p+2]*model.w[p+2])**2 for p in range(P-2))
        return obj + self.phi*self.lambd*pen1 + (1-self.phi)*self.lambd*pen2

               
