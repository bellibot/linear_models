import sys, os, time, joblib
import numpy as np
import nevergrad as ng
import matplotlib.pyplot as plt
        
    
def _rect(A,T,t0,t_grid):
    template = np.zeros(t_grid.shape[0])
    condition = np.abs((t_grid-t0)/T)  
    template[condition<0.5] = A
    return template
    

def _tri(A,T,t0,t_grid):
    template = A*(1-2*np.abs((t_grid-t0)/T))
    condition = np.abs((t_grid-t0)/T)  
    template[condition>=0.5] = 0
    return template
    

def _objective_hard(template_function,T,t0,X,Y,t_grid,coeff):      
    S = []       
    for _T,_t0 in zip(T,t0):
        unit_template = template_function(1,_T,_t0,t_grid)
        S.append(coeff*np.array([np.dot(x,unit_template) for x in X]))
    S = np.array(S).transpose()
    sigma_SS = np.array([np.outer(S[i,:],S[i,:]) for i in range(S.shape[0])]).sum(axis=0)
    sigma_yS = np.matmul(Y,S)
    inv_sigma_SS = np.linalg.pinv(sigma_SS)
    A = np.matmul(inv_sigma_SS,sigma_yS)
    template = np.zeros(len(t_grid))
    for i in range(len(A)): 
        template += template_function(A[i],T[i],t0[i],t_grid)
    Xtemplate = coeff*np.array([np.dot(x,template) for x in X])
    return np.square(Y-Xtemplate).sum()

    
def _objective_beta(template_function,T,t0,X,Y,beta,lambd,t_grid,coeff):
    S = []       
    for _T,_t0 in zip(T,t0):
        unit_template = template_function(1,_T,_t0,t_grid)
        S.append(coeff*np.array([np.dot(x,unit_template) for x in X]))
    S = np.array(S).transpose()
    sigma_yS = np.matmul(Y,S)
    sigma_SS = np.array([np.outer(S[i,:],S[i,:]) for i in range(S.shape[0])]).sum(axis=0)
    G = []
    for _T,_t0 in zip(T,t0):
        G.append(template_function(1,_T,_t0,t_grid))
    G = np.array(G)
    sigma_betaG = coeff*np.matmul(G,beta)
    sigma_GG = coeff*np.matmul(G,G.transpose())
    inv_sigma_SSGG = np.linalg.pinv(sigma_SS+lambd*sigma_GG)
    A = np.matmul(inv_sigma_SSGG,sigma_yS+lambd*sigma_betaG)
    template = np.zeros(len(t_grid))
    for i in range(len(A)): 
        template += template_function(A[i],T[i],t0[i],t_grid)
    Xtemplate = coeff*np.array([np.dot(x,template) for x in X])
    return np.square(Y-Xtemplate).sum() + lambd*np.square(beta-template).sum()
    

def optimize(X,Y,beta,parameters,solver,random_state,n_jobs=1):
    np.random.seed(random_state)
    seeds = np.random.randint(0,1000,size=parameters['n_global_search'])
    results = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(global_optimizer)(X,Y,beta,parameters,solver,seed) for seed in seeds)
    best_index = 0
    best_loss = results[0][4]
    for i in range(1,len(results)):
        if results[i][4]<best_loss:
            best_index = i
            best_loss = results[i][4]            
    return results[i][0],results[i][1],results[i][2],results[i][3]


def global_optimizer(X,Y,beta,parameters,solver,seed):
    np.random.seed(seed)
    t_grid = np.linspace(parameters['begin_t'],parameters['end_t'],parameters['n_points_t'])
    coeff = (parameters['end_t']-parameters['begin_t'])/len(t_grid)
    budget = parameters['budget']
    num_workers = parameters['num_workers']
    n_shapes = parameters['n_shapes']
    T_init = np.random.uniform(0.01,2,n_shapes)
    t0_init = np.random.uniform(-1,1,n_shapes)
    T = ng.p.Array(init=T_init).set_bounds(0.01,2)
    t0 = ng.p.Array(init=t0_init).set_bounds(-1,1)
    if parameters['shape_type']=='rect':
        template_function = _rect
    elif parameters['shape_type']=='tri':
        template_function = _tri
    if beta is None:
        instrum = ng.p.Instrumentation(template_function,T,t0,X,Y,t_grid,coeff)
        obj = _objective_hard
    else:
        lambd = parameters['lambda']
        instrum = ng.p.Instrumentation(template_function,T,t0,X,Y,beta,lambd,t_grid,coeff)
        obj = _objective_beta  
    if solver=='OnePlusOne':
        optimizer = ng.optimizers.OnePlusOne(parametrization=instrum,budget=budget,num_workers=num_workers)
    elif solver=='RandomSearch':
        optimizer = ng.optimizers.RandomSearch(parametrization=instrum,budget=budget,num_workers=num_workers)
    elif solver=='PSO':
        optimizer = ng.optimizers.PSO(parametrization=instrum,budget=budget,num_workers=num_workers)
    elif solver=='DE':
        optimizer = ng.optimizers.TwoPointsDE(parametrization=instrum,budget=budget,num_workers=num_workers)
    optimizer.parametrization.random_state = np.random.RandomState(seed)
    recommendation = optimizer.minimize(obj)   
    T = recommendation.args[1]
    t0 = recommendation.args[2]
    S = []
    for _T,_t0 in zip(T,t0):
        unit_template = template_function(1,_T,_t0,t_grid)
        S.append(coeff*np.array([np.dot(x,unit_template) for x in X]))
    S = np.array(S).transpose()
    sigma_SS = np.array([np.outer(S[i,:],S[i,:]) for i in range(S.shape[0])]).sum(axis=0)
    sigma_yS = np.matmul(Y,S)
    if beta is None:
        inv_sigma_SS = np.linalg.pinv(sigma_SS)
        A = np.matmul(inv_sigma_SS,sigma_yS)
        min_ = _objective_hard(template_function,T,t0,X,Y,t_grid,coeff)
    else:
        sigma_SS = np.array([np.outer(S[i,:],S[i,:]) for i in range(S.shape[0])]).sum(axis=0)
        G = []
        for _T,_t0 in zip(T,t0):
            G.append(template_function(1,_T,_t0,t_grid))
        G = np.array(G)
        sigma_betaG = coeff*np.matmul(G,beta)
        sigma_GG = coeff*np.matmul(G,G.transpose())
        inv_sigma_SSGG = np.linalg.pinv(sigma_SS+lambd*sigma_GG)
        A = np.matmul(inv_sigma_SSGG,sigma_yS+lambd*sigma_betaG)
        min_ = _objective_beta(template_function,T,t0,X,Y,beta,lambd,t_grid,coeff)
    template = np.zeros(len(t_grid))
    for i in range(len(A)): 
        template += template_function(A[i],T[i],t0[i],t_grid) 
    return (template,A,T,t0,min_)
    
