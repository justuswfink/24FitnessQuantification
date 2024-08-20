

import pandas as pd           

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy import integrate
from scipy import stats
import random


# I occasionally use a logger to send a warning to the user
# todo: replace this by loguru?

import logging

logger = logging.getLogger()

import numpy as np
import matplotlib.pyplot as plt 
from bunch import Bunch

from numpy.linalg import norm as vector_norm

def check_remaining_resources_are_small(sol, Y = None, atol = 1e-4):
    """ Returns True if the available resources at the last time point are small.
    
    To define what is small, we demand that the expected change in selection coefficient 
    is small.
    
    INPUTS
    ------
    sol         ODESolution object, just as returned by `scipy.solve_ivp` or `get_ODE_solution`
    Y           2-tuple of the yield for the both strains, i.e. Y = (Y1, Y2),
                if not passed explicitly, tries to read this from the `sol` object
    atol        positive float, the maximum tolerance for small values of `x`
    rtol        positive float, the maximum relative tolerance for large values of `x
    
    """
    try: 
        Y1, Y2 = Y
    except:
        Y1, Y2 = sol.params['Y']
    
    y_0   = sol.y[:,0]    # extract initial state of the system
    y_f = sol.y[:,-1]   # extract final state of the system
    
    #asssert that resources are positive, if they are negative, we can stop anyways
    N1_f, N2_f, R_f = y_f
    if R_f <= 0: return True
    
    # calculate selection coefficient
    N1_0, N2_0, R_0 = y_0
    s = np.log(N2_f/N1_f) - np.log(N2_0/N1_0)
    
    
                  
    # if all remaining resources go to strain one, the population size is multiplied by a factor 1 + delta_1
    delta1 = R_f * Y1 / N1_f
    # if all remaining resources go to strain two, the population size is multiplied by a factor 1 + delta_1
    delta2 = R_f * Y2 / N2_f
    
    delta_s = np.maximum(np.log(1+delta1), np.log(1 + delta2))            # estimate maximum possible deviations
    
    rtol = 0 ## only want absolute error bound
    return np.isclose(a = s, b = s+delta_s, rtol=rtol, atol = atol)
    

from scipy.integrate import solve_ivp

def get_ODE_solution(problem, t_final = 1e3, adaptive_timewindow=True, timestep = 10, atol = 1e-3, rtol = 1e-4, scoeff_atol = 1e-5, **kwargs):
    """ Returns the solution to a generic ODE for the population sizes

    INPUTS
    ------
    problem              an object of the type returned by 'get_default_problem_M3' or 'get_default_problem_CRM'
    t_final              positive float, the final time until which the solution should be calculated (in hours)
    adaptive_timewindow  boolean, if True the simulation stops when we reach a steady state
    timestep             positive float, the step size for the adaptive time window
    atol                 positive small float, absolute tolerance as parameter for ODE solver `solve_ivp`
    rtol                 positive small float, relative tolerance as parameter for ODE solver `solve_ivp`
    **kwargs             additional keyword arguments passed to `solve_ivp`
    
    This function uses the built in ODE solver `scipy.integrate.solve_ivp`
    
    Note: When does the simulation stop? The vale `t_final` is the maximum possible time of simulation. 
    If in addition `adaptive_timewindow = True`, then the simulation proceeds in discrete chunks of time and stops
    as soon as we reach a steady state - which, in general, will be earlier than `t_final` and in the latest case 
    the timewindow ends at `t_final + timestep` where `timestep` is set below.
    
    The default value of `t_final` is 1000 hours (~40 days) which should be more than enough for the typical laboratory
    growth cycle.

    RETURNS
    -------
    sol         Bunch object as returned by scipy.integrate.solve_ivp

    """
    

    # first stage: both are in lag phase

    lam_min = problem.lam.min()

    # pick the minimum of the lag time and the final time
    # if the final time is within this first stage, do only return timepoints until t_final

    if lam_min > t_final:
        t1 = t_final
        logger.warning('The final time is less than the minimum lag time.')
    else:
        t1 = lam_min
        

    k1 = 2   # the number of points on the time grid in this first stage
    t = np.linspace(0, t1, k1)
    y0 = problem.initial_state()
    y = np.outer(y0, np.ones(k1))

    sol1 = Bunch(t=t, y = y, **kwargs)
    
    if t_final  <= t1:
        logger.warning('The final time falls within the first stage, all strains are in lag phase. Return solution.')
        sol1.update(params = problem.params())
        return sol1

    # second stage: one strain in lag phase, the other already growing
    lam_max = problem.lam.max()
    t2 = np.minimum(lam_max, t_final)
    # define the timespan for the solver
    assert sol1.t[-1] == t1
    t_span2 = [t1, t2]
    sol2 = solve_ivp( fun = problem.derivs, t_span = t_span2, y0 = sol1.y[:,-1], atol = atol, rtol = rtol, **kwargs)

    # glue the solutions

    t12 = np.hstack((sol1.t[:-1],sol2.t))
    y12 = np.hstack((sol1.y[:,:-1],sol2.y))
    sol12 = Bunch(sol2)
    sol12.update(t = t12)
    sol12.update(y= y12)

    if t2 >= t_final:
        logger.warning('The final time falls within the second stage, only one strain has been growing. Return solution.')
        sol12.update(params = problem.params())
        return sol12

    
    # third stage: both growing
    assert sol12.t[-1] == t2
    
    
    if adaptive_timewindow == False:  # first case: timewindow fixed
        t_span3 = [t2,t_final]


        sol3 = solve_ivp( fun = problem.derivs, t_span = t_span3, y0 = sol12.y[:,-1], atol = atol, rtol = rtol, **kwargs)
        logger.debug('Succesfully called solve_ivp to compute joint growth.')


        # glue the solutions

        t123 = np.hstack((sol12.t[:-1],sol3.t))
        y123 = np.hstack((sol12.y[:,:-1],sol3.y))
        sol123 = Bunch(sol3)
        sol123.update(t = t123)
        sol123.update(y= y123)
    else:                           # second case: timewindow adaptive
        sol_current = sol12
        t_final_current = sol_current.t[-1]
        
        Y = problem.params()['Y']   # extract yield values
        
        ## fix the parameters to check whether solution has reached stationary selection coefficient
        atol_stationary = scoeff_atol
        
        
        while check_remaining_resources_are_small(sol_current, Y = Y,\
                                                  atol = atol_stationary) == False \
        and t_final_current < t_final :
            
            t_span = [t_final_current,t_final_current + timestep]
            
            sol_extension = solve_ivp( fun = problem.derivs, t_span = t_span, y0 = sol_current.y[:,-1],\
                                       atol = atol, rtol = rtol,  **kwargs)
            logger.debug(f'Succesfully extended solution by an additional {timestep:.1f} hours.')

            # glue extension to current solution
            
            t_current = np.hstack((sol_current.t[:-1],sol_extension.t))
            y_current = np.hstack((sol_current.y[:,:-1],sol_extension.y))
            sol_current = Bunch(sol_extension)          
            sol_current.update(t = t_current)
            sol_current.update(y= y_current)
            
            
            t_final_current = sol_current.t[-1]         # we update to the new final time of simulation
            
        sol123 = sol_current
           
        if check_remaining_resources_are_small(sol123, Y = Y,   # extract yield values,\
                                               atol = atol_stationary) == False:
            logger.warning("Solution stops in non-stationary state. ")
            
            
        
        
    
    ## pass the parameters
    
    sol123.update(params = problem.params())


    return sol123

### I define the heaviside function, used later

def heaviside(x):
    """ A Heaviside function."""

    # we choose the right-continuous version
    # >>> heaviside(0)
    # 0.0
    return np.heaviside(x,1.)

class Problem_M3(object):
    """ A class for the CRM model in the linear regime $R << K$.

    """
    
    def __compute_auxiliary_variables__(self):
        
        #compute Y_bar
        self.Y_bar = 1./( (1.-self.x)/self.Y[0] + (self.x/self.Y[1]) )
        #compute g_bar
        self.g_bar = (  (self.g[0] * (1.-self.x)/self.Y[0])  +  (self.g[1] * self.x/self.Y[1])  ) * self.Y_bar
        
        # compute total resources in the system
        self.R_tot = self.R_0 + self.N_0/self.Y_bar
        # compute total fold change
        self.eta = self.R_tot*self.Y_bar/ self.N_0
        
    
    
    def __init__(self, g = [0.8,1.1], lam = [0.,0.], Y = [1.,1.5], x = 0.4, N_0 = 0.01, R_0 = 1., K = None):
        
        if type(K) != type(None): 
            print("Got problem with parameters 'K'. These will be ignored in the M3 model.")
            
        
        self.n_species = 2
        
        assert R_0 >= 0., "Parameter R_0 for initial resource concentration must be greater or equal 0. "
        
        # we need to turn lists into numpy arrays for later use with numpy functions
        self.g, self.lam, self.Y = np.array(g), np.array(lam), np.array(Y)
        self.x, self.N_0 =  x, N_0
        
        self.R_0 = R_0
        
        self.__compute_auxiliary_variables__()
        
        ## 
        
  
    def params(self):
        """ Returns parameters as dictionary. """
        return dict(g = self.g,Y = self.Y, x = self.x, eta = self.eta)
    
    def copy(self):
        return Problem_M3(**self.params())

    def derivs(self, t,y):
        """ Returns the derivatives at time t and state y.
        
        This will be passed to scipy.integrate.solve_ivp in the ODE solver.

        INPUTS
        ------
        t       scalar, time point
        y       numpy array of shape (3,1), state of the system


        RETURNS
        -------
        derivs  numpy array of shape (3,1), velocities

        """

        # extract the population sizes from the state vector
        N, R_available = y[:-1], y[-1]

        N_prime = np.zeros_like(N)

        N_prime[0] = heaviside(t - self.lam[0]) *  N[0]  *  self.g[0]  *  heaviside(R_available)
        N_prime[1] = heaviside(t - self.lam[1]) *  N[1]  *  self.g[1]  *  heaviside(R_available)
        
        R_prime =  - N_prime[0]/self.Y[0] - N_prime[1]/self.Y[1]
        R_prime = np.array([R_prime])

        derivs = np.concatenate((N_prime, R_prime))

        return derivs

    def initial_state(self):
        """ Returns the initial state as a numpy array. """
         # calculate initial population sizes
        N1_0 = (1.-self.x) * self.N_0
        N2_0 = self.x * self.N_0

        # calculate initial amount of available resources
        R_0 = self.R_0

        y0 = [ N1_0, N2_0, R_0]

        return np.array(y0)

def sol_exact_M3(t, problem):
    ## exact solution for species 1 growing on its own
    lam = problem.lam[0]
    g = problem.g[0]
    Y = problem.Y[0]
    N_0 = problem.N_0
    eta = problem.eta
    
    ## todo: vectorize this expression in t
    tsat = 1/g * np.log(eta) + lam
    assert tsat >= 0, "Parameters invalid, lead to negative saturation time."
    
    if t < lam:
        return N_0
    elif t > tsat:
        return eta*N_0
    else:
        return N_0*np.exp((t-lam)*g)



                      

def plot_solution(sol,ax):
    """ Plots solution as returned by solver to given ax. """
    
    ax.set_xlabel('time')

    ax2 = ax.twinx()
    ax2.plot(sol.t,sol.y[2], label = 'available resources', marker = 'x', color = 'grey')
    ax2.set_ylabel('resources', color = 'grey')
    ax2.tick_params(labelcolor= 'grey')
    
    ax.set_ylabel('log population sizes')
    
    ax.plot(sol.t, np.log(sol.y[0]), label = 'wildtype', marker = 'o', color = 'tab:green')
    ax.plot(sol.t, np.log(sol.y[1]), label = 'mutant', marker = 'o', color = 'tab:orange')
    
    ax.legend(loc ='center left')
    
    return ax,ax2
    

class Problem_Monod(Problem_M3):
    """ A class for the Monod model. Interpreting g as the nominal growth rate g_max.

    """
    
    def __init__(self, lam = [0.,0.], g= [0.8,1.1],  Y = [1.,1.5], K = [1., 0.8], x = 0.4, N_0 = 0.01, R_0 = 1.):
        
        assert R_0 >= 0., "Parameter R_0 must be greater 0."
        ## fix initial population size  
        
        
        self.n_species = 2
        
        
        # we need to turn lists into numpy arrays for later use with numpy functions
        self.g, self.lam, self.Y, self.K  = np.array(g), np.array(lam), np.array(Y), np.array(K)
        self.x, self.N_0, self.R_0 =  x, N_0, R_0
        
        # compute Y_bar, R_0, R_tot, g_bar as for the M3 model
        self.__compute_auxiliary_variables__()
        # compute a_bar
        self.a = np.divide(self.g,self.K)
        self.a_bar = (((1-self.x)*self.a[0]/self.Y[0]) + (self.x*self.a[1]/self.Y[1]))* self.Y_bar
        
            
        # compute resource scale
        self.Z = self.K[0]*self.K[1]*self.a_bar/self.g_bar 

      
    def params(self):
        """ Returns parameters as dictionary. """
        return dict(lam = self.lam, g = self.g, Y = self.Y, K = self.K, x = self.x, N_0 = self.N_0, R_0 = self.R_0)
    
    def copy(self):
        return Problem_Monod(**self.params())

    
    def derivs(self, t,y):
        """ Returns the derivatives at time t and state y.
        
        This will be passed to scipy.integrate.solve_ivp in the ODE solver.

        INPUTS
        ------
        t       scalar, time point
        y       numpy array of shape (3,1), state of the system


        RETURNS
        -------
        derivs  numpy array of shape (3,1), velocities

        """
 
        
        # extract the population sizes from the state vector
        N, R_available = y[:-1], y[-1]

        N_prime = np.zeros_like(N)

        N_prime[0] = heaviside(t - self.lam[0]) * N[0]  *\
                     self.g[0]  *  (R_available/(R_available + self.K[0])) * heaviside(R_available)
        N_prime[1] = heaviside(t - self.lam[1]) *  N[1]  *\
                     self.g[1]  *  (R_available/(R_available + self.K[1])) * heaviside(R_available)
        
        R_prime =  - N_prime[0]/self.Y[0] - N_prime[1]/self.Y[1]
        R_prime = np.array([R_prime])

        derivs = np.concatenate((N_prime, R_prime))

        return derivs
def estimate_s21(sol):
    y_end = sol['y'][:,-1]
    y_start = sol['y'][:,0]
    
    N1_0, N2_0,R_0 = y_start
    N1_f, N2_f,R_f = y_end
    
    s = np.log(N2_f/N1_f) - np.log(N2_0/N1_0)
    return s
    
def estimate_s21_timecourse(sol):
    N1_0, N2_0 = sol.y[0,0], sol.y[1,0]
    
    N1_t, N2_t = sol.y[0,:], sol.y[1,:]
    
    s_t = np.log(np.divide(N2_t,N1_t)) - np.log(N2_0/N1_0)
    
    return sol.t, s_t

def eval_smaxgrowth_fixed_dilution(x, g, Y, D):
    
    g_1,g_2 = g
    Y_1,Y_2 = Y
    
    Y_bar = 1/((1-x)/Y_1 + x/Y_2)
    g_bar = (  ((1-x)/(Y_1/Y_bar)) * g_1 )\
           +(  (    x/(Y_2/Y_bar)) * g_2 ) 
    
    
    s_maxgrowth = (g_2-g_1)/g_bar * np.log(D)
    
    return s_maxgrowth
def eval_smaxgrowth_fixed_bottleneck(x, g, Y, R_zero, N_0):
    
    Y_1,Y_2 = Y
    Y_bar = 1/((1-x)/Y_1 + x/Y_2)
    # compute dilution factor
    D =  1 + (R_zero*Y_bar/N_0)
    
    
    s_maxgrowth = eval_smaxgrowth_fixed_dilution(x=x,g=g,Y=Y, D=D)
    
    return s_maxgrowth

def eval_sthreshold_fixed_dilution(x, K, g, Y, R_zero, D ):
    
    K1,K2 = K
    g1,g2 = g
    Y1,Y2 = Y
    Y_bar = 1/(((1-x)/Y1) + (x/Y2))
    g_bar = (1-x)*g1/(Y1/Y_bar) + x*g2/(Y2/Y_bar) 
    
    if (K1 ==0.) & (K2==0.): # catch this case, because this leads to division by zero
        s_threshold = 0
    else:
    

        Z = K1 * K2 * ( ( (1-x)/(Y1/Y_bar)*(g1/g_bar)*(1/K1))\
                       +(   x  /(Y2/Y_bar)*(g2/g_bar)*(1/K2))   )



        first_frac = (K1 - K2)/( R_zero*(D/(D-1)) + Z) 
        second_frac = (g1/g_bar)*(g2/g_bar)

        log_frac = (R_zero + Z )/(Z)
        log_term = np.log( D * log_frac)

        s_threshold = first_frac * second_frac*log_term




    return s_threshold

def eval_sthreshold_fixed_bottleneck(x, K, g, Y, R_zero, N_0):
    
    Y_1,Y_2 = Y
    Y_bar = 1/((1-x)/Y_1 + x/Y_2)
    D = 1 + (R_zero*Y_bar/N_0)
    

    s = eval_sthreshold_fixed_dilution(x =x,K=K, g=g, Y=Y, R_zero = R_zero, D = D)
    return s
def eval_s21_formula_MCA_M3(problem):
    # MCA = mean consumer ansatz = established formula in case of M3 model
    g1,g2 = problem.g  # use nominal growth rates
    eta = problem.eta
    
    lam1, lam2 = problem.lam
    delta_lam = lam2 - lam1
    
    
    # use precomputed value of g_bar
    g_bar = problem.g_bar
    
    delta_g = g2 - g1
    s_growth = delta_g/g_bar * np.log(eta)
    
    s_lag = -delta_lam*g1*g2/g_bar
    
    return s_lag + s_growth
def eval_s21_formula_Monod(problem):
    # read model parameters
    params = problem.params()
    x = params['x']
    g = params['g']
    Y = params['Y']
    K = params['K']
    

    R_zero = problem.R_0
    N_0 = problem.N_0
    
    s_maxgrowth = eval_smaxgrowth_fixed_bottleneck(x=x,g=g,Y=Y, R_zero=R_zero, N_0=N_0)
    
    s_threshold = eval_sthreshold_fixed_bottleneck(x=x, K=K, g=g,Y=Y,R_zero = R_zero, N_0 = N_0 )
   
    return s_maxgrowth, s_threshold, s_maxgrowth+s_threshold

def eval_smax_alternative_fixed_dilution(x,K, g, Y, R_zero, D):
    
    K1,K2 = K
    g1,g2 = g
    Y1,Y2 = Y
    
    ### compute effective traits
    Y_bar = 1/((1-x)/Y1 + x/Y2)
    g_bar = (  ((1-x)/(Y1/Y_bar)) * g1 )\
           +(  (    x/(Y2/Y_bar)) * g2 ) 
    
    ### compute effective scale Z
    if (K1 ==0.) & (K2==0.): # catch this case, because this leads to division by zero
        Z=0
    else:
        Z = K1 * K2 * ( ( (1-x)/(Y1/Y_bar)*(g1/g_bar)*(1/K1))\
                   +(   x  /(Y2/Y_bar)*(g2/g_bar)*(1/K2))   )

    ### compute total resources
    R_tot = R_zero * (1 + (1/D))
    
    s_max  = (g2 - g1)/g_bar *  (R_tot/(R_tot + Z)) *  np.log(D)
    s_max -= (g2 - g1)/g_bar *  (Z/(R_tot + Z))     *  np.log(1 + (R_zero/Z))
    
    
  
    return s_max
def eval_smax_alternative_fixed_bottleneck(x,K, g, Y, R_zero, N_0):

    Y_1,Y_2 = Y
    Y_bar = 1/((1-x)/Y_1 + x/Y_2)
    
    # compute dilution factor
    D =  1 + (R_zero*Y_bar/N_0)
    
    
    s_max = eval_smax_alternative_fixed_dilution(x=x,K=K, g=g,Y=Y, R_zero=R_zero,D=D)
    
    return s_max

def eval_slin_alternative_fixed_dilution(x, K, g, Y, R_zero, D ):
    
    K1,K2 = K
    g1,g2 = g
    Y1,Y2 = Y
    Y_bar = 1/(((1-x)/Y1) + (x/Y2))
    g_bar = (1-x)*g1/(Y1/Y_bar) + x*g2/(Y2/Y_bar) 
    

    
    
    
    
    if (K1 ==0.) & (K2==0.): # catch this case, because this leads to division by zero
        s_lin = 0
    else:
    

        Z = K1 * K2 * ( ( (1-x)/(Y1/Y_bar)*(g1/g_bar)*(1/K1))\
                       +(   x  /(Y2/Y_bar)*(g2/g_bar)*(1/K2))   )
        
        a1,a2 = g1/K1,g2/K2
        
        a_bar = ( ( (1-x)/(Y1/Y_bar)*a1)\
                 +(   x  /(Y2/Y_bar)*a2)   ) 
        
        ### compute total resources
        R_tot = R_zero * (1 + (1/D))
        
        ### compute total resources (shorthand only)
        R_tot = R_zero * (1 + (1/D))

        s_lin   = (a2 - a1)/a_bar *  (Z/(R_tot + Z))     *  np.log(D)
        s_lin  += (a2 - a1)/a_bar *  (Z/(R_tot + Z))     *  np.log(1 + (R_zero/Z))
    


    return s_lin

def eval_slin_alternative_fixed_bottleneck(x, K, g, Y, R_zero, N_0):
    
    Y_1,Y_2 = Y
    Y_bar = 1/((1-x)/Y_1 + x/Y_2)
    D = 1 + (R_zero*Y_bar/N_0)
    

    s = eval_slin_alternative_fixed_dilution(x =x,K=K, g=g, Y=Y, R_zero = R_zero, D = D)
    return s
def eval_s21_formula_Monod_alternative(problem):
    # read model parameters
    params = problem.params()
    x = params['x']
    g = params['g']
    Y = params['Y']
    K = params['K']

    ## initial conditions
    N_0 = problem.N_0
    R_0 = problem.R_0
    

    s_max = eval_smax_alternative_fixed_bottleneck(x=x,g=g, K=K, Y=Y,R_zero= R_0, N_0 = N_0)
    s_lin = eval_slin_alternative_fixed_bottleneck(x=x,g=g, K=K, Y=Y,R_zero= R_0, N_0 = N_0)
    
    s = s_max + s_lin
    
    return s_max, s_lin, s

#####

if __name__ == '__main__': 
    pass
    
