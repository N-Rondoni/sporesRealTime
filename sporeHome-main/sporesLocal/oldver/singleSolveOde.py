import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def sys(t, A):
    # state vars
    y, z = A    
    # parameters
    a, b, c, d, e, u = p

    dA = [a*u - c*y,
          b*u - d*y*z - e*z]

    return dA



if __name__=="__main__":

    # define time 
    n = 1000
    tEnd = 180
    timeVec = np.linspace(0, tEnd, n)
    tSolve = [0, tEnd]

    # set ICs
    Y = 0
    Z = 0
    A0 = [Y, Z]

    # define parameters. Want: a,b,c,d > 0. b>a, e >> 1, c*b << d*a from SS analysis. 
    a = 2
    b = 20
    c = 0.2
    d = 100
    e = 100
    u = .75
    p = [a, b, c, d, e, u] # pack to be read out in sys. 

    # generate solution
    sol = solve_ivp(sys, tSolve, A0, t_eval = timeVec)
    
    #plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, constrained_layout=True)

    # first plot
    axs[0].plot(timeVec, sol.y[0], label='Signal 1', color='tab:blue', linewidth=2)
    axs[0].set_ylabel('Y Dynamics')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title('Y vs Frame')

    # second plot
    axs[1].plot(timeVec, sol.y[1], label='Signal 2', color='tab:orange', linewidth=2)
    axs[1].set_xlabel('Frames')
    axs[1].set_ylabel('Z Dynamics')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    axs[1].set_title('Z vs Frame')

    plt.show()
