import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd

def sys(t, A):
    # state vars
    y, z = A    
    # parameters
    a, b, c, d, e, u = p

    dA = [a*u - c*y,
          b*u - d*y*z - e*z]

    return dA

def computeRates(csv):
    ##
    # Alex's plotting routine, computes rates also. 
    # takes: CSV name as a string. 
    ##
    rates_data = []

    df = pd.read_csv(data_dir + csv)
    df = df[["Frame", "Track_PhC", "Status_PhC", "Germinant", "Germination_Index_PhC"]]
    max_frames = max(df["Frame"].values)
    germinant_concentration = df.loc[~df["Germinant"].isin([0]), "Germinant"].unique()
    germinant_exposure_frames = df.loc[~df["Germinant"].isin([0]), "Frame"].values
    total_spores = df["Track_PhC"].nunique()

    for frame_i in range(max_frames + 1 - window_size):

        if frame_i in germinant_exposure_frames:
          germinant_value = germinant_concentration.max()
        else:
          germinant_value = 0

        frame_window = list(range(frame_i, frame_i + window_size))
        spores_germinated = df[df["Germination_Index_PhC"].isin(frame_window)]
        num_germinated = spores_germinated["Track_PhC"].nunique()
        norm_num_germinated = num_germinated / total_spores
        rates_data.append([frame_i, norm_num_germinated, num_germinated, germinant_value])

    df_germination_rates = pd.DataFrame(rates_data, columns = ["Frame", "Normalized_Rate", "Rate", "Germinant_Concentration"])

    return df_germination_rates



if __name__=="__main__":

    # set ICs
    Y = 0
    Z = 0
    A0 = [Y, Z]

    # define parameters. Want: a,b,c,d > 0. b>a, e >> 1, c*b << d*a from SS analysis. 
    a = 2.0517
    b = 60.9509
    c = .02
    d = 100.01
    e = 100.0035
    u = .7567
    #u = 0.3
    p = [a, b, c, d, e, u] # pack to be read out in sys. 
    
    # load in raw dat
    window_size = 3 # unit: frames. Each frame is 5 minutes. 
    data_dir = "/home/nicho/workspace/sporesLocal/data/"
    fpath1 = "M6813_s4_Data.csv"
    fpath2 = "M6813_s1_Data.csv"

    df_germination_rates = computeRates(fpath1)
    rawDat = df_germination_rates[["Frame", "Normalized_Rate"]].to_numpy()
    # rawDat has dim (140, 2), organized by (frame, normalized_rate) 
    # recall we don't really care about the dynamics in Y. Just need Z to behave. 
    zDat = rawDat[:, 1]
    timeDat = rawDat[:, 0]

    # set up time
    tStart = timeDat[0]
    tEnd = timeDat[-1]
    tSolve = [tStart, tEnd]
    # define a timevec for plotting.
    n = 100
    timeVec = np.linspace(tSolve[0], tSolve[1], n)
   
    nonzeroInds = np.nonzero(zDat)
    firstNonzeroInd = nonzeroInds[0][0]


    print(zDat)

    # generate solution
    sol = solve_ivp(sys, tSolve, A0, t_eval = timeVec)
    
    # delay solution by how ever many zeros were present in data to account for shock at germination
    yDynam = sol.y[1]
    yDynam = np.insert(yDynam, 0, zDat[0:firstNonzeroInd]) 
    endInds = np.arange(len(yDynam) - firstNonzeroInd, len(yDynam))
    yDynam = np.delete(yDynam, endInds)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, constrained_layout=True)

    # first plot
    axs[0].plot(timeVec, sol.y[0], label='Signal 1', color='tab:blue', linewidth=2)
    axs[0].set_ylabel('Y Dynamics')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title('Y vs Frame')

    # second plot
    axs[1].plot(timeVec, yDynam, label='Signal 2', color='tab:orange', linewidth=2)
    axs[1].plot(timeDat, zDat, 'o', markersize=2, label='Raw Z Data', alpha=0.5) #raw dat
    axs[1].set_xlabel('Frames')
    axs[1].set_ylabel('Z Dynamics')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    axs[1].set_title('Z vs Frame')

    plt.show()
