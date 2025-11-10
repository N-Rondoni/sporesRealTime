import seaborn as sns 
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds
import pandas as pd
import time

# Function to create ODE system with parameters 
def make_sys(params):
    def sys(t, A):
        y, z = A
        a, b, c, d, e, f, g, h, u = params
        dA = [a*u + b*y*z + c*y + d*z,
              e*u + f*y*z + g*y + h*z]
        return dA
    return sys

# Cost function: squared error between model and data
def cost(params, t_data, z_data, A0):
    sys = make_sys(params)
    try:
        sol = solve_ivp(sys, [t_data[0], t_data[-1]], A0, t_eval=t_data)
        if not sol.success:
            return np.inf
        y_model, z_model = sol.y
        return np.sum(((z_model - z_data)**2))
    except Exception:
        return np.inf


def computeRates(csv):
    ##
    # Alex's plotting routine (commented out below), computes rates also. 
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

    #for germinant in germinant_exposure_frames:
    #    plt.axvline(germinant, alpha = 0.5, color = "lightgrey")
    #    exp = csv.replace(data_dir, "").replace("_Data.csv","")
    #sns.lineplot(x = "Frame", y = f"Normalized_Rate", data = df_germination_rates, linewidth = 4)

    #title = exp + f" {germinant_concentration.max()}mM"
    #plt.title(title)
    #plt.savefig(data_save_dir + f"Germination_Rate_{exp}.jpg")
    #plt.show()
    #df_germination_rates.to_csv(data_save_dir + f"Germination_Rate_{exp}.csv", index = False)

if __name__=="__main__":
    
    # define ICs
    Y = 3
    Z = 0
    A0 = [Y, Z]

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
    # try normalizing
    zDatMean = np.mean(zDat)
    print(zDatMean)
    #zDatSTD = np.std(zDat)
    #zDat = (zDat - zDatMean)/zDatSTD
    
    timeDat = rawDat[:, 0]

    # set up time
    tStart = timeDat[0]
    tEnd = timeDat[-1]
    tSolve = [tStart, tEnd]
    # define a timevec for when params are found, plot an extra smooth soln
    n = 1000
    timeVec = np.linspace(tSolve[0], tSolve[1], n)
    

    # Initial guess for params
    #        [a  ,  b,   c,   d,   e, f, g, h,  u]
    #guess = [0.8, 10, 0.2, 90, 80, 0.9]  
    guess = [5, 20, 0.1, 100, 100, 1, 3, 3, 2]  # Ensure positive value
    guess = [1, -2, -0.1, -2, -1, -1, -3, -3, -2]  # Ensure positive value

    # Run optimization
    start = time.time()
    print("Beginning to optimize paramters...")
   
    methodChoice = 'Nelder-Mead'
    result = minimize(cost, guess, args=(timeDat, zDat, A0), method=methodChoice)
    best_params = result.x
    
    end = time.time()
    elapsedMinutes = (end - start)/60
    print("Best parameters found using ", methodChoice, " after ", elapsedMinutes, "minutes")


    # Solve using fitted parameters
    sys_fit = make_sys(best_params)
    sol_fit = solve_ivp(sys_fit, tSolve, A0, t_eval=timeVec)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

    #axs[0].plot(timeDat, y_data, 'o', markersize=2, label='Noisy Y Data', alpha=0.5)
    axs[0].plot(timeVec, sol_fit.y[0], '-', label='Fitted Y Model', linewidth=2)
    axs[0].set_ylabel('Y Dynamics')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title('Y vs Time')

    axs[1].plot(timeDat, zDat, 'o', markersize=2, label='Noisy Z Data', alpha=0.5)
    axs[1].plot(timeVec, sol_fit.y[1], '-', label='Fitted Z Model', linewidth=2)
    axs[1].set_ylabel('Z Dynamics')
    axs[1].set_xlabel('Time')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title('Z vs Time')

    plt.suptitle('Parameter Fit to ODE System')
    plt.show()

    print("Fitted parameters:", best_params)

