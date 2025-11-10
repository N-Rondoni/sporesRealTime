import numpy as np

def desiredTraj(x):
    # x is current frame number
    # returns linear fitting between (0, 0) and (finalFrame, setpoint), evaluated at x.
    setPoint = 0.30 # 30 percent normalized germ rate at final time
    finalFrame = 150
    return (setPoint/finalFrame)*x
