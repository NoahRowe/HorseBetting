import numpy as np
import pandas as pd

DIST_TRACK_AVGS = np.load('averages.npy',allow_pickle='TRUE').item()
CLASS_AVGS = np.load("class_averages.npy", allow_pickle='TRUE').item()
SEC1_CLASS_AVGS = np.load("sec1_class_averages.npy", allow_pickle='TRUE').item()
LAST_SEC_CLASS_AVGS = np.load("last_sec_class_averages.npy", allow_pickle='TRUE').item()
SECTIONAL_DISTANCES = np.load("sectional_distances.npy", allow_pickle="TRUE").item()
SEC1_DIST_TRACK_AVGS = np.load("sec1_averages.npy", allow_pickle="TRUE").item()
LAST_SEC_DIST_TRACK_AVGS = np.load("last_sec_averages.npy", allow_pickle="TRUE").item()

def speed2rating(time, distance, venue, surface):
    
    avgTime = DIST_TRACK_AVGS[venue + " " + str(surface) + " " + str(distance)]
    pctOfAvg = (avgTime - time) / float(avgTime) * 100
    
    rating = 80 + pctOfAvg * 10
    return rating

def speed2rating_sec1(time, distance, venue, surface):
    
    avgTime = SEC1_DIST_TRACK_AVGS[venue + " " + str(surface) + " " + str(distance)]
    pctOfAvg = (avgTime - time) / float(avgTime) * 100
    
    rating = 80 + pctOfAvg * 10
    return rating

def speed2rating_last_sec(time, distance, venue, surface):
    
    avgTime = LAST_SEC_DIST_TRACK_AVGS[venue + " " + str(surface) + " " + str(distance)]
    pctOfAvg = (avgTime - time) / float(avgTime) * 100
    
    rating = 80 + pctOfAvg * 10
    return rating

def printProgress(i, l, jump=1000):
    if i % jump == 0:
        print("{:.2f}% Complete ...".format(i /float(l) * 100))
        
        
        
        

        
        



# dic = {}
# dic["HV 0 1000"] = [200, 400, 400]
# dic["HV 0 1200"] = [400, 400, 400]
# dic["HV 0 1650"] = [450, 400, 400, 400]
# dic["HV 0 1800"] = [200, 400, 400, 400, 400]

# dic["ST 0 1000"] = [200, 400, 400]
# dic["ST 0 1200"] = [400, 400, 400]
# dic["ST 0 1400"] = [200, 400, 400, 400]
# dic["ST 0 1600"] = [400, 400, 400, 400]
# dic["ST 0 1800"] = [200, 400, 400, 400, 400]
# dic["ST 0 2000"] = [400, 400, 400, 400, 400]

# dic["ST 1 1200"] = [400, 400, 400]
# dic["ST 1 1650"] = [450, 400, 400, 400]
# dic["ST 1 1800"] = [200, 400, 400, 400, 400]
# np.save("Sectional_distances.npy", dic)