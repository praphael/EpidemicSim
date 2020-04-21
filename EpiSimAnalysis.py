# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 23:29:50 2020

@author: praph
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from matplotlib.ticker import ScalarFormatter
import copy

#fName1 = 'stats1_2020_04_21_00_19_49.csv'
#fName2 = 'stats2_2020_04_21_00_19_49.csv'
#fName1 = 'stats1_2020_04_21_01_32_56.csv'
#fName2 = 'stats2_2020_04_21_01_32_56.csv'
fName1 = 'stats1_2020_04_21_13_04_11.csv'
fName2 = 'stats2_2020_04_21_13_04_11.csv'
fName3 = 'stats3_2020_04_21_13_04_11.csv'
fName4 = 'stats4_2020_04_21_13_04_11.csv'


class sim_stats:
    pass

def readStatsLineArray(fp, dt=np.uint32):
    ln = fp.readline()
    s = re.split(':,', ln)
    s = s[1]
    arry = np.fromstring(s, dtype=dt, sep=',')
    return arry
    
def loadStats(fName):
    all_stats = []
    fp = open(fName, 'r')
    
    while(True):
        try:
            stats = sim_stats();
            stats.tday = readStatsLineArray(fp, dt=np.float32)
            stats.inf = readStatsLineArray(fp)
            stats.cntgs = readStatsLineArray(fp)
            stats.sick = readStatsLineArray(fp)
            stats.hosp = readStatsLineArray(fp)
            stats.rec = readStatsLineArray(fp)
            stats.dec = readStatsLineArray(fp)
            all_stats.append(stats)
        except Exception as e:
            break
    
    fp.close()
    
    return all_stats

def avgStats(statsList):
    statsAvg = None
    for stats in statsList:
        if(statsAvg == None):
            statsAvg = copy.deepcopy(stats)
        else:
            statsAvg.inf += stats.inf
            statsAvg.cntgs += stats.cntgs
            statsAvg.sick += stats.sick
            statsAvg.hosp += stats.hosp
            statsAvg.rec += stats.rec
            statsAvg.dec += stats.dec        
            
    numStats = len(statsList)
    statsAvg.inf = statsAvg.inf / numStats
    statsAvg.cntgs = statsAvg.cntgs / numStats
    statsAvg.sick = statsAvg.sick / numStats
    statsAvg.hosp = statsAvg.hosp / numStats
    statsAvg.rec = statsAvg.rec / numStats
    statsAvg.dec = statsAvg.dec / numStats
    
    return statsAvg

sList1 = loadStats(fName1)
s1 = sList1[0]
#s1 = avgStats(sList1)
sList2 = loadStats(fName2)
#s2 = avgStats(sList2)
s2 = sList2[0]

sList3 = loadStats(fName3)
s3 = sList3[0]

#s1 = avgStats(sList1)
sList4 = loadStats(fName4)
#s2 = avgStats(sList2)
s4 = sList4[0]
    
plt.figure(1)
plt.plot(s1.tday, s1.inf, '-b')
plt.plot(s2.tday, s2.inf, ':b')
plt.plot(s3.tday, s3.inf, '--b')
plt.plot(s4.tday, s4.inf, '-.b')
plt.yscale('log')
plt.title('Infections')
plt.xlabel('Day')    
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend(['Do Nothing', 'Shutdown Schools Only', 'Shutdown Both', 'Shutdown Both Earlier'])


plt.figure(2)
plt.plot(s1.tday, s1.hosp, '-b')
plt.plot(s2.tday, s2.hosp, ':b')
plt.plot(s3.tday, s3.hosp, '--b')
plt.plot(s4.tday, s4.hosp, '-.b')
plt.yscale('log')
plt.title('Hospitalizations')
plt.xlabel('Day')    
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend(['Do Nothing', 'Shutdown Schools Only', 'Shutdown Both', 'Shutdown Both Earlier'])

plt.figure(3)
plt.plot(s1.tday, s1.dec, '-r')
plt.plot(s2.tday, s2.dec, ':r')
plt.plot(s3.tday, s3.dec, '--r')
plt.plot(s4.tday, s4.dec, '-.r')
plt.yscale('log')
plt.title('Deaths')
plt.xlabel('Day')    
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.legend(['Do Nothing', 'Shutdown Schools Only', 'Shutdown Both', 'Shutdown Both Earlier'])




