import csv
import math
import numpy as np
from scipy import spatial
from scipy.stats import t
from scipy import linalg
import time
writer = csv.writer(file('/home/shiyc/Documents/sta/data/sample.csv', 'wb'))
frames = np.arange(250)+1
rou = np.arange(5,20,1)
theta = np.arange(-np.pi,np.pi,np.pi/15)
omiga = 3.14/25.0
writer.writerow([len(frames),'',''])
writer.writerow(['x', 'y', 'vx','vy','frame'])
for f in frames:
    for r in rou:
        for ind,th in enumerate(theta):
            x = r*np.cos(th)
            y = r*np.sin(th)
            vx = -r*omiga*np.sin(th)
            vy = r*omiga*np.cos(th)
            writer.writerow([x,y,vx,vy,f])
            theta[ind] = th-omiga

