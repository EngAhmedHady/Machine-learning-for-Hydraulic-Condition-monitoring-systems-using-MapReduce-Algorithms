import numpy as np
import pandas as pd
from pandas import DataFrame as df

dfs2= pd.read_csv("FS2.txt",delimiter="\t",header=None )
dt3 = pd.read_csv("TS3.txt",delimiter="\t",header=None )
dt4 = pd.read_csv("TS4.txt",delimiter="\t",header=None )
dcp= pd.read_csv("CP.txt",delimiter="\t",header=None )
dce = pd.read_csv("CE.txt",delimiter="\t",header=None )

fs2 = np.round(np.array(dfs2.values),2)
t3 = np.round(np.array(dt3.values),2)
t4 = np.round(np.array(dt4.values),2)
cp = np.round(np.array(dcp.values),2)
ce = np.round(np.array(dce.values),2)

t_amb = t3 - ((t3-t4)*100/ce)

fs2New = []
for i in range(0,len(fs2[1])-9,10):
    fs2New.append(np.sum(fs2[:,0:9], axis=1)/10)

oil= (cp*6*10**7)/((t3-t4)*np.transpose(fs2New))

t_amb = np.round(t_amb,2)
oil = np.round(oil,2)

a = []
for i in range(0,2205):
        for j in range (0,60):
            a.append (("T3",i,j,t3[i,j]))
            a.append(("T4",i,j,t4[i,j]))
            a.append(("Tamb",i,j,t_amb[i,j]))
            a.append(("FS2",i,j,fs2[i,j]))
            a.append(("Oil",i,j,oil[i,j]))

Df = pd.DataFrame(a) #qyuadrables

f1 = open('CE-CP.txt', "w+")
Df.to_csv('CE-CP.txt', header=None, index=None, sep=' ', mode='a')
f1.close()

