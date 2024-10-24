import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

import gurobipy as gp
from gurobipy import GRB, tuplelist

import pandapower.networks as pn
import scipy.io

from data_ieee69 import get_ieee_data, build_case69

## CHOOSE ONE OF THE BUS SYSTEMS BELOW:
# case = 'bus33'
case = 'bus69'
if case == 'bus33':
    net = pn.case33bw()
elif case == 'bus69':
    ie69 = get_ieee_data()
    net = build_case69(ie69)

T = 1
kVAbase = 1000 # kVA
MVAbase = kVAbase / 1000 # MVA
impbase = net.bus.vn_kv[0]**2/(MVAbase)

frombus,tobus = [],[]
for i in net.line.index:
    frombus.append(net.bus.loc[net.line.from_bus[i],'name'])
    tobus.append(net.bus.loc[net.line.to_bus[i],'name'])

[frombus,tobus]
net.line.from_bus = frombus
net.line.to_bus = tobus

net.bus.index = net.bus.name
net.line.index = [frombus,tobus]
net.line = net.line.loc[net.line.in_service == True]
net.line['rateA'] = 10000

r = net.line.length_km * net.line.r_ohm_per_km
x = net.line.length_km * net.line.x_ohm_per_km
net.line = net.line.assign(r=r)
net.line = net.line.assign(x=x)

net.line = net.line.assign(r_pu=r/impbase)
net.line = net.line.assign(x_pu=x/impbase)
net.line = net.line[['from_bus','to_bus','r','x','r_pu','x_pu','rateA']]
net.bus = net.bus[['name','vn_kv']]

ldp,ldq = [],[]
for i in net.load.index:
    ldp.append(net.bus.loc[net.load.bus[i],'name'])
    ldq.append(net.bus.loc[net.load.bus[i],'name'])

net.bus['Pd'] = 0
net.bus['Qd'] = 0
for i in net.load.index:
    loca = net.bus.loc[net.load.bus[i],'name']
    net.bus.Pd[loca] = net.load.p_mw[i]*1000/kVAbase
    net.bus.Qd[loca] = net.load.q_mvar[i]*1000/kVAbase

net.ext_grid['Pmin'] = net.ext_grid.min_p_mw*MVAbase
net.ext_grid['Pmax'] = net.ext_grid.max_p_mw*MVAbase
net.ext_grid['Qmin'] = net.ext_grid.min_q_mvar*MVAbase
net.ext_grid['Qmax'] = net.ext_grid.max_q_mvar*MVAbase

net.ext_grid.index = net.ext_grid.bus
net.ext_grid = net.ext_grid[['bus','Pmin','Pmax','Qmin','Qmax']]

net.ext_grid.index = net.ext_grid.bus

gen = net.ext_grid
line = net.line
bus = net.bus

slackbus = [0]
set_t = list(range(T)) # Please dont make in in range(x) format. It should be list or np.arrays

nGen = gen.shape[0]
set_gen = gen.index
gen_t = tuplelist([(i,t) for t in set_t for i in set_gen])

nBus = bus.shape[0]
set_bus = bus.index
bus_t = tuplelist([(i,t) for t in set_t for i in set_bus])

nLine = line.shape[0]
set_line = line.index
line_t = tuplelist([(i,j,t) for t in set_t for i,j in set_line])

# Create index for aux vars
line_t_dir = tuplelist() # index for power flow direction
for t in set_t:
    for i,j in set_line:
        line_t_dir = line_t_dir + tuplelist([(i,j,t)]) + tuplelist([(j,i,t)])

def resi(line_df, m, n, impbase=impbase):
    # Using a tuple to check for membership is faster than using 'not in'
    idx = (m, n) if (m, n) in line_df.index else (n, m)
    # Accessing the 'r' column directly and then using .at for scalar value access
    return line_df.at[idx, 'r'] / impbase

def reac(line_df, m, n, impbase=impbase):
    # Using a tuple to check for membership is faster than using 'not in'
    idx = (m, n) if (m, n) in line_df.index else (n, m)
    # Accessing the 'x' column directly and then using .at for scalar value access
    return line_df.at[idx, 'x'] / impbase


## Power Flow Variables
m = gp.Model(case + '_convex')
vmin, vmax = 0.9, 1.1
u_i = m.addVars(bus_t, vtype=GRB.CONTINUOUS, lb=vmin**2, ub=vmax**2, name='V_squared')

p_g = m.addVars(gen_t, vtype=GRB.CONTINUOUS, lb=0, ub=gen.Pmax.tolist()*T, name='P_Gen')
q_g = m.addVars(gen_t, vtype=GRB.CONTINUOUS, lb=gen.Qmin.tolist()*T, ub=gen.Qmax.tolist()*T, name='Q_Gen')

p_inj = m.addVars(bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='P_Inj')
q_inj = m.addVars(bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Inj')

br_lim = 5
p_ij = m.addVars(line_t, vtype=GRB.CONTINUOUS, lb=-br_lim, ub=br_lim, name='P_Line')
q_ij = m.addVars(line_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Line')

l_ij = m.addVars(line_t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='L_Line')


## Identify bus neighbors
ne = {}

for x in set_bus:
    ne[x] = []
    for y in set_line:
        if x == y[0]:
            ne[x].append(y[1])
        elif x == y[1]:
            ne[x].append(y[0])

# Convert adjacency dictionary to DataFrame
neighbors = pd.DataFrame({'Neighbors': [ne[node] for node in ne]}, index=list(ne.keys()))

## Constraints
P_balance = m.addConstrs(((p_ij.sum(i,'*',t) + gp.quicksum(resi(line,i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - p_ij.sum('*',i,t) == p_inj[i,t] for i,t in bus_t), name='P-Balance')
Q_balance = m.addConstrs(((q_ij.sum(i,'*',t) + gp.quicksum(reac(line,i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - q_ij.sum('*',i,t) == q_inj[i,t] for i,t in bus_t), name='Q-Balance')
V_drop = m.addConstrs((u_i[i,t] - u_i[j,t] == 2*(resi(line,i,j) * p_ij[i,j,t] + reac(line,i,j) * q_ij[i,j,t]) - (resi(line,i,j)**2 + reac(line,i,j)**2) * l_ij[i,j,t] for i,j,t in line_t),name='V-Drop')
V_slack = m.addConstrs((u_i[i,t] == 1 for i in [0] for t in set_t), name='V-Slack')
P_injection = m.addConstrs((p_inj[i,t] == p_g.sum(i,t) - bus.Pd[i] for i,t in bus_t), name='P-Injection')
Q_injection = m.addConstrs((q_inj[i,t] == q_g.sum(i,t) - bus.Qd[i] for i,t in bus_t), name='Q-Injection')
Ohm_law = m.addConstrs((l_ij[i,j,t] * u_i[j,t] >= p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in line_t), name='Ohms-Law')

m.setObjective(gp.quicksum(l_ij[i,j,t] for i,j,t in line_t),GRB.MINIMIZE)
m.update()
m.write(case + '_convex.lp')
m.optimize()

# Print voltage
for i in u_i:
    print(np.sqrt(u_i[i].X))

