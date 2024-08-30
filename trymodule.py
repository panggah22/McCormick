from module import *

import gurobipy as gp
from gurobipy import GRB,tuplelist
import scipy.io

# DG input
dgloc = {'bus':[3,13,22], 'Pmin':[0,0,0], 'Pmax':[0.5,0.5,0.5], 'Qmin':[-0.3,-0.3,-0.3], 'Qmax':[0.3,0.3,0.3]}
gci = [0.55,0.7,0.8,0.75]

# PV input
pvloc = {'bus':[9, 15, 19], 'p_max':[0.3, 0.3, 0.3], 'q_min':[-0.3, -0.3, -0.3], 'q_max':[0.3, 0.3, 0.3]}

# ESS input
essloc = {'bus':[7,21,29],'Cap':[1,1,1],'Pmin':[0,0,0],'Pmax':[0.2,0.2,0.2],'Qmin':[-0.15,-0.15,-0.15],'Qmax':[0.15,0.15,0.15]}

# Data preprocessing
# ----------- INPUTS -----------
T = 1
mins = 30
vmin, vmax = 0.9,1.1
use_mdt = True
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
# data.include_dg(dgloc,gci)
# data.include_pv(pvloc)
# data.include_ess(essloc)

sets = define_sets(data)

## Power Flow Variables
m = gp.Model('Case33-Linear')

u_i = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=vmin**2, ub=vmax**2, name='V_squared')

p_g = m.addVars(sets.gen_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.gen.Pmax.tolist()*T, name='P_Gen')
q_g = m.addVars(sets.gen_t, vtype=GRB.CONTINUOUS, lb=data.gen.Qmin.tolist()*T, ub=data.gen.Qmax.tolist()*T, name='Q_Gen')

p_inj = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='P_Inj')
q_inj = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Inj')

br_lim = 5
p_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=-br_lim, ub=br_lim, name='P_Line')
q_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Line')

l_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='L_Line')

neighbors = define_neighbor(sets.bus,sets.line)
impbase = data.impbase

P_balance = m.addConstrs(((p_ij.sum(i,'*',t) + gp.quicksum(data.resi(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - p_ij.sum('*',i,t) == p_inj[i,t] for i,t in sets.bus_t), name='P-Balance')
Q_balance = m.addConstrs(((q_ij.sum(i,'*',t) + gp.quicksum(data.reac(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - q_ij.sum('*',i,t) == q_inj[i,t] for i,t in sets.bus_t), name='Q-Balance')
V_drop = m.addConstrs((u_i[i,t] - u_i[j,t] == 2*(data.resi(i,j) * p_ij[i,j,t] + data.reac(i,j) * q_ij[i,j,t]) - (data.resi(i,j)**2 + data.reac(i,j)**2) * l_ij[i,j,t] for i,j,t in sets.line_t),name='V-Drop')
V_slack = m.addConstrs((u_i[i,t] == 1 for i in [0] for t in sets.t), name='V-Slack')

P_injection = m.addConstrs((p_inj[i,t] == p_g.sum(i,t) - data.bus.Pd[i]*1 for i,t in sets.bus_t), name='P-Injection')
Q_injection = m.addConstrs((q_inj[i,t] == q_g.sum(i,t) - data.bus.Qd[i]*1 for i,t in sets.bus_t), name='Q-Injection')

# Ohm_law = m.addConstrs((l_ij[i,j,t] * u_i[j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')
if use_mdt == True:
    w = mdt(m,l_ij,u_i,sets.line_t,p=-2,P=2)
    Ohm_law = m.addConstrs((w[i,j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')
else:
    Ohm_law = m.addConstrs((l_ij[i,j,t] * u_i[j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')


## Direction replacement
# p_line_dir = m.addConstrs((p_ij[i,j,t] == p_hat[i,j,t] - p_hat[j,i,t] for i,j,t in line_t), name='Line-Direction')

# # v--- p_hat_ij * p_hat_ji = 0 ---v
# for t in set_t: # Somehow Special Ordered Set is sometimes faster than binary linearization
#     for i,j in set_line:
#         m.addSOS(GRB.SOS_TYPE1,[p_hat[i,j,t], p_hat[j,i,t]],[1,1])


# ---------------- OBJECTIVE FUNCTION ----------------
# ploss = m.addVars(sets.t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='ploss')
# m.addConstrs(ploss[t] == gp.quicksum(resi(data.line,i,j)*l_ij[i,j,t] for i,j in sets.line) for t in sets.t)
# m.setObjective(ploss.sum())

m.update()
m.write('test.lp')

m.optimize()
