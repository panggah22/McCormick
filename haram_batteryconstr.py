from module import *

import gurobipy as gp
from gurobipy import GRB,tuplelist
import scipy.io
import math

import pandas as pd

essloc = {'bus':[5,17],'Cap':[0.5,0.5],'Pmin':[0,0],'Pmax':[0.1,0.1],'Qmin':[-0.1,-0.1],'Qmax':[0.1,0.1]}

pvloc = {'bus':[5,17], 'p_max':[0.5,0.5], 'q_min':[-0.1,-0.1], 'q_max':[0.1,-0.1]}

T = 24
mins = 30
delta_t = mins/60
soc_init = 0
vmin, vmax = 0.95,1.1
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
data.include_ess(essloc)
data.include_pv(pvloc)

sets = define_sets(data)

# print(data.ess_t)
m = gp.Model('Case33-Linear')

u_i = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=vmin**2, ub=vmax**2, name='V_squared')

p_g = m.addVars(sets.gen_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.gen.Pmax.tolist()*T, name='P_Gen')
q_g = m.addVars(sets.gen_t, vtype=GRB.CONTINUOUS, lb=data.gen.Qmin.tolist()*T, ub=data.gen.Qmax.tolist()*T, name='Q_Gen')

p_inj = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='P_Inj')
q_inj = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Inj')

br_lim = 5
p_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=-br_lim, ub=br_lim, name='P_Line')
q_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Q_Line')

l_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=0, ub=30, name='L_Line')

# ------------------ ESS
soc_e = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Cap.tolist()*T, name='SOC') # SOC unit is MWh
p_ch = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Pmax.tolist()*T, name='P_Chg_Ess')
p_dc = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Pmax.tolist()*T, name='P_Dch_Ess')
x_ch = m.addVars(sets.ess_t, vtype=GRB.BINARY, name='Ch_Status')
# ------------------ PV
p_pv = m.addVars(sets.pv_t, vtype=GRB.CONTINUOUS, name='PV_Power')

neighbors = define_neighbor(sets.bus,sets.line)

P_balance = m.addConstrs(((p_ij.sum(i,'*',t) + gp.quicksum(data.resi(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - p_ij.sum('*',i,t) == p_inj[i,t] for i,t in sets.bus_t), name='P-Balance')
Q_balance = m.addConstrs(((q_ij.sum(i,'*',t) + gp.quicksum(data.reac(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - q_ij.sum('*',i,t) == q_inj[i,t] for i,t in sets.bus_t), name='Q-Balance')
V_drop = m.addConstrs((u_i[i,t] - u_i[j,t] == 2*(data.resi(i,j) * p_ij[i,j,t] + data.reac(i,j) * q_ij[i,j,t]) - (data.resi(i,j)**2 + data.reac(i,j)**2) * l_ij[i,j,t] for i,j,t in sets.line_t),name='V-Drop')
V_slack = m.addConstrs((u_i[i,t] == 1 for i in [0] for t in sets.t), name='V-Slack')

Ohm_law = m.addConstrs((l_ij[i,j,t] * u_i[j,t] >= p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')

P_injection = m.addConstrs((p_inj[i,t] == p_g.sum(i,t) + p_pv.sum(i,t) + p_dc.sum(i,t) - p_ch.sum(i,t) - data.bus.Pd[i]*data.load_profile[t] for i,t in sets.bus_t), name='P-Injection')
Q_injection = m.addConstrs((q_inj[i,t] == q_g.sum(i,t) - data.bus.Qd[i]*data.load_profile[t] for i,t in sets.bus_t), name='Q-Injection') # No battery reactive power

#----------------- ESS
SOC_time = m.addConstrs((soc_e[i,t] == soc_e[i,t-1] + (p_ch[i,t] - p_dc[i,t])*delta_t for i,t in sets.ess_t if t != 0), name='SOC_time')
SOC_init = m.addConstrs((soc_e[i,t] == soc_init*data.essdata.Cap[i] for i,t in sets.ess_t if t == 0), name='SOC_init')
m.addConstrs((p_ch[i,t] <= x_ch[i,t] * data.essdata.Pmax[i] for i,t in sets.ess_t))
m.addConstrs((p_dc[i,t] <= (1-x_ch[i,t]) * data.essdata.Pmax[i] for i,t in sets.ess_t))
# ---------------- PV
m.addConstrs((p_pv[i,t] == data.pvdata.p_max[i] * data.pv_profile[t] for i,t in sets.pv_t))

#--------------- Objective
ploss = m.addVars(sets.t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='ploss')
m.addConstrs(ploss[t] == gp.quicksum(data.resi(i,j)*l_ij[i,j,t] for i,j in sets.line) for t in sets.t)
m.setObjective(ploss.sum())


m.optimize()

for i,t in sets.ess_t:
    print('[',i,t,']',round(p_ch[i,t].X,4),' | ',round(p_dc[i,t].X,4))
