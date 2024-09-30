from module import *

import gurobipy as gp
from gurobipy import GRB,tuplelist
import scipy.io
import math

import pandas as pd

# DG input
dgloc = {'bus':[3,13,22], 'Pmin':[0,0,0], 'Pmax':[0.5,0.5,0.5], 'Qmin':[-0.3,-0.3,-0.3], 'Qmax':[0.3,0.3,0.3]}
gci = [0.55,0.7,0.8,0.75]

# PV input
pvloc = {'bus':[9, 15, 19], 'p_max':[0.3, 0.3, 0.3], 'q_min':[-0.3, -0.3, -0.3], 'q_max':[0.3, 0.3, 0.3]}

# ESS input
essloc = {'bus':[7,21,29],'Cap':[1,1,1],'Pmin':[0,0,0],'Pmax':[0.2,0.2,0.2],'Qmin':[-0.15,-0.15,-0.15],'Qmax':[0.15,0.15,0.15]}

# Data preprocessing
# ----------- INPUTS -----------
T = 49
mins = 30
vmin, vmax = 0.95,1.05
# use_mdt = True
seqs = [0]
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
data.include_dg(dgloc,gci)
# data.include_pv(pvloc)
data.include_ess(essloc)

sets = define_sets(data)

df = pd.DataFrame(columns=['ObjVal','V(nonlinear)','V(u discrete)','V(l discrete)'])
for seq in seqs:
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

    l_ij = m.addVars(sets.line_t, vtype=GRB.CONTINUOUS, lb=0, ub=30, name='L_Line')


    neighbors = define_neighbor(sets.bus,sets.line)
    impbase = data.impbase

    
    p_hat = m.addVars(sets.line_t_dir, vtype=GRB.CONTINUOUS, lb=0, ub=br_lim, name='P_hat')  

    P_balance = m.addConstrs(((p_ij.sum(i,'*',t) + gp.quicksum(data.resi(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - p_ij.sum('*',i,t) == p_inj[i,t] for i,t in sets.bus_t), name='P-Balance')
    Q_balance = m.addConstrs(((q_ij.sum(i,'*',t) + gp.quicksum(data.reac(i,j) * l_ij.sum(i,j,t) for j in neighbors.Neighbors[i])) - q_ij.sum('*',i,t) == q_inj[i,t] for i,t in sets.bus_t), name='Q-Balance')
    V_drop = m.addConstrs((u_i[i,t] - u_i[j,t] == 2*(data.resi(i,j) * p_ij[i,j,t] + data.reac(i,j) * q_ij[i,j,t]) - (data.resi(i,j)**2 + data.reac(i,j)**2) * l_ij[i,j,t] for i,j,t in sets.line_t),name='V-Drop')
    V_slack = m.addConstrs((u_i[i,t] == 1 for i in [0] for t in sets.t), name='V-Slack')

    if data.essdata is not None:
    ## ESS Variables
        soc_e = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Cap.tolist()*T, name='SOC') # SOC unit is MWh
        p_ch = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Pmax.tolist()*T, name='P_Chg_Ess')
        p_dc = m.addVars(sets.ess_t, vtype=GRB.CONTINUOUS, lb=0, ub=data.essdata.Pmax.tolist()*T, name='P_Dch_Ess')
        x_ch = m.addVars(sets.ess_t, vtype=GRB.BINARY, name='Ch_Status')

        delta_t = 60/mins
        SOC_time = m.addConstrs((soc_e[i,t] == soc_e[i,t-1] + (p_ch[i,t] - p_dc[i,t])*delta_t for i,t in sets.ess_t if t != 0), name='SOC_time')
        SOH_init = m.addConstrs((soc_e[i,t] == 0.5*data.essdata.Cap[i] for i,t in sets.ess_t if t == 0), name='SOC_init')
        SOH_loop = m.addConstrs((soc_e[i,0] == soc_e[i,T-1] for i in sets.ess), name='SOC_init')

        # -------- Charging ----------
        # P_charging = m.addConstrs((p_ch[i,t] == p_ch_grid[i,t] + p_ch_pv[i,t] for i,t in ess_t), name='P_Chg_Combi')
        P_ch_status = m.addConstrs((p_ch[i,t] <= data.essdata.Pmax[i]*x_ch[i,t] for i,t in sets.ess_t), name='Chg-Status')
        P_dc_status = m.addConstrs((p_dc[i,t] <= data.essdata.Pmax[i]*(1-x_ch[i,t]) for i,t in sets.ess_t), name='Dch-Status')
    else:
        m.addConstrs((p_ch[i,t] == 0 for i,t in sets.ess_t))
        m.addConstrs((p_dc[i,t] == 0 for i,t in sets.ess_t))

    if T == 1:
        P_injection = m.addConstrs((p_inj[i,t] == p_g.sum(i,t) - data.bus.Pd[i]*1 for i,t in sets.bus_t), name='P-Injection')
        Q_injection = m.addConstrs((q_inj[i,t] == q_g.sum(i,t) - data.bus.Qd[i]*1 for i,t in sets.bus_t), name='Q-Injection')
    else:
        P_injection = m.addConstrs((p_inj[i,t] == p_g.sum(i,t) - data.bus.Pd[i]*data.load_profile[t] + p_dc.sum(i,t) - p_ch.sum(i,t) for i,t in sets.bus_t), name='P-Injection')
        Q_injection = m.addConstrs((q_inj[i,t] == q_g.sum(i,t) - data.bus.Qd[i]*data.load_profile[t] for i,t in sets.bus_t), name='Q-Injection')

    if seq == 1:
        aux_1, delta_x = mdt(m,l_ij,u_i,sets.line_t,p=-2,P=1)
        Ohm_law = m.addConstrs((aux_1[i,j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')
    elif seq == 2:
        aux_1, delta_x = mdt_iij(m,u_i,l_ij,sets.line_t,p=-4,P=1)
        Ohm_law = m.addConstrs((aux_1[i,j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')
    else:
        Ohm_law = m.addConstrs((l_ij[i,j,t] * u_i[j,t] == p_ij[i,j,t]**2 + q_ij[i,j,t]**2 for i,j,t in sets.line_t), name='Ohms-Law')

    # Direction replacement
    p_line_dir = m.addConstrs((p_ij[i,j,t] == p_hat[i,j,t] - p_hat[j,i,t] for i,j,t in sets.line_t), name='Line-Direction')

    # v--- p_hat_ij * p_hat_ji = 0 ---v
    for t in sets.t: # Somehow Special Ordered Set is sometimes faster than binary linearization
        for i,j in sets.line:
            m.addSOS(GRB.SOS_TYPE1,[p_hat[i,j,t], p_hat[j,i,t]],[1,1])

    w_i = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=0, ub=1, name='node_int')
    r_g = m.addVars(sets.gen_t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='rate_gen')
    
    Rate_gen = m.addConstrs((r_g[i,t] == data.gen.gci[i] * p_g[i,t] for i,t in sets.gen_t), name='Gen-emission-rate')
    Int_balance = m.addConstrs((w_i[i,t]*p_g.sum(i,t) + w_i[i,t]*p_hat.sum('*',i,t) >= r_g.sum(i,t) + gp.quicksum(w_i[j,t] * p_hat[j,i,t] for j in neighbors.Neighbors[i]) for i,t in sets.bus_t),name='Int-balance') # Without ESS CEF

    penalty = m.addVars(sets.bus_t, lb=0,ub=1,name='penalty')
    p_hs = m.addVars(sets.bus_t, vtype=GRB.CONTINUOUS, lb=0, ub=3, name='p_hat_sum')
    m.addConstrs((p_hs[i,t] == p_hat.sum('*',i,t) for i,t in sets.bus_t))

    # aux_2,del_wi = mdt_ii(m,w_i,p_hs,sets.bus_t,p=-2,P=1)
    # aux_2,del_wi = mdt_ii(m,p_hs,w_i,sets.bus_t,p=-1,P=0)
    # Int_balance = m.addConstrs((w_i[i,t]*p_g.sum(i,t) + aux_2[i,t] >= r_g.sum(i,t) + gp.quicksum(w_i[j,t] * p_hat[j,i,t] for j in neighbors.Neighbors[i]) for i,t in sets.bus_t),name='Int-balance') # Without ESS CEF

    # ---------------- OBJECTIVE FUNCTION ----------------
    ploss = m.addVars(sets.t, vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name='ploss')
    m.addConstrs(ploss[t] == gp.quicksum(data.resi(i,j)*l_ij[i,j,t] for i,j in sets.line) for t in sets.t)
    m.setObjective(ploss.sum())

    m.update()
    m.write('test.lp')

    # m.Params.TimeLimit = 180
    # m.Params.OutputFlag = 0
    m.optimize()

    # voltage = [math.sqrt(i.X) for i in u_i.values()]
    # print(min(voltage), max(voltage))
    # print(p_g)
    # print(max([i.X for i in l_ij.values()]))

    # for i,j,t in sets.line_t:
    #     # print(abs(l_ij[i,j,t].X * u_i[i,t].X - w[i,j,t].X)/(l_ij[i,j,t].X * u_i[i,t].X)*100)
    #     if seq == 0:
    #         print(l_ij[i,j,t].X * u_i[i,t].X)
    #     else:
    #         print(w[i,j,t].X)
        # print(l_ij[i,j,t].X * u_i[i,t].X)
        # df2 = pd.Dataframe

    print('----------------')    
    # print(sets.line_t[0],sets.line_t[1])
    print('The optimization runs in ',m.Runtime)
    
   
    # if seq != 0:
    #     for i,j,t in sets.line_t:
    #         # print(p_ij[i,j,t].X, ';', p_hat[i,j,t].X)
    #         print(aux_1[i,j,t].X, ';', u_i[i,t].X*l_ij[i,j,t].X)
    
    for i,t in sets.bus_t:
    #     print(w_i[i,t].X)
        print(p_hs[i,t].X)

    # print(p_g)