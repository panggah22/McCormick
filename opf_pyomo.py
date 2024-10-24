import pyomo.environ as pyo
from pyomo.environ import *

from module import *
import matplotlib.pyplot as plt

solver = 'couenne'

SOLVER = pyo.SolverFactory(solver)
assert SOLVER.available(), f"Solver {solver} is not available."

# DG input
dgloc = {'bus':[3,13,22], 'Pmin':[0,0,0], 'Pmax':[0.5,0.5,0.5], 'Qmin':[-0.3,-0.3,-0.3], 'Qmax':[0.3,0.3,0.3]}
gci = [0.55,0.7,0.8,0.75]

# PV input
pvloc = {'bus':[9, 15, 19], 'p_max':[0.3, 0.3, 0.3], 'q_min':[-0.3, -0.3, -0.3], 'q_max':[0.3, 0.3, 0.3]}

# ESS input
essloc = {'bus':[7,21,29],'Cap':[1,1,1],'Pmin':[0,0,0],'Pmax':[0.2,0.2,0.2],'Qmin':[-0.15,-0.15,-0.15],'Qmax':[0.15,0.15,0.15]}

# ----------- INPUTS -----------
T = 25
mins = 30
vmin, vmax = 0.9,1.1
# use_mdt = True
seqs = [0]
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
data.include_dg(dgloc,gci)
data.include_ess(essloc)

sets = define_sets(data)

neighbors = define_neighbor(sets.bus,sets.line)
impbase = data.impbase

m = ConcreteModel()

# def set_bus_t(m):
#     return tuple(sets.bus_t)

## DEFINE SETS
m.set_t = Set(initialize=sets.t)
m.setGenT = Set(initialize=sets.gen_t)
m.setBusT = Set(initialize=sets.bus_t)
m.setLineT = Set(initialize=sets.line_t)
m.setEssT = Set(initialize=sets.ess_t)
# m.neighbors = Set(initialize=neighbors.Neighbors)

m.u_i = Var(m.setBusT,within=NonNegativeReals,bounds=(vmin**2,vmax**2))

def pg_bound(m,i,t): return (0,data.gen.Pmax[i])
m.p_g = Var(m.setGenT,within=NonNegativeReals,bounds=pg_bound)

def qg_bound(m,i,t): return (data.gen.Qmin[i],data.gen.Qmax[i])
m.q_g = Var(m.setGenT,bounds=qg_bound)

m.p_inj = Var(m.setBusT)
m.q_inj = Var(m.setBusT)

br_lim = 4
m.p_ij = Var(m.setLineT,bounds=(-br_lim,br_lim))
m.q_ij = Var(m.setLineT,bounds=(-br_lim,br_lim))
m.l_ij = Var(m.setLineT,bounds=(0,30))

# def src(m, i, t):
#     return sum(m.p_ij[i,j,t] for j in neighbors.Neighbors[i]) == m.p_inj[i,t]
# m.P_balance = Constraint(m.setBusT,rule=src)
def P_balance_rule(m, i, t):
    p_out_sum = sum(m.p_ij[i, j, t] for j in neighbors.Neighbors[i] if (i, j, t) in m.setLineT)
    loss = sum(data.resi(i, j) * m.l_ij[i, j, t] for j in neighbors.Neighbors[i] if (i, j, t) in m.setLineT)
    p_in_sum = sum(m.p_ij[j, i, t] for j in neighbors.Neighbors[i] if (j, i, t) in m.setLineT)
    return  p_out_sum + loss - p_in_sum == m.p_inj[i,t]

def Q_balance_rule(m, i, t):
    q_out_sum = sum(m.q_ij[i, j, t] for j in neighbors.Neighbors[i] if (i, j, t) in m.setLineT)
    loss = sum(data.reac(i, j) * m.l_ij[i, j, t] for j in neighbors.Neighbors[i] if (i, j, t) in m.setLineT)
    q_in_sum = sum(m.q_ij[j, i, t] for j in neighbors.Neighbors[i] if (j, i, t) in m.setLineT)
    return  q_out_sum + loss - q_in_sum == m.q_inj[i,t]

def V_drop_rule(m,i,j,t):
    return m.u_i[i,t] - m.u_i[j,t] == 2*(data.resi(i,j) * m.p_ij[i,j,t] + data.reac(i,j) * m.q_ij[i,j,t]) - (data.resi(i,j)**2 + data.reac(i,j)**2) * m.l_ij[i,j,t]

def V_slack_rule(m,t):
    return m.u_i[0,t] == 1

## ESS
if data.essdata is not None:
    def soc_bound(m,i,t): return (0,data.essdata.Cap[i])
    m.soc_e = Var(m.setEssT, bounds=soc_bound)

    def pess_bound(m,i,t): return (0,data.essdata.Pmax[i])
    m.p_ch = Var(m.setEssT, bounds=pess_bound)
    m.p_dc = Var(m.setEssT, bounds=pess_bound)
    m.x_ch = Var(m.setEssT, within=Binary)

    delta_t = 60/mins
    def soc_time_rule(m,i,t):
        if t == 0:
            return m.soc_e[i,t] == 0.5*data.essdata.Cap[i]
        else:
            return m.soc_e[i,t] == m.soc_e[i,t-1] + (m.p_ch[i,t] - m.p_dc[i,t])*delta_t
    m.Soc_time = Constraint(m.setEssT, rule=soc_time_rule)

    def p_charge_rule(m,i,t):
        if t == 0:
            return m.p_ch[i,t] == 0
        else:
            return m.p_ch[i,t] <= data.essdata.Pmax[i]*m.x_ch[i,t]
    m.P_ch_status = Constraint(m.setEssT, rule=p_charge_rule)

    def p_discharge_rule(m,i,t):
        if t == 0:
            return m.p_dc[i,t] == 0
        else:
            return m.p_dc[i,t] <= data.essdata.Pmax[i]*(1-m.x_ch[i,t])
    m.P_dc_status = Constraint(m.setEssT, rule=p_discharge_rule)

def P_injection_rule(m,i,t):
    if data.essdata is not None:
        return m.p_inj[i,t] == (m.p_g[i,t] if (i,t) in m.setGenT else 0) + (m.p_dc[i,t] - m.p_ch[i,t] if (i,t) in m.setEssT else 0)  - data.bus.Pd[i]*data.load_profile[t]
    else:
        return m.p_inj[i,t] == (m.p_g[i,t] if (i,t) in m.setGenT else 0)  - data.bus.Pd[i]*data.load_profile[t]

def Q_injection_rule(m,i,t):
    return m.q_inj[i,t] == (m.q_g[i,t] if (i,t) in m.setGenT else 0)  - data.bus.Qd[i]*data.load_profile[t]

def Ohm_law_rule(m,i,j,t):
    return m.l_ij[i,j,t] * m.u_i[j,t] >= m.p_ij[i,j,t]**2 + m.q_ij[i,j,t]**2
    # return SecondOrderCone(model.l_ij[i,j,t] * model.u_i[j,t], [model.p_ij[i,j,t], model.q_ij[i,j,t]])

m.P_balance = Constraint(m.setBusT, rule=P_balance_rule)
m.Q_balance = Constraint(m.setBusT, rule=Q_balance_rule)
m.V_drop = Constraint(m.setLineT, rule=V_drop_rule)
m.V_slack = Constraint(m.set_t, rule=V_slack_rule)
m.P_injection = Constraint(m.setBusT, rule=P_injection_rule)
m.Q_injection = Constraint(m.setBusT, rule=Q_injection_rule)
m.Ohm_law = Constraint(m.setLineT, rule=Ohm_law_rule)


## MDT base-2
def mdt_iij(m,x,y,datapair,p=0,P=0):
    pairs = datapair
    idx_x = [idx for idx,v in x.items()]
    
    set_l = range(p,P+1)
    set_kl = [(*g,l) for l in set_l for g in pairs]
    set_z = [(i,t,l) for l in set_l for i,t in idx_x]

    w = Var(pairs)
    delta_w = Var(pairs)
    delta_x1 = Var(idx_x, bounds=(0,2**p))
    hat_x = Var(set_kl)
    z = Var(set_z, within=Binary)

    def aux1_rule(m,i,j,t): # set: pairs
        return w[i,j,t] == (x[j,t].ub - x[j,t].lb) * sum(2**l *hat_x[i,j,t,l] for l in set_l) + delta_w[i,j,t]
    
    # sliced = [(j,t) for (i,j,t) in pairs] # DISABLE THIS ONE, just for testing
    # def aux2_rule(m,j,t): # set: idx_x sliced with pairs
    #     return x[j,t] == sum(2**l * z[j,t,l] for l in set_l) + delta_x1[j,t]
    
    def aux2_rule(m,i,j,t): # set: pairs
        return x[j,t] == (x[j,t].ub - x[j,t].lb) * sum(2**l * z[j,t,l] for l in set_l) + delta_x1[j,t]
    
    # def aux3_rule(m,i,j,t): # set: pairs
    #     return y[i,j,t] == sum(hat_x[i,j,t,l] for l in set_l)
    
    def xhat_lb_rule(m,i,j,t,l): # set: set_kl
        return hat_x[i,j,t,l] >= y[i,j,t].lb * z[j,t,l]
    def xhat_ub_rule(m,i,j,t,l):
        return hat_x[i,j,t,l] <= y[i,j,t].ub * z[j,t,l]
    
    def delta_w_lb_rule(m,i,j,t): #pairs
        return delta_w[i,j,t] >= y[i,j,t].lb * delta_x1[j,t]
    def delta_w_ub_rule(m,i,j,t):
        return delta_w[i,j,t] <= y[i,j,t].ub * delta_x1[j,t]

    def delta_w_mc1_rule(m,i,j,t): # pairs
        return delta_w[i,j,t] <= (y[i,j,t] - y[i,j,t].lb) * 2**p + y[i,j,t].lb * delta_x1[j,t]
    def delta_w_mc2_rule(m,i,j,t): # pairs
        return delta_w[i,j,t] >= (y[i,j,t] - y[i,j,t].ub) * 2**p + y[i,j,t].ub * delta_x1[j,t]
    
    def newconst_lb_rule(m,i,j,t,l): # set_kl
        return y[i,j,t] - hat_x[i,j,t,l] >= y[i,j,t].lb*(1-z[j,t,l])
    def newconst_ub_rule(m,i,j,t,l): # set_kl
        return y[i,j,t] - hat_x[i,j,t,l] <= y[i,j,t].ub*(1-z[j,t,l])
    
    # Create the constraints
    aux1 = Constraint(pairs, rule=aux1_rule)
    aux2 = Constraint(pairs, rule=aux2_rule)
    aux3_a = Constraint(pairs, rule=delta_w_mc1_rule)
    aux3_b = Constraint(pairs, rule=delta_w_mc2_rule)
    aux4_a = Constraint(pairs, rule=delta_w_lb_rule)
    aux4_b = Constraint(pairs, rule=delta_w_ub_rule)
    aux5_a = Constraint(set_kl, rule=xhat_lb_rule)
    aux5_b = Constraint(set_kl, rule=xhat_ub_rule)
    aux6_a = Constraint(set_kl, rule=newconst_lb_rule)
    aux6_b = Constraint(set_kl, rule=newconst_ub_rule)

    return w, delta_w, delta_x1, hat_x, z,\
         aux1, aux2, aux3_a, aux3_a, aux3_b, aux4_a, aux4_b,\
         aux5_a, aux5_b, aux6_a, aux6_b


def loss_objective(m):
    return sum(m.l_ij[i,j,t]*data.resi(i,j) for i,j,t in m.setLineT)
m.Objective = Objective(rule=loss_objective,sense=minimize)

# import math
# print(m.l_ij.lb())
m.w, m.delta_w, m.delta_x1, m.hat_x, m.z,\
m.aux1, m.aux2, m.aux3_a, m.aux3_a, m.aux3_b, m.aux4_a, m.aux4_b,\
m.aux5_a, m.aux5_b, m.aux6_a, m.aux6_b \
= mdt_iij(m,m.u_i,m.l_ij,sets.line_t,p=0,P=2)

# m.Ohm_law.deactivate()
def Ohm_law_new_rule(m,i,j,t):
    return m.w[i,j,t] >= m.p_ij[i,j,t]**2 + m.q_ij[i,j,t]**2
# m.Ohm_law_new = Constraint(m.w.index_set(), rule=Ohm_law_new_rule)
# m.Ohm_law_new.pprint()
# m.aux1.pprint()
# m.w.pprint()
# m.Aux_con2.pprint()
# m.V_drop.pprint()
# print(SOLVER)
# print(sets.t)

def disp_vars(x):
# First, create an empty dictionary to store the values in a nested format
    data = {}

    # Iterate through the keys of model.x (assuming it's a 2D indexed variable)
    for key in x.keys():
        row_key = key[1]  # First part of the tuple (row)
        col_key = key[0]  # Second part of the tuple (column)
        
        if row_key not in data:
            data[row_key] = {}
            
        # Assign the value of model.x at the given key
        data[row_key][col_key] = x[key].value

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Optionally, you can add labels to rows and columns
    df.index.name = 'Row'
    df.columns.name = 'Column'

    return df

SOLVER.solve(m,tee=True)
print(disp_vars(m.soc_e))
plt.plot(disp_vars(m.soc_e))