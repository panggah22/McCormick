import pyomo.environ as pyo
from pyomo.environ import *

from module import *


solver = 'ipopt'

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
T = 2
mins = 30
vmin, vmax = 0.9,1.1
# use_mdt = True
seqs = [0]
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
# data.include_dg(dgloc,gci)

sets = define_sets(data)

neighbors = define_neighbor(sets.bus,sets.line)
impbase = data.impbase

m = ConcreteModel()

# def set_bus_t(m):
#     return tuple(sets.bus_t)

m.set_t = Set(initialize=sets.t)
m.setGenT = Set(initialize=sets.gen_t)
m.setBusT = Set(initialize=sets.bus_t)
m.setLineT = Set(initialize=sets.line_t)
# m.neighbors = Set(initialize=neighbors.Neighbors)

m.u_i = Var(m.setBusT,within=NonNegativeReals,bounds=(vmin**2,vmax**2))

def pg_bound(model,i,t): return (0,data.gen.Pmax[i])
m.p_g = Var(m.setGenT,within=NonNegativeReals,bounds=pg_bound)

def qg_bound(model,i,t): return (data.gen.Qmin[i],data.gen.Qmax[i])
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
    return  q_out_sum + loss - q_in_sum == m.p_inj[i,t]

def V_drop_rule(m,i,j,t):
    return m.u_i[i,t] - m.u_i[j,t] == 2*(data.resi(i,j) * m.p_ij[i,j,t] + data.reac(i,j) * m.q_ij[i,j,t]) - (data.resi(i,j)**2 + data.reac(i,j)**2) * m.l_ij[i,j,t]

def V_slack_rule(m,t):
    return m.u_i[0,t] == 1

def P_injection_rule(m,i,t):
    return m.p_inj[i,t] == (m.p_g[i,t] if (i,t) in m.setGenT else 0)  - data.bus.Pd[i]

def Q_injection_rule(m,i,t):
    return m.q_inj[i,t] == (m.q_g[i,t] if (i,t) in m.setGenT else 0)  - data.bus.Qd[i]

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

m.V_slack.pprint()
print(SOLVER)
print(sets.t)

SOLVER.solve(m)
