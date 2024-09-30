import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, Var, Set, Constraint, NonNegativeReals
from module import *


solver = 'mosek'

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
T = 1
mins = 30
vmin, vmax = 0.95,1.05
# use_mdt = True
seqs = [0]
# ------------------------------
data = IEEE33(T=T,period=mins)
data.loadsys()
sets = define_sets(data)

# print(type(tuple((sets.bus_t))))
# print(tuple(sets.bus_t))
# print(sets.bus_t)

m = ConcreteModel()

def set_bus_t(m):
    return tuple(sets.bus_t)

# m.setBusT = Set(initialize=tuple(sets.bus_t))

m.u_i = Var(Set(initialize=tuple(sets.bus_t)),within=NonNegativeReals)

m.pprint()
