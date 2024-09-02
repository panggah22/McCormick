import pandapower.networks as pn
import pandas as pd
import numpy as np

import gurobipy as gp
from gurobipy import GRB, tuplelist

class IEEE33:
    def __init__(self,T,period):
        self.T = T
        self.div = 60/period
        self.net = pn.case33bw()
        self.kVAbase = 1000 # kVA
        self.MVAbase = self.kVAbase / 1000 # MVA
        self.impbase = self.net.bus.vn_kv[0]**2/(self.MVAbase)
        self.pvdata = None
        self.essdata = None

    def loadsys(self):
        # Line
        frombus,tobus = [],[]
        for i in self.net.line.index:
            frombus.append(self.net.bus.loc[self.net.line.from_bus[i],'name'])
            tobus.append(self.net.bus.loc[self.net.line.to_bus[i],'name'])

        self.net.line.from_bus = frombus
        self.net.line.to_bus = tobus 

        self.net.bus.index = self.net.bus.name
        self.net.line.index = [frombus,tobus]
        self.net.line = self.net.line.loc[self.net.line.in_service == True]
        self.net.line['rateA'] = 10000

        r = self.net.line.length_km * self.net.line.r_ohm_per_km
        x = self.net.line.length_km * self.net.line.x_ohm_per_km
        self.net.line = self.net.line.assign(r=r)
        self.net.line = self.net.line.assign(x=x)

        self.net.line = self.net.line.assign(r_pu=r/self.impbase)
        self.net.line = self.net.line.assign(x_pu=x/self.impbase)
        self.net.line = self.net.line[['from_bus','to_bus','r','x','r_pu','x_pu','rateA']]
        self.net.bus = self.net.bus[['name','vn_kv']]

        self.line = self.net.line

        # Bus and load
        ldp,ldq = [],[]
        for i in self.net.load.index:
            ldp.append(self.net.bus.loc[self.net.load.bus[i],'name'])
            ldq.append(self.net.bus.loc[self.net.load.bus[i],'name'])

        self.net.bus['Pd'] = 0
        self.net.bus['Qd'] = 0
        for i in self.net.load.index:
            loca = self.net.bus.loc[self.net.load.bus[i],'name']
            self.net.bus.Pd[loca] = self.net.load.p_mw[i]*1000/self.kVAbase
            self.net.bus.Qd[loca] = self.net.load.q_mvar[i]*1000/self.kVAbase

        self.bus = self.net.bus

        # Generation
        self.net.ext_grid['Pmin'] = self.net.ext_grid.min_p_mw*self.MVAbase
        self.net.ext_grid['Pmax'] = self.net.ext_grid.max_p_mw*self.MVAbase
        self.net.ext_grid['Qmin'] = self.net.ext_grid.min_q_mvar*self.MVAbase
        self.net.ext_grid['Qmax'] = self.net.ext_grid.max_q_mvar*self.MVAbase

        self.net.ext_grid.index = self.net.ext_grid.bus
        self.net.ext_grid = self.net.ext_grid[['bus','Pmin','Pmax','Qmin','Qmax']]

        self.gen = self.net.ext_grid

        loaddem = np.genfromtxt(fname='LD69-15mins.txt')
        l30 = loaddem[1:96:int(self.div)]
        l30 = np.append(l30,l30[0])
        self.load_profile = l30
        # return self.net
    
        # self.gen = self.net.ext_grid
        # self.bus = self.net.bus
        # self.line = self.net.line

    def include_pv(self,pvloc=None):
        pvdata = pd.DataFrame(data=pvloc)
        pvdata.index = pvdata.bus
        pvgen = np.genfromtxt(fname='PV45-15mins.txt')
        p_pv_max = pvgen / np.max(pvgen)

        p30 = p_pv_max[1:96:int(self.div)]
        p30 = np.append(p30,p30[0]) # Last hour same as the initial

        self.pv_profile = p30
        self.pvdata = pvdata

    def include_ess(self,essloc=None):
        essdata = pd.DataFrame(essloc)
        essdata.index = essdata.bus
        self.essdata = essdata

    def include_dg(self,dgloc=None,gci=None):
        dg = pd.DataFrame(dgloc)
        self.gen = pd.concat([self.net.ext_grid,dg])
        self.gen['gci'] = gci 
        self.gen.index = self.gen.bus

    def resi(self, m, n):
        # Using a tuple to check for membership is faster than using 'not in'
        idx = (m, n) if (m, n) in self.line.index else (n, m)
        # Accessing the 'r' column directly and then using .at for scalar value access
        return self.line.at[idx, 'r'] / self.impbase

    def reac(self, m, n,):
        # Using a tuple to check for membership is faster than using 'not in'
        idx = (m, n) if (m, n) in self.line.index else (n, m)
        # Accessing the 'x' column directly and then using .at for scalar value access
        return self.line.at[idx, 'x'] / self.impbase

class define_sets(IEEE33):
    def __init__(self, parent_instance):
        # Initialize the parent class with its existing properties
        # self.__dict__.update(parent_instance.__dict__)

        gen = parent_instance.gen # switch from the precious testsys
        bus = parent_instance.bus
        line = parent_instance.line
        
        self.t = list(range(parent_instance.T))
        
        self.nGen = gen.shape[0]
        self.gen = gen.index
        self.gen_t = tuplelist([(i,t) for t in self.t for i in self.gen])

        self.nBus = bus.shape[0]
        self.bus = bus.index
        self.bus_t = tuplelist([(i,t) for t in self.t for i in self.bus])

        self.nLine = line.shape[0]
        self.line = line.index
        self.line_t = tuplelist([(i,j,t) for t in self.t for i,j in self.line])

        if parent_instance.pvdata is not None:
            pvdata = parent_instance.pvdata
            self.nPv = pvdata.shape[0]
            self.pv = pvdata.index
            self.pv_t = tuplelist([(i,t) for t in self.t for i in self.pv])
        
        if parent_instance.essdata is not None:
            essdata = parent_instance.essdata
            self.nEss = essdata.shape[0]
            self.ess = essdata.index
            self.ess_t = tuplelist([(i,t) for t in self.t for i in self.ess])

        self.line_t_dir = tuplelist() # index for power flow direction
        for t in self.t:
            for i,j in self.line:
                self.line_t_dir = self.line_t_dir + tuplelist([(i,j,t)]) + tuplelist([(j,i,t)])

def define_neighbor(set_bus, set_line):
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

    return neighbors

def mdt(m,x,y,datapair,p=0,P=0):
    pairs = datapair

    set_l = range(p,P+1)
    set_k = range(10)
    set_kl = tuplelist([(*g,k,l) for l in set_l for k in set_k for g in pairs])
    set_z = tuplelist([(i,j,t,k,l) for l in set_l for k in set_k for i,j,t in pairs])

    # Here, 'left_set' are the variables that are discretized and 'right_set' are the variables that are continuous
    w = m.addVars(pairs, name='w')
    delta_w = m.addVars(pairs, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='delta_w')
    delta_x1 = m.addVars(pairs, lb=0, ub=10**p, name='delta_x1')

    # Indexed continuous variables (hat_x_k) and binary variables (z_k)
    hat_x = m.addVars(set_kl, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='hat_x')
    z = m.addVars(set_z, vtype=GRB.BINARY, name='z')

    m.update()
        
    m.addConstrs((w[i,j,t] == gp.quicksum(gp.quicksum(10**l * k * hat_x[i,j,t,k,l] for k in set_k) for l in set_l) + delta_w[i,j,t] for i,j,t in pairs))
    
    m.addConstrs((x[i,j,t] == gp.quicksum(gp.quicksum(10**l * k * z[i,j,t,k,l] for k in set_k) for l in set_l) + delta_x1[i,j,t] for i,j,t in pairs))

    m.addConstrs((y[j,t] == gp.quicksum(hat_x[i,j,t,k,l] for k in set_k) for l in set_l for i,j,t in pairs))

    m.addConstrs((hat_x[i,j,t,k,l] >= y[j,t].LB * z[i,j,t,k,l] for i,j,t,k,l in set_kl))
    m.addConstrs((hat_x[i,j,t,k,l] <= y[j,t].UB * z[i,j,t,k,l] for i,j,t,k,l in set_kl))

    m.addConstrs((z.sum(i,j,t,'*',l) == 1 for i,j,t,k,l in set_z))

    m.addConstrs((delta_w[i,j,t] >= y[j,t].LB * delta_x1[i,j,t] for i,j,t in pairs))
    m.addConstrs((delta_w[i,j,t] <= y[j,t].UB * delta_x1[i,j,t] for i,j,t in pairs))

    m.addConstrs((delta_w[i,j,t] <= (y[j,t] - y[j,t].LB) * 10**p + y[j,t].LB * delta_x1[i,j,t] for i,j,t in pairs))
    m.addConstrs((delta_w[i,j,t] >= (y[j,t] - y[j,t].UB) * 10**p + y[j,t].UB * delta_x1[i,j,t] for i,j,t in pairs))

    return w,delta_x1

def mdt_test(m,x,y,common,p=0,P=0):
    m.update()
    idx_x = [idx for idx,v in x.items()]
    idx_y = [idx for idx,v in y.items()]
    if len(idx_x[0]) < len(idx_y[0]):   
        pairs = idx_y
    else:
        pairs = idx_x

    set_l = range(p,P+1)
    set_k = range(10)
    set_kl = tuplelist([(*g,k,l) for l in set_l for k in set_k for g in pairs])
    set_z = tuplelist([(*g,k,l) for l in set_l for k in set_k for g in pairs])

    # Here, 'left_set' are the variables that are discretized and 'right_set' are the variables that are continuous
    w = m.addVars(pairs, name='w')
    delta_w = m.addVars(pairs, lb=-GRB.INFINITY, ub=GRB.INFINITY, name='delta_w')
    delta_x1 = m.addVars(pairs, lb=0, ub=10**p, name='delta_x1')

    # Indexed continuous variables (hat_x_k) and binary variables (z_k)
    hat_x = m.addVars(set_kl, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='hat_x')
    z = m.addVars(set_z, vtype=GRB.BINARY, name='z')

    m.update()
        
    m.addConstrs((w[i,j,t] == gp.quicksum(gp.quicksum(10**l * k * hat_x[i,j,t,k,l] for k in set_k) for l in set_l) + delta_w[i,j,t] for i,j,t in pairs))
    
    m.addConstrs((x[i,j,t] == gp.quicksum(gp.quicksum(10**l * k * z[i,j,t,k,l] for k in set_k) for l in set_l) + delta_x1[i,j,t] for i,j,t in pairs))

    m.addConstrs((y[j,t] == gp.quicksum(hat_x[i,j,t,k,l] for k in set_k) for l in set_l for i,j,t in pairs))

    m.addConstrs((hat_x[i,j,t,k,l] >= y[j,t].LB * z[i,j,t,k,l] for i,j,t,k,l in set_kl))
    m.addConstrs((hat_x[i,j,t,k,l] <= y[j,t].UB * z[i,j,t,k,l] for i,j,t,k,l in set_kl))

    m.addConstrs((z.sum(i,j,t,'*',l) == 1 for i,j,t,k,l in set_z))

    m.addConstrs((delta_w[i,j,t] >= y[j,t].LB * delta_x1[i,j,t] for i,j,t in pairs))
    m.addConstrs((delta_w[i,j,t] <= y[j,t].UB * delta_x1[i,j,t] for i,j,t in pairs))

    m.addConstrs((delta_w[i,j,t] <= (y[j,t] - y[j,t].LB) * 10**p + y[j,t].LB * delta_x1[i,j,t] for i,j,t in pairs))
    m.addConstrs((delta_w[i,j,t] >= (y[j,t] - y[j,t].UB) * 10**p + y[j,t].UB * delta_x1[i,j,t] for i,j,t in pairs))

    return w,delta_x1

def discretize_var(m,x,p,P):
    idx_x = [idx for idx,v in x.items()]
    # Input var
    set_l = range(p,P+1)
    set_k = range(10)

    set_lbd = tuplelist([(*g,k,l) for l in set_l for k in set_k for g in idx_x])

    delta_x = m.addVars(idx_x, lb=0, ub=10**p, name='delta_x1')
    lbd = m.addVars(set_lbd, vtype=GRB.BINARY, name='lambda')

def var_values(y,mult=1):
    z = []
    for v in y.values():
        z.append(v.X*mult)
    return z