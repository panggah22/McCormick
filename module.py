import pandapower.networks as pn
import pandas as pd
import numpy as np
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

def resi(line_df, m, n, impbase=1):
    # Using a tuple to check for membership is faster than using 'not in'
    idx = (m, n) if (m, n) in line_df.index else (n, m)
    # Accessing the 'r' column directly and then using .at for scalar value access
    return line_df.at[idx, 'r'] / impbase

def reac(line_df, m, n, impbase=1):
    # Using a tuple to check for membership is faster than using 'not in'
    idx = (m, n) if (m, n) in line_df.index else (n, m)
    # Accessing the 'x' column directly and then using .at for scalar value access
    return line_df.at[idx, 'x'] / impbase

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