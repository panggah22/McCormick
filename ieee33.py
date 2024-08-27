import pandapower.networks as pn
import pandas as pd
import numpy as np
from gurobipy import GRB, tuplelist

def net33(pvloc=None,dgloc=None,essloc=None):
    # Load the IEEE 33 bus and line data
    net = pn.case33bw()
    kVAbase = 1000 # kVA
    MVAbase = kVAbase / 1000 # MVA
    impbase = net.bus.vn_kv[0]**2/(MVAbase)

    # Line
    frombus,tobus = [],[]
    for i in net.line.index:
        frombus.append(net.bus.loc[net.line.from_bus[i],'name'])
        tobus.append(net.bus.loc[net.line.to_bus[i],'name'])

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

    # Bus and load
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

    # Generation
    net.ext_grid['Pmin'] = net.ext_grid.min_p_mw*MVAbase
    net.ext_grid['Pmax'] = net.ext_grid.max_p_mw*MVAbase
    net.ext_grid['Qmin'] = net.ext_grid.min_q_mvar*MVAbase
    net.ext_grid['Qmax'] = net.ext_grid.max_q_mvar*MVAbase

    # net.ext_grid.index = net.ext_grid.bus
    net.ext_grid = net.ext_grid[['bus','Pmin','Pmax','Qmin','Qmax']]

    if dgloc is not None:
        # dgloc = {'bus':[3,13,22], 'Pmin':[0,0,0], 'Pmax':[0.5,0.5,0.5], 'Qmin':[-0.3,-0.3,-0.3], 'Qmax':[0.3,0.3,0.3]}
        dg = pd.DataFrame(dgloc)
        net.ext_grid = pd.concat([net.ext_grid,dg])
        gci = [0.55,0.7,0.8,0.75]
        net.ext_grid['gci'] = gci 

    net.ext_grid.index = net.ext_grid.bus

    divisor = 2
    if pvloc is not None:
        # pvloc = {'bus':[9, 15, 19], 'p_max':[0.3, 0.3, 0.3], 'q_min':[-0.3, -0.3, -0.3], 'q_max':[0.3, 0.3, 0.3]}
        pvdata = pd.DataFrame(data=pvloc)
        pvdata.index = pvdata.bus
        pvgen = np.genfromtxt(fname='PV45-15mins.txt')
        p_pv_max = pvgen / np.max(pvgen)

        p30 = p_pv_max[1:96:divisor]
        p30 = np.append(p30,p30[0]) # Last hour same as the initial
    
    loaddem = np.genfromtxt(fname='LD69-15mins.txt')
    l30 = loaddem[1:96:divisor]
    l30 = np.append(l30,l30[0])

    if essloc is not None:
        essdata = pd.DataFrame(essloc)
        essdata.index = essdata.bus
    
    # print(net)    
    return net,pvdata,essdata

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


def define_sets(T,gen,bus,line,pvdata,essdata):
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

    nPv = pvdata.shape[0]
    set_pv = pvdata.index
    pv_t = tuplelist([(i,t) for t in set_t for i in set_pv])

    nEss = essdata.shape[0]
    set_ess = essdata.index
    ess_t = tuplelist([(i,t) for t in set_t for i in set_ess])

    
def ident_ne(set_bus,set_line):
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


# pvloc = {'bus':[9, 15, 19], 'p_max':[0.3, 0.3, 0.3], 'q_min':[-0.3, -0.3, -0.3], 'q_max':[0.3, 0.3, 0.3]}
# dgloc = {'bus':[3,13,22], 'Pmin':[0,0,0], 'Pmax':[0.5,0.5,0.5], 'Qmin':[-0.3,-0.3,-0.3], 'Qmax':[0.3,0.3,0.3]}
# essloc = {'bus':[7,21,29],'Cap':[1,1,1],'Pmin':[0,0,0],'Pmax':[0.2,0.2,0.2],'Qmin':[-0.15,-0.15,-0.15],'Qmax':[0.15,0.15,0.15]}


# net,pvdata,essdata = net33(pvloc,dgloc,essloc)
# print(net.line)