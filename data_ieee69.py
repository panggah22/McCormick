import pandas as pd
import numpy as np
import os
import sys
import pandapower as pp
import pandapower.networks as pn

def get_ieee_data():
    column_name = ['from','to','P','Q','rohm','xohm','maxi']
    path = os.getcwd() # Current folder

    data_69 = 'ieee69bus.txt' # External line data
    file69 = os.path.join(path,data_69)

    # Convert txt to dataframe
    data69 = []
    fo = open(file69,'r')
    data69 = [[float(x) for x in line.split()] for line in fo]
    df_69 = pd.DataFrame(data69,columns=column_name)

    # Float to int
    df_69['from'] = df_69['from'].astype(int)
    df_69['to'] = df_69['to'].astype(int)
    df_69['maxi'] = df_69['maxi'].astype(int)

    # Reduce the bus numbers by one, so that they range from 0~68
    df_69['from'] = df_69['from']-1
    df_69['to'] = df_69['to']-1
    
    # Convert to kilo to mega
    df_69['P'] = df_69['P']/1000
    df_69['Q'] = df_69['Q']/1000

    return df_69

def build_case69(df):
    # Create network
    net = pp.create_empty_network()

    for i in range(0,69):
        pp.create_bus(net, vn_kv=12.66, name=i)
    pp.create_ext_grid(net, bus=0, vm_pu=1.00, min_p_mw=0, max_p_mw=10, min_q_mvar=-10, max_q_mvar=10)
    
    # Create lines based on the df, and add loads to the buses 
    for index,row in df.iterrows():
        pp.create_line_from_parameters(net, from_bus=int(row['from']), to_bus=int(row['to']), length_km=1, r_ohm_per_km=float(row['rohm']), x_ohm_per_km=float(row['xohm']), c_nf_per_km=0, max_i_ka=float(row['maxi']))
        pp.create_load(net, bus=int(row['to']), p_mw=float(row['P']), q_mvar=float(row['Q']))

    return net