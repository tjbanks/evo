import numpy as np
import pandas as pd
from neuron import h
import csv 
import math
h.load_file('stdrun.hoc')
import sys

def new_cell(gna,gk,gleak):
    
    soma = h.Section(name='soma')
    soma.diam = 200 #micrometers
    soma.L = 100 #micrometers
    soma.cm = 1.4884e-4/6.2832e-4 #uF
    soma.insert('hh')
    soma.el_hh = -70
    #soma.gnabar_hh = (gna)
    #soma.gkbar_hh = (gk)
    #soma.gl_hh = (gleak)
    soma.gnabar_hh = (gna/1000)
    soma.gkbar_hh = (gk/1000)
    soma.gl_hh = (gleak/1000000)

    return soma


def get_I_properties(soma):
    i_prop = {}

    v_init = h.v_init= -60
    stim = h.IClamp(soma(0.5))
    stim.delay = 100.0 #ms
    stim.dur = 500.0 #ms
    tstop = h.tstop = 1000   #ms
    h.dt = 0.025


    for stim.amp in np.arange(0,6,0.1):
        v0_vec = h.Vector()
        t_vec = h.Vector()
        v0_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        nc = h.NetCon(soma(0.5)._ref_v,None,sec=soma)
        nc.threshold = 0
        spvec = h.Vector()
        nc.record(spvec)
        h.run()
        
        if len(spvec) > 1 and spvec.x[0] > 100:
            
            freq = len(spvec)/stim.dur
            #writer.writerow({'gna_predicted':soma.gnabar_hh, 'gk_predicted':soma.gkbar_hh,'gleak_predicted':soma.gl_hh,'I_0':stim.amp,'F_0':freq,'num_spk':len(spvec)})  
            i_prop["I0"] = stim.amp
            break

    #Here on is unnecessary
    soma1 = new_cell(soma.gnabar_hh, soma.gkbar_hh, soma.gl_hh)   
    soma = soma1

    v_init = h.v_init= -60
    stim = h.IClamp(soma(0.5))
    stim.delay = 100.0 #ms
    stim.dur = 500.0 #ms
    tstop = h.tstop = 1000   #ms
    h.dt = 0.025

    for stim.amp in np.arange(i_prop["I0"],0,-0.01):
        v0_vec = h.Vector()
        t_vec = h.Vector()
        v0_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        nc = h.NetCon(soma(0.5)._ref_v,None,sec=soma)
        nc.threshold = 0
        spvec = h.Vector()
        nc.record(spvec)
        h.run()

        if len(spvec) == 1 and spvec.x[0] > 100:
            freq = len(spvec)/stim.dur
            #writer.writerow({'gna_predicted':soma.gnabar_hh, 'gk_predicted':soma.gkbar_hh,'gleak_predicted':soma.gl_hh,'I_0':stim.amp,'F_0':freq,'num_spk':len(spvec)})  
            i_prop["I0"] = stim.amp
            break

    return i_prop
    
def get_F_properties(soma,I_0):
    f_prop = {}

    v_init = h.v_init= -60
    stim = h.IClamp(soma(0.5))
    stim.delay = 100.0 #ms
    stim.dur = 500.0 #ms
    tstop = h.tstop = 1000   #ms
    h.dt = 0.025
    freq = []
    count = 0

    for stim.amp in np.arange(I_0,I_0+0.41,0.1):
        v0_vec = h.Vector()
        t_vec = h.Vector()
        v0_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        nc = h.NetCon(soma(0.5)._ref_v,None,sec=soma)
        nc.threshold = 0
        spvec = h.Vector()
        nc.record(spvec)
        h.run()

        if len(spvec) > 1 and spvec.x[0] > 100:
            #freq.append(len(spvec)/stim.dur)
            f_prop["F"+str(count)] = len(spvec)/stim.dur
            count = count + 1
            
    return f_prop
    

def get_passive_properties(soma):
    v_init = h.v_init= -60
    stim = h.IClamp(soma(0.5))
    stim.delay = 100.0 #ms
    stim.dur = 500.0 #ms
    tstop = h.tstop = 1000   #ms
    h.dt = 0.025
    q =int(tstop/h.dt)
    t_i = int((stim.delay+stim.dur-5)/h.dt)
    t_r = int((stim.delay-1)/h.dt)

    stim.amp = -0.1
    v0_vec = h.Vector()
    t_vec = h.Vector()
    v0_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)
    h.run()
    dt_V = v0_vec[t_i]-v0_vec[t_r]
    
    props = {}
    props["R_in"] = dt_V/stim.amp
    props["tau"] = soma.cm*props["R_in"]
    props["Vrest"] = v0_vec[30000]

    return props

def get_cell_properties(gna,gk,gleak, as_list=False):
    chan_prop = {'gna':gna,"gk":gk,"gleak":gleak}
    print(chan_prop)
    soma = new_cell(gna,gk,gleak)
    p_prop = get_passive_properties(soma)

    soma = new_cell(gna,gk,gleak)
    i_prop = get_I_properties(soma)

    soma = new_cell(gna,gk,gleak)
    f_prop = get_F_properties(soma,i_prop["I0"])
   
    all_props = dict(f_prop, **i_prop)
    all_props.update(p_prop)
    #all_props.update(chan_prop)
    
    if as_list:
        return list(all_props.values())
    else:
        return all_props

def test_a_cell(gna=132,gk=19,gleak=101):
    props = get_cell_properties(gna, gk, gleak, as_list=True)
    print(props)
    return

if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        test_a_cell(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))
    else:
        test_a_cell()