import json
from scipy.signal import find_peaks
from vm_plotter import plot_stim_volts_pair
from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from scalebary import add_scalebar
my_dpi = 96
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
#plt.rcParams['font.sans-serif'] = "Arial"
#plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
tick_major = 6
tick_minor = 4
plt.rcParams["xtick.major.size"] = tick_major
plt.rcParams["xtick.minor.size"] = tick_minor
plt.rcParams["ytick.major.size"] = tick_major
plt.rcParams["ytick.minor.size"] = tick_minor
font_small =9
font_medium = 13
font_large = 14
plt.rc('font', size=font_small)          # controls default text sizes
plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_small)    # legend fontsize
plt.rc('figure', titlesize=font_large)   # fontsize of the figure title
"""
ntimestep = 10000
dt = 0.02
def_times = np.array([dt for i in range(ntimestep)])
def_times = np.cumsum(def_times)
"""
def cm_to_in(cm):
    return cm/2.54




def get_fi_curve(mdl,s_amp,e_amp,nruns,wt_data=None,ax1=None,fig = None,dt = 0.025,fn = './Plots/ficurve.pdf'):
    all_volts = []
    npeaks = []
    x_axis = np.linspace(s_amp,e_amp,nruns)
    stim_length = int(2000/dt)
    pik_height = 10
    pik_dist = 0.1/dt
    for curr_amp in x_axis:
        mdl.init_stim(amp = curr_amp,dt = dt,sweep_len = 2000,stim_start = 300, stim_dur = 1000)
        curr_volts,_,_,_ = mdl.run_model(dt = dt)
        curr_peaks,_ = find_peaks(curr_volts[:stim_length],prominence = pik_height, distance = pik_dist)
        all_volts.append(curr_volts)
        npeaks.append(len(curr_peaks))
    print(npeaks)
    if ax1 is None:
        fig,ax1 = plt.subplots(1,1)
        ax1.plot(x_axis,npeaks,marker = 'o',linestyle = '-',color = 'red')
    ax1.set_title('FI Curve')
    ax1.set_xlabel('Stim [nA]')
    ax1.set_ylabel('nAPs for 500ms epoch')
    if wt_data is None:
        #fig.show()
        fig.savefig(fn)
        return npeaks
    else:
        ax1.plot(x_axis,wt_data,marker = 'o',linestyle = '-',color = 'black')
        #ax1.plot(x_axis,wt_data,'black')
    fig.show()
    fig.savefig(fn)

def plot_dvdt_from_volts(volts,dt,axs=None,clr = 'black',skip_first = False):
    pik_height = 25
    pik_dist = 2/dt

    if skip_first:
        curr_peaks,_ = find_peaks(volts,prominence = pik_height, distance = pik_dist)
        volts = volts[curr_peaks[0]+int(3/dt):]
    if axs is None:
        fig,axs = plt.subplots(1,1)
    dvdt = np.gradient(volts)/dt
    axs.plot(volts, dvdt, color = clr)
    return axs

def plot_dg_dt(g,volts,dt,axs=None,clr = 'black'):
    if axs is None:
        fig,axs = plt.subplots(1,1)
    dgdt = np.gradient(g)/dt
    axs.plot(volts, dgdt, color = clr)

def plot_extra_volts(t,extra_vms,axs = None,clr = 'black'):
    if axs is None:
        fig,axs = plt.subplots(3,figsize=(cm_to_in(20),cm_to_in(20)))
    axs[0].plot(t,extra_vms['ais'], label='ais', color=clr,linewidth=1)
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].locator_params(axis='y', nbins=8)
    axs[0].set_title('AIS')
    axs[1].plot(t,extra_vms['nexus'], label='nexus', color=clr,linewidth=1)
    axs[1].locator_params(axis='x', nbins=5)
    axs[1].locator_params(axis='y', nbins=8)
    axs[1].set_title('Nexus')
    axs[2].plot(t,extra_vms['dist_dend'], label='dist_dend', color=clr,linewidth=1)
    axs[2].locator_params(axis='x', nbins=5)
    axs[2].locator_params(axis='y', nbins=8)
    axs[2].set_title('dist_dend')


def update_mech_from_dict(mdl,dict_fn,mechs,input_dict = False):
    if input_dict:
        param_dict = dict_fn
    else:
        with open(dict_fn) as f:
            data = f.read()
        param_dict = json.loads(data)
    print(f'updating {mechs} with {param_dict}')
    for curr_sec in mdl.sl:
        #print(curr_sec)
        for curr_mech in mechs:
            #print(curr_mech)
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                #print(curr_name)
                for p_name in param_dict.keys():
                    #print(f'p_name is {p_name}')
                    hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} = {param_dict[p_name]}'
                    #print(hoc_cmd)
                    h(hoc_cmd)
                #in case we need to go per sec:
                    for seg in curr_sec:
                        hoc_cmd = f'{curr_name}.{p_name}_{curr_mech}({seg.x}) = {param_dict[p_name]}'
                        print(hoc_cmd)
    return param_dict

def update_mod_param(mdl,mechs,mltplr,gbar_name = 'gbar',print_flg = False):
    for curr_sec in mdl.sl:
        curr_name = h.secname(sec=curr_sec)
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                for seg in curr_sec:
                    hoc_cmd = f'{curr_name}.{gbar_name}_{curr_mech}({seg.x}) *= {mltplr}'
                    #print(hoc_cmd)
                    par_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    h(hoc_cmd)
                    assigned_value = h(f'{curr_name}.{gbar_name}_{curr_mech}({seg.x})')
                    if print_flg:
                        print(f'{curr_name}_{curr_mech}_{seg}_par_value before{par_value} and after {assigned_value}')
    

def multiply_param(mdl,mechs,p_name,multiplier):
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} *= {multiplier}'
                #print(hoc_cmd)
                h(hoc_cmd)
def offset_param(mdl,mechs,p_name,offset):
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} += {offset}'
                print(hoc_cmd)
                h(hoc_cmd)
def update_param_value(mdl,mechs,p_name,value):
    for curr_sec in mdl.sl:
        for curr_mech in mechs:
            if h.ismembrane(curr_mech, sec=curr_sec):
                curr_name = h.secname(sec=curr_sec)
                hoc_cmd = f'{curr_name}.{p_name}_{curr_mech} = {value}'
                print(hoc_cmd)
                h(hoc_cmd)
#### Emily's code
def update_channel(mdl, channel_name, channel, dict_fn, wt_mul, mut_mul):
    """
    channel_name: str e.g 'na16mut'
    channel: str e.g. 'na16'
    """
    with open(dict_fn) as f:
        data = f.read()
    param_dict = json.loads(data)
    for curr_sec in mdl.sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_{channel_name}({seg.x}) *= {mut_mul}'
                #print(hoc_cmd)
                h(hoc_cmd)
            for p_name in param_dict.keys():
                hoc_cmd = f'{curr_name}.{p_name} = {param_dict[p_name]}'
                #print(hoc_cmd)
                h(hoc_cmd)
        if h.ismembrane(channel, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.gbar_{channel}({seg.x}) *= {wt_mul}'
                #print(hoc_cmd)
                h(hoc_cmd)


def update_K(mdl, channel_name, gbar_name, mut_mul):
    k_name = f'{gbar_name}_{channel_name}'
    prev = []
    for curr_sec in mdl.sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = f'{curr_name}.{k_name}({seg.x}) *= {mut_mul}'
                print(hoc_cmd)
                h(f'a = {curr_name}.{k_name}({seg.x})')  # get old value
                prev_var = h.a
                prev.append(f'{curr_name}.{k_name}({seg.x}) = {prev_var}')  # store old value in hoc_cmd
                h(hoc_cmd)
    return prev


def reverse_update_K(mdl, channel_name, gbar_name, prev):
    k_name = f'{gbar_name}_{channel_name}'
    index = 0
    for curr_sec in mdl.sl:
        if h.ismembrane(channel_name, sec=curr_sec):
            curr_name = h.secname(sec=curr_sec)
            for seg in curr_sec:
                hoc_cmd = prev[index]
                h(hoc_cmd)
                index += 1

def plot_stim(mdl, amp,fn,clr='blue'):
    mdl.init_stim(amp=amp)
    Vm, I, t, stim = mdl.run_model()
    plot_stim_volts_pair(Vm, f'Step Stim {amp}pA', file_path_to_save=f'./Plots/V1/{fn}_{amp}pA',times=t,color_str=clr)
    return I

def plot_FIs(fis, extra_cond = False):
    data = fis
    # save multiple figures in one pdf file
    filename= f'Plots/FI_plots.pdf'
    fig = plt.figure()
    x_axis, npeaks, name = data[0]
    plt.plot(x_axis, npeaks, label=name, color='black')
    # plot mut
    x_axis, npeaks, name = data[1]
    plt.plot(x_axis, npeaks, label=name, color='red')
    if extra_cond:
        # plot wtTTX
        x_axis, npeaks, name = data[2]
        plt.plot(x_axis, npeaks, label=name, color='black', linestyle='dashed')
        # plot mutTTX
        x_axis, npeaks, name = data[3]
        plt.plot(x_axis, npeaks, label=name, color='red', linestyle='dashed')

    plt.legend()
    plt.xlabel('Stim [nA]')
    plt.ylabel('nAPs for 600ms epoch')
    plt.title(f'FI Curve')
    fig.savefig(filename)


def plot_all_FIs(fis, extra_cond = False):
    for i in range(len(fis)):
        data = fis[i]
        # save multiple figures in one pdf file
        filename= f'Plots/FI_plots{i}.pdf'
        fig = plt.figure()
        x_axis, npeaks, name = data[0]
        plt.plot(x_axis, npeaks, label=name, color='black')
        # plot mut
        x_axis, npeaks, name = data[1]
        plt.plot(x_axis, npeaks, label=name, color='red')
        if extra_cond:
            # plot wtTTX
            x_axis, npeaks, name = data[2]
            plt.plot(x_axis, npeaks, label=name, color='black', linestyle='dashed')
            # plot mutTTX
            x_axis, npeaks, name = data[3]
            plt.plot(x_axis, npeaks, label=name, color='red', linestyle='dashed')

        plt.legend()
        plt.xlabel('Stim [nA]')
        plt.ylabel('nAPs for 500ms epoch')
        plt.title(f'FI Curve: for range {i}')
        fig.savefig(filename)
def scan12_16():
    for i12 in np.arange(0.5,1.5,0.1):
        for i16 in np.arange(0.5,1.5,0.1):
            sim = Na1612Model(nav12=i12, nav16=i16)
            sim.make_wt()
            fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(20),cm_to_in(20)))
            sim.plot_stim(axs = axs[0],stim_amp = 0.7,dt=0.005)
            NH.plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
            fn = f'./Plots/na1216_trials/vs_dvdt12_{i12}_16_{i16}.pdf'
            fig_volts.savefig(fn)

def get_spike_times(volts,times):
    inds,peaks = find_peaks(volts,height = -20)
    ans = [times[x] for x in inds]
    return ans