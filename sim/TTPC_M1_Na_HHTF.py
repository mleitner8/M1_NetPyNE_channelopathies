from NeuronModelClass import NeuronModel
from NrnHelper import *
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import csv
import numpy as np
#from currentscape.currentscape import plot_currentscape

class Na1612Model_TF:
    #def __init__(self,na12name = 'na12_orig1', na12mechs = ['na12','na12mut'],na16name = 'na16_orig2', na16mechs = ['na16','na16mut'], params_folder = './params/',nav12=1,nav16=1,K=1,KT=1,KP=1,somaK=1,ais_ca = 1,ais_Kca = 1,soma_na16=1,soma_na12 = 1,node_na = 1,plots_folder = f'./Plots/'):
    #def __init__(self,na12name = 'na12_orig1', na12mechs = ['na12','na12mut'],na16name = 'na16_orig2', na16mechs = ['na16','na16mut'], params_folder = './params/',nav12=1,nav16=1,K=1,KT=1,KP=1,somaK=1,ais_ca = 1,ais_Kca = 1,soma_na16=1,soma_na12 = 1,node_na = 1,plots_folder = f'./Plots/'):
    def __init__(self, na12mechs = ['na12','na12mut'], na16mechs = ['na16','na16mut'], params_folder = './params/',nav12=1,nav16=1,K=1,KT=1,KP=1,somaK=1,ais_ca = 1,ais_Kca = 1,soma_na16=1,soma_na12 = 1,node_na = 1,plots_folder = f'./Plots/12HH16HH/100923/scan_soma1216_1nA/'):
        
        
        # ais_Kca = 0.03*ais_Kca
        # ais_ca = 0.04*ais_ca
        
        # nav12 = 1.8 *nav12
        # nav16 = 1.8 *nav16#the wt should be 1.1 and then add to that what we get from the input
        # nav16 = 3.6 #2*1.8
        # nav12 = 3.6 #2*1.8
        
        nav16 = 3
        nav12 = 3
        
        
        # KP = 1.2*KP
        # somaK = 0.5 * somaK
        # KP=0.95*KP
        # K = 4.8*K
        # KT = 0.025*0.5*KT

        ##Tim Param Search
        # ais_Kca = 4
        # ais_ca = 1
        # KP = 4
        # somaK = 4
        # KP=4
        # K = 10
        # KT = 4
        # soma_na16 = 1
        soma_na12 = 10

        
        #nav12 = 1.2
        #nav16 = 1.2
        #soma_na16 = 0.7
        #soma_na12 = 0.7
        #K = 0.6
        #update_param_value(self.l5mdl,['SKv3_1'],'vtau',25)
        

        self.l5mdl = NeuronModel(mod_dir = './Neuron_Model_HH/',nav12=nav12, nav16=nav16,axon_K = K,axon_Kp = KP,axon_Kt = KT,soma_K = somaK,ais_ca = ais_ca,ais_KCa=ais_Kca,soma_nav16=soma_na16,soma_nav12 = soma_na12,node_na = node_na)
        #self.l5mdl = NeuronModel(mod_dir = './Neuron_Model_HH/')
        #update_param_value(self.l5mdl,['SKv3_1'],'mtaumul',6)
        
        #update_param_value(self.l5mdl,['SKv3_1'],'vtau',25)
        self.params_folder = params_folder
        self.na12mechs = na12mechs
        self.na16mechs = na16mechs
        self.plot_folder = plots_folder 
        self.plot_folder = f'{plots_folder}'
        Path(self.plot_folder).mkdir(parents=True, exist_ok=True)
        """
        print(f'using na12_file {na12name}')
        p_fn_na12 = f'{params_folder}{na12name}.txt'
        self.na12_p = update_mech_from_dict(self.l5mdl, p_fn_na12, self.na12mechs) 
    
        print(f'using na16_file {na16name}')
        p_fn_na16 = f'{params_folder}{na16name}.txt'
        self.na16_p = update_mech_from_dict(self.l5mdl, p_fn_na16, self.na16mechs) 
        """
    def make_mut(self,mut_mech,mut_params_fn):
        print(f'updating mut {mut_mech} with {mut_params_fn}')
        self.na16_p_mut = update_mech_from_dict(self.l5mdl, self.params_folder + mut_params_fn, mut_mech) 

    def update_gfactor(self,gbar_factor = 1):
        update_mod_param(self.l5mdl, self.mut_mech, gbar_factor, gbar_name='gbar')

    def plot_stim(self,stim_amp = 0.3,dt = 0.1,clr = 'black',plot_fn = 'step',axs = None,rec_extra = False):
        self.dt = dt
        if not axs:
            fig,axs = plt.subplots(1,figsize=(cm_to_in(20),cm_to_in(20)))
        self.l5mdl.init_stim(amp=stim_amp,sweep_len = 2000,stim_start = 300, stim_dur = 1000)
        if rec_extra:
            Vm, I, t, stim,extra_vms = self.l5mdl.run_model(dt=dt,rec_extra = rec_extra)
            self.extra_vms = extra_vms
        else:
            Vm, I, t, stim = self.l5mdl.run_model(dt=dt)
            
        self.volt_soma = Vm
        self.I = I
        self.t = t
        self.stim = stim
        
        axs.plot(t,Vm, label='Vm', color=clr,linewidth=1)
        axs.locator_params(axis='x', nbins=5)
        axs.locator_params(axis='y', nbins=8)
        #plt.show()
        #add_scalebar(axs)
        plt.savefig("PT_0.5.png")
        #file_path_to_save=f'{self.plot_folder}{plot_fn}.pdf'
        #plt.savefig(file_path_to_save, format='pdf')
        return axs

    def plot_currents(self,stim_amp = 0.3,dt = 0.01,clr = 'black',plot_fn = 'step',axs = None):
        if not axs:
            fig,axs = plt.subplots(4,figsize=(cm_to_in(8),cm_to_in(30)))
        self.l5mdl.init_stim(amp=stim_amp,sweep_len = 200)
        Vm, I, t, stim = self.l5mdl.run_model(dt=dt)
        axs[0].plot(t,Vm, label='Vm', color=clr,linewidth=1)
        axs[0].locator_params(axis='x', nbins=5)
        axs[0].locator_params(axis='y', nbins=8)
        
        axs[1].plot(t,I['Na'],label = 'Na',color = 'red')
        axs[2].plot(t,I['K'],label = 'K',color = 'black')
        axs[3].plot(t,I['Ca'],label = 'Ca',color = 'green')
        #add_scalebar(axs)
        file_path_to_save=f'{self.plot_folder}Ktrials2_{plot_fn}.pdf'
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=my_dpi)
        return axs
    def get_axonal_ks(self, start_Vm = -72, dt= 0.1,rec_extra = False):
        h.dt=dt
        self.dt = dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop/h.dt)

        Vm = np.zeros(timesteps)
        I = {}
        I['Na'] = np.zeros(timesteps)
        I['K'] = np.zeros(timesteps)
        I['K31'] = np.zeros(timesteps)
        I['KT'] = np.zeros(timesteps)
        I['KCa'] = np.zeros(timesteps)
        I['KP'] = np.zeros(timesteps)
        stim = np.zeros(timesteps)
        t = np.zeros(timesteps)

        for i in range(timesteps):
            Vm[i] = h.cell.soma[0].v
            I['Na'][i] = self.l5mdl.ais(0.5).ina
            I['K'][i] = self.l5mdl.ais(0.5).ik
            I['K31'][i] = self.l5mdl.ais.gSKv3_1_SKv3_1
            I['KP'][i] = self.l5mdl.ais.gK_Pst_K_Pst
            I['KT'][i] = self.l5mdl.ais.gK_Tst_K_Tst
            I['KCa'][i] = self.l5mdl.ais.gSK_E2_SK_E2
            t[i] = i*h.dt / 1000
            stim[i] = h.st.amp
            h.fadvance()
        return Vm, I, t, stim


    def plot_axonal_ks(self,stim_amp = 0.5,dt = 0.01,clr = 'black',plot_fn = 'step_axon_ks',axs = None):
        if not axs:
            fig,axs = plt.subplots(7,2,figsize=(cm_to_in(16),cm_to_in(70)))
        self.l5mdl.init_stim(amp=stim_amp,sweep_len = 500)
        Vm, I, t, stim = self.get_axonal_ks(dt=dt)
        axs[0][0].plot(t,Vm, label='Vm', color=clr,linewidth=1)
        plot_dvdt_from_volts(Vm,self.dt,axs[0][1])
        axs[0][0].locator_params(axis='x', nbins=5)
        axs[0][0].locator_params(axis='y', nbins=8)
        
        axs[1][0].plot(t,I['Na'],label = 'Na',color = 'red')
        axs[1][0].legend()
        plot_dg_dt(I['Na'],Vm,self.dt,axs[1][1])
        axs[2][0].plot(t,I['K'],label = 'K',color = 'black')
        plot_dg_dt(I['K'],Vm,self.dt,axs[2][1])
        axs[2][0].legend()
        axs[3][0].plot(t,I['K31'],label = 'K31',color = 'green')
        plot_dg_dt(I['K31'],Vm,self.dt,axs[3][1])
        axs[3][0].legend()
        axs[4][0].plot(t,I['KP'],label = 'KP',color = 'orange')
        plot_dg_dt(I['KP'],Vm,self.dt,axs[4][1])
        axs[4][0].legend()
        axs[5][0].plot(t,I['KT'],label = 'KT',color = 'yellow')
        plot_dg_dt(I['KT'],Vm,self.dt,axs[5][1])
        axs[5][0].legend()
        axs[6][0].plot(t,I['KCa'],label = 'KCa',color = 'grey')
        plot_dg_dt(I['KCa'],Vm,self.dt,axs[6][1])
        axs[6][0].legend()
        



        #add_scalebar(axs)
        file_path_to_save=plot_fn
        plt.savefig(file_path_to_save, format='pdf', dpi=my_dpi)
        return axs
        
    def plot_fi_curve(self,start=0,end=0.6,nruns=14,wt_data = None,ax1 = None, fig = None,fn = 'ficurve'):
        fis = get_fi_curve(self.l5mdl,start,end,nruns,dt = 0.025,wt_data = wt_data,ax1=ax1,fig=fig,fn=f'{self.plot_folder}{fn}.pdf')
        return fis
    

    def plot_volts_dvdt(self,stim_amp = 0.3):
        fig_volts,axs_volts = plt.subplots(1,figsize=(cm_to_in(20),cm_to_in(20)))
        fig_dvdt,axs_dvdt = plt.subplots(1,figsize=(cm_to_in(20),cm_to_in(20)))
        self.plot_stim(axs = axs_volts,dt=0.02)
        plot_dvdt_from_volts(self.volt_soma,self.dt,axs_dvdt)
        file_path_to_save=f'{self.plot_folder}AnnaModel_volts_dvdt_{stim_amp}.pdf'
        fig_dvdt.savefig(file_path_to_save, format='pdf', dpi=my_dpi)
        file_path_to_save=f'{self.plot_folder}AnnaModel_volts_{stim_amp}.pdf'
        fig_volts.savefig(file_path_to_save, format='pdf', dpi=my_dpi)


    def get_ap_init_site(self):
        self.plot_stim(stim_amp = 0.7,rec_extra=True)
        soma_spikes = get_spike_times(self.volt_soma,self.t)
        axon_spikes = get_spike_times(self.extra_vms['axon'],self.t)
        ais_spikes = get_spike_times(self.extra_vms['ais'],self.t)
        for i in range(len(soma_spikes)):
            print(f'spike #{i} soma - {soma_spikes[i]}, ais - {ais_spikes[i]}, axon - {axon_spikes[i]}')
    
    

    #________________Original plot_model_FI_Vs_dvdt___________________________________________
    # def plot_model_FI_Vs_dvdt(self,vs_amp,fnpre = '',wt_fi = None, start=0,end=2,nruns=30):
    #     print('plot_model_FI_Vs_dvdt' + fnpre)
    #     #wt_fi = [0, 0, 0, 0, 3, 5, 7, 9, 10, 12, 13]
    #     for curr_amp in vs_amp:
    #         fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(3),cm_to_in(3.5)))
    #         axs[0] = self.plot_stim(axs = axs[0],stim_amp = curr_amp,dt=0.025)
    #         #axs[0] = self.plot_stim(axs = axs[0],stim_amp = curr_amp,dt=0.005)
    #         axs[1] = plot_dvdt_from_volts(self.volt_soma,self.dt,axs[1])
    #         add_scalebar(axs[0])
    #         add_scalebar(axs[1])
    #         fn = f'{self.plot_folder}/{fnpre}dvdt_vs_{curr_amp}.pdf'
    #         fig_volts.savefig(fn)
    #     fi_ans = self.plot_fi_curve(start,end,nruns,wt_data = wt_fi,fn = fnpre + '_fi')
    #     with open(f'{self.plot_folder}/{fnpre}.csv', 'w+', newline='') as myfile:
    #         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #         wr.writerow(fi_ans)
    #     return fi_ans
    #_________________________________________________________________________________________

    ##############################################################################################
    def plot_model_FI_Vs_dvdt(self,vs_amp,fnpre = '',wt_fi = None, start=0,end=2,nruns=21):
        
        
        #wt_fi = [0, 0, 0, 1, 5, 8, 11, 13, 15, 16, 17, 19, 20, 21, 22, 22, 23, 24, 25, 25, 26] #20%WT

        for curr_amp in vs_amp:
            fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(20),cm_to_in(20)))
            self.plot_stim(axs = axs[0],stim_amp = curr_amp,dt=0.01,plot_fn = f'{fnpre}_step')
            plot_dvdt_from_volts(self.volt_soma,self.dt,axs[1])
            fn2 = f'{self.plot_folder}/{fnpre}_{curr_amp}.pdf'
            fig_volts.savefig(fn2)
            
        
        self.plot_fi_curve_2line(start,end,nruns, wt_data=wt_fi, fn = fnpre + '_fi')
        
    ##_________________________________________________________________________________________________
    def plot_fi_curve_2line(self,start,end,nruns,wt_data=None,ax1 = None, fig = None,fn = 'ficurve'): #start=0,end=0.6,nruns=14 (change wt_data from None to add WT line)
        fis = get_fi_curve(self.l5mdl,start,end,nruns,dt = 0.1,wt_data = wt_data,ax1=ax1,fig=fig,fn=f'{self.plot_folder}{fn}.pdf')
        return fis
    
    
    
    ##############################################################################################
    
def default_model():
    sim = Na1612Model_TF()
    #sim.plot_currents()
    fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(20),cm_to_in(20)))
    sim.plot_stim(axs = axs[0], stim_amp = 0.5,dt=0.025)
    plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
    fn = f'{sim.plot_folder}/default_na12HMM.pdf'
    fig_volts.savefig(fn)

def make_currentscape_plot(sim_config = {
                'section' : 'soma',
                'segment' : 0.5,
                'section_num': 0,
                'currents'  : ['na12.ina_ina','na12mut.ina_ina','na16.ina_ina','na16mut.ina_ina','ica_Ca_HVA','ica_Ca_LVAst','ihcn_Ih','ik_SK_E2','ik_SKv3_1'],
                'ionic_concentrations' :["cai", "ki", "nai"]
                
            }):
    sim_obj = NeuronModel()   #TO DO : send in different parameters???
    sim_obj.init_stim(amp=0.5,sweep_len = 200)
    Vm, I, t, stim,ionic = sim_obj.run_sim_model(dt=0.01,sim_config=sim_config)
    current_names = sim_config['currents']
    plot_config = {
        "output": {
            "savefig": True,
            "dir": "./Plots/Currentscape/",
            "fname": "test_plot2",
            "extension": "pdf",
            "dpi": 600,
            "transparent": False
        },
        "current": {"names": current_names},
        "ions":{"names": ["ca", "k", "na"]},
        "voltage": {"ylim": [-90, 50]},
        "legendtextsize": 5,
        "adjust": {
            "left": 0.15,
            "right": 0.8,
            "top": 1.0,
            "bottom": 0.0
            }
        }
    fig = plot_currentscape(Vm, [I[x] for x in I.keys()], plot_config,[ionic[x] for x in ionic.keys()])


def scanK():
    for i in np.arange(5,15,1):
        
        sim = Na1612Model_TF(ais_ca=i)
        print(f'ais_ca={i}')
        #sim.make_wt()
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(10),cm_to_in(15)))
        sim.plot_stim(axs = axs[0],stim_amp = 1,dt=0.025) #stim_amp = 0.6
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/ais_CA_{i}_.pdf'
        fig_volts.savefig(fn)
        
        sim = Na1612Model_TF(ais_Kca=i)
        print(f'ais_Kca={i}')
        #sim.make_wt()
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(10),cm_to_in(15)))
        sim.plot_stim(axs = axs[0],stim_amp = 1,dt=0.025) #stim_amp = 0.6
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/ais_Kca_{i}_.pdf'
        fig_volts.savefig(fn)
        
        sim = Na1612Model_TF(K=i)
        print(f'K={i}')
        #sim.make_wt()
        
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(9.5),cm_to_in(15)))
        stim_amp = 1 #stim_amp = 0.6
        sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/K_{i}_amp_{stim_amp}.pdf'
        fig_volts.savefig(fn)
        # stim_amp = 0.3 #stim_amp = 0.6
        # sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        # plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        # fn = f'{sim.plot_folder}/K_{i}_amp_{stim_amp}.pdf'
        # fig_volts.savefig(fn)
        
        sim = Na1612Model_TF(somaK=i)
        print(f'somaK={i}')
        #sim.make_wt()
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(9.5),cm_to_in(15)))
        stim_amp = 1 #stim_amp = 0.6
        sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/somaK_{i}_amp_{stim_amp}.pdf'
        fig_volts.savefig(fn)
        # stim_amp = 0.3
        # sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        # plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        # fn = f'{sim.plot_folder}/somaK_{i}_amp_{stim_amp}.pdf'
        # fig_volts.savefig(fn)


        sim = Na1612Model_TF(KP=i)
        print(f'KP={i}')
        #sim.make_wt()
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(9),cm_to_in(15)))
        stim_amp = 1 #stim_amp = 0.6
        sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/Kp_{i}_amp_{stim_amp}.pdf'
        fig_volts.savefig(fn)
        # stim_amp = 0.3
        # sim.plot_stim(axs = axs[0],stim_amp = stim_amp,dt=0.025)
        # plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        # fn = f'{sim.plot_folder}/Kp_{i}_amp_{stim_amp}.pdf'
        # fig_volts.savefig(fn)        
        
        sim = Na1612Model_TF(KT=i)
        print(f'KT={i}')
        #sim.make_wt()
        fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(10),cm_to_in(15)))
        sim.plot_stim(axs = axs[0],stim_amp = 1,dt=0.025) #stim_amp = 0.6
        plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
        fn = f'{sim.plot_folder}/Kt_{i}_.pdf'
        fig_volts.savefig(fn)

def scan12_16():
    for i12 in np.arange(.5,1.2,.1): #(.4,2,.5)
        for i16 in np.arange(.5,1.2,.1):
            print(f'nav12={i12}, nav16={i16}')
            sim = Na1612Model_TF(nav12=i12, nav16=i16)
            #sim.make_wt()
            fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
            sim.plot_stim(axs = axs[0],stim_amp = 1,dt=0.025)
            plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
            fn = f'{sim.plot_folder}/12_{i12}_16_{i16}.pdf'
            fig_volts.savefig(fn)

def scan_soma_12_16():
    for i12 in np.arange(.5,5,1): #(.4,2,.5)
        for i16 in np.arange(.5,5,1):
            print(f'soma_nav12={i12}, soma_nav16={i16}')
            sim = Na1612Model_TF(soma_na12=i12, soma_na16=i16)
            #sim.make_wt()
            fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
            sim.plot_stim(axs = axs[0],stim_amp = 1,dt=0.025)
            plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
            fn = f'{sim.plot_folder}/12_{i12}_16_{i16}.pdf'
            fig_volts.savefig(fn)


def scan_soma_12_16_point5nA():
    for i12 in np.arange(.5,5,1): #(.4,2,.5)
        for i16 in np.arange(.5,5,1):
            print(f'soma_nav12={i12}, soma_nav16={i16}')
            sim = Na1612Model_TF(soma_na12=i12, soma_na16=i16)
            #sim.make_wt()
            fig_volts,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(15)))
            sim.plot_stim(axs = axs[0],stim_amp = 0.5,dt=0.025)
            plot_dvdt_from_volts(sim.volt_soma,sim.dt,axs[1])
            fn = f'{sim.plot_folder}/12_{i12}_16_{i16}.pdf'
            fig_volts.savefig(fn)



#make_currentscape_plot()
"""
Main
"""


default_model()
# #scanK()
# #scan12_16()
#sim = Na1612Model_TF()
#sim.plot_model_FI_Vs_dvdt([0.3,0.4,0.5,0.6],fnpre='m1v1')
# #python3 TTPC_M1_Na_HH.py