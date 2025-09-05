from optim import mse, huber, run_nsga3_optimization_with_init_values2, run_nsga3_optimization_with_init_values2_min_max_logging
from netpyne import specs, sim
import json
import os
import numpy as np
import statistics
import sys
from async_stats import get_central_spike_times, compute_plp, compute_plp2

def megaTargetFunc(t_matrix, 
                    C_PSDC, C_Proj, C_Inh, 
                    k_PSDC, k_Proj, k_Inh,
                    vr_PSDC, vr_Proj, vr_Inh,
                    vt_PSDC, vt_Proj, vt_Inh,
                    a_PSDC, a_Proj, a_Inh,
                    b_PSDC, b_Proj, b_Inh,
                    c_PSDC, c_Proj, c_Inh,
                    d_PSDC, d_Proj, d_Inh,
                    celltype_PSDC, celltype_Proj, celltype_Inh,
                    ItoE_weight, InterPSDC_weight, PANtoE_weight, PANtoI_weight,
                    tauExcLoop, tauExcProj, tauExcInh, tauExcPSDC, tauPSDCtoProj, tauPSDCtoInh, tauInh,
                    tau_recExcLoop, tau_recExcProj, tau_recExcInh, tau_recExcPSDC, tau_recPSDCtoProj, tau_recPSDCtoInh, tau_rec_Inh,
                    UExcLoop, UExcProj, UExcInh, UExcPSDC, UPSDCtoProj, UPSDCtoInh, UInh):
    netParams = specs.NetParams() 
    netParams.ItoE_weight = ItoE_weight
    netParams.InterPSDC_weight = InterPSDC_weight
    netParams.PANtoE_weight = PANtoE_weight
    netParams.PANtoI_weight = PANtoI_weight

    IzhiPSDC = {'secs': {}}
    IzhiPSDC['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiPSDC['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiPSDC['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': abs(C_PSDC),
        'k': abs(k_PSDC),
        'vr': abs(vr_PSDC),
        'vt': abs(vt_PSDC),
        'vpeak': 20,
        'a': abs(a_PSDC),
        'b': abs(b_PSDC),
        'c': abs(c_PSDC),
        'd': abs(d_PSDC),
        'celltype': int(round(celltype_PSDC))
    } 

    IzhiProj = {'secs': {}}
    IzhiProj['secs']['soma'] = {'geom': {}, 'pointps': {}}                        
    IzhiProj['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiProj['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': abs(C_Proj),
        'k': abs(k_Proj),
        'vr': abs(vr_Proj),
        'vt': abs(vt_Proj),
        'vpeak': 20,
        'a': abs(a_Proj),
        'b': abs(b_Proj),
        'c': abs(c_Proj),
        'd': abs(d_Proj),
        'celltype': int(round(celltype_Proj))
    }

    IzhiInh = {'secs': {}}
    IzhiInh['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiInh ['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiInh['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': abs(C_Inh),
        'k': abs(k_Inh),
        'vr': abs(vr_Inh),
        'vt': abs(vt_Inh),
        'vpeak': 20,
        'a': abs(a_Inh),
        'b': abs(b_Inh),
        'c': abs(c_Inh),
        'd': abs(d_Inh),
        'celltype': int(round(celltype_Inh))
    } 

    netParams.cellParams['IzhiPSDC'] = IzhiPSDC 
    netParams.cellParams['IzhiProj'] = IzhiProj
    netParams.cellParams['IzhiInh'] = IzhiInh

    levels = 2 #Если > 2 - то слоев PSDC слоев больше
    numNeurons = 11; #10 проекционных клеток и 10 ингибиторов. Выбрано для ускорения оптимизации
    #Ранее замечено, что если применять scale=1/10 фактор к синаптическим весам при 100 нейронах, то поведение будет аналогично, что и при 10.
    #Сравнивал времена спайков при фиксированых delay - практически идеальные совпадения.
    numRightWay = int(numNeurons * 0.6) #Путей идущих на прямую в к проэкционным нейронам 60% (взято из Nature статьи)
    numLongWay = numNeurons - numRightWay;

    #Маски чтобы сделать более равномерное распределение PSDC путей и прямых
    step = (numNeurons - 1) / (numLongWay - 1)
    maskLongWays = np.arange(1, numNeurons, step).astype(int)
    maskRightWays = np.arange(1, numNeurons, 1)
    maskRightWays = np.setdiff1d(maskRightWays, maskLongWays)

    #Создание PSDC слоев
    for l in range(1, levels):
        for i in range(1, numLongWay):
            netParams.popParams[f'EInterCell1C{i}'] = {'cellType': 'IzhiPSDC', 'numCells': 1}

    for i in range(1, numNeurons):
        netParams.popParams[f'ECell{i}'] = {'cellType': 'IzhiProj', 'numCells': 1}
        netParams.popParams[f'ICell{i}'] = {'cellType': 'IzhiInh', 'numCells': 1}

    ## Tsodyks-Markram Pressets ::::::::::::::::::::::::::::::::::::::::::::::::::::::
    netParams.synMechParams['ExcLoopTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcLoop),          
        'tau_rec': abs(tau_recExcLoop),    
        'tau_facil': 100.0,    
        'U': abs(UExcLoop),          
        'u0': 0.0,         
        'e': 0.0            
    }

    netParams.synMechParams['ExcProjTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcProj), #спад после спайка
        'tau_rec': abs(tau_recExcProj), #восстановление после депрессии
        'tau_facil': 100.0, #фасцилитация
        'U': abs(UExcProj), #высокий уровень высвобождения
        'u0': 0.0, #начальное значение u
        'e': 0.0             
    }

    netParams.synMechParams['ExcInhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcInh),          
        'tau_rec': abs(tau_recExcInh),   
        'tau_facil': 100.0,    
        'U': abs(UExcInh),           
        'u0': 0.0,         
        'e': 0.0             
    }

    netParams.synMechParams['ExcPSDCTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcPSDC),          
        'tau_rec': abs(tau_recExcPSDC),    
        'tau_facil': 100.0,    
        'U': abs(UExcPSDC),          
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCtoProjTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoProj),        
        'tau_rec': abs(tau_recPSDCtoProj),   
        'tau_facil': 100.0,    
        'U': abs(UPSDCtoProj),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCtoInhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoInh),        
        'tau_rec': abs(tau_recPSDCtoInh),   
        'tau_facil': 100.0,    
        'U': abs(UPSDCtoInh),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['InhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauInh),           
        'tau_rec': abs(tau_rec_Inh),   
        'tau_facil': 100.0,    
        'U': abs(UInh),           
        'u0': 0.0,         
        'e': -70             
    }
    ## Tsodyks-Markram Pressets ::::::::::::::::::::::::::::::::::::::::::::::::::::::

    typePSDCExc = 'ExcPSDCTMG'
    typePSDCtoInh = 'PSDCtoInhTMG'
    typePSDCtoProj = 'PSDCtoProjTMG'
    typeProjExcSyn = 'ExcProjTMG'
    typeInhExcSyn = 'ExcInhTMG'
    typeInhSyn = 'InhTMG'
    ## Synaptic mechanism parameters ------------------------------------------ 
    
    #Расчет задержен для нейронов с отклонением в 10%
    delta = 0.1
    #PSDC-output delays
    def_delay_for_PSDC_output = 2
    lower_bound = def_delay_for_PSDC_output - def_delay_for_PSDC_output*delta
    upper_bound = def_delay_for_PSDC_output + def_delay_for_PSDC_output*delta
    delay_for_PSDC_output = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    def_delay_for_PSDC_inh = 2
    lower_bound = def_delay_for_PSDC_inh - def_delay_for_PSDC_inh*delta
    upper_bound = def_delay_for_PSDC_inh + def_delay_for_PSDC_inh*delta
    delay_for_PSDC_inh = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    #Inhs_Right-output delays
    def_delay_for_InhsRight_output = 2.0
    lower_bound = def_delay_for_InhsRight_output - def_delay_for_InhsRight_output*delta
    upper_bound = def_delay_for_InhsRight_output + def_delay_for_InhsRight_output*delta
    delay_for_InhsRight_output = np.random.uniform(lower_bound, upper_bound, numRightWay*numNeurons)

    #Inhs_Right-output delays
    def_delay_for_InhsLong_output = 3.0
    lower_bound = def_delay_for_InhsLong_output - def_delay_for_InhsLong_output*delta
    upper_bound = def_delay_for_InhsLong_output + def_delay_for_InhsLong_output*delta
    delay_for_InhsLong_output = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    #Шаблон для системы с PSDC путями
    connParamsAllPathActive = {}
    ## Connections parameters -------------------------------------------------
    # PSDC Levels=========================================
    # Input
    for i in range(1, numLongWay):
        connParamsAllPathActive[f'STIMLW->EInterCell1C{i}'] = {
            'preConds': {'pop': 'stimLW'}, 
            'postConds': {'pop': f'EInterCell1C{i}'},  
            'probability': 1,         
            'weight': netParams.PANtoE_weight,           
            'delay': 0,                 
            'sec': 'dend',              
            'loc': 1.0,                 
            'synMech': typePSDCExc,           
    }
            
    # Inter - Output
    for i in range(1, numLongWay): 
        d = 0
        for j in range(1, numNeurons):
            connParamsAllPathActive[f'EInterCell{levels - 1}C{i}->ECell{j}'] = {
                'preConds': {'pop': f'EInterCell{levels - 1}C{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                'weight': netParams.InterPSDC_weight,           
                'delay': delay_for_PSDC_output[d],                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typePSDCtoProj,           
            }
            d += 1
        d = 0
        for j in maskLongWays:
            connParamsAllPathActive[f'EInterCell{levels - 1}C{i}->ICell{j}'] = {
                'preConds': {'pop': f'EInterCell{levels - 1}C{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': netParams.InterPSDC_weight,           
                'delay': delay_for_PSDC_inh[d],                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typePSDCtoInh,           
            }
            d += 1
    # PSDC Levels=========================================  

    # Fully connected Levels==============================
    for i in range(1, numNeurons):
        connParamsAllPathActive[f'STIMRW->ECell{i}'] = {
                'preConds': {'pop': 'stimRW'}, 
                'postConds': {'pop': f'ECell{i}'},  
                'probability': 1,         
                'weight': netParams.PANtoE_weight,           
                'delay': 0,                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeProjExcSyn,           
            }

        if i < numNeurons - 1:
            connParamsAllPathActive[f'ECell{i}->ECell{i + 1}'] = {
                    'preConds': {'pop': f'ECell{i}'}, 
                    'postConds': {'pop': f'ECell{i + 1}'},  
                    'probability': 1,         
                    'weight': netParams.PANtoE_weight * 0.85,           
                    'delay': 3,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ExcLoopTMG',           
                }

            connParamsAllPathActive[f'ECell{i+1}->ECell{i}'] = {
                    'preConds': {'pop': f'ECell{i + 1}'}, 
                    'postConds': {'pop': f'ECell{i}'},  
                    'probability': 1,         
                    'weight': netParams.PANtoE_weight * 0.85,           
                    'delay': 3,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ExcLoopTMG'           
                }
               
    for i in maskRightWays:
        connParamsAllPathActive[f'STIMRW->ICell{i}'] = {
            'preConds': {'pop': 'stimRW'}, 
            'postConds': {'pop': f'ICell{i}'},  
            'probability': 1,         
            'weight': netParams.PANtoE_weight,           
            'delay': 0,                 
            'sec': 'dend',              
            'loc': 1.0,                 
            'synMech': typeInhExcSyn,           
        }
    # Fully connected Levels==============================
            
    #EXC
    for i in range(1, numNeurons):
        #INH
        d = 0
        for j in maskLongWays:
            connParamsAllPathActive[f'ICell4ms{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': netParams.ItoE_weight,   
                'delay': delay_for_InhsLong_output[d],                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeInhSyn,
            }
            d += 1    

        #INH
        d = 0
        for j in maskRightWays:
            connParamsAllPathActive[f'ICell2ms{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': netParams.ItoE_weight,   
                'delay': delay_for_InhsRight_output[d],                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeInhSyn,
            }   
            d += 1  
    ## Connections parameters -------------------------------------------------
    
    #Шаблон без PSDC
    connParamsPSDCoff = connParamsAllPathActive.copy()

    #Удаление PSDC слоев
    for i in range(1, numLongWay):
        del connParamsPSDCoff[f'STIMLW->EInterCell1C{i}']
    for i in range(1, numLongWay): 
        for j in range(1, numNeurons):
            del connParamsPSDCoff[f'EInterCell{levels - 1}C{i}->ECell{j}']
        for j in maskLongWays:
            del connParamsPSDCoff[f'EInterCell{levels - 1}C{i}->ICell{j}']

    #Симуляция с PSDC слоями
    ys_SPC_all= [] #Spike per cycle
    ys_freq_all= [] #Freq output
    ys_PLP= [] #Phase Locking Probability
    netParams.connParams = connParamsAllPathActive
    for f in t_matrix[0]:
        with open(f'inputs/{f}HzPattern.json', 'r') as json_file:
            spikes = json.load(json_file)
        netParams.popParams['stimLW'] = {'cellModel': 'VecStim', 'numCells': len(maskLongWays), 'spkTimes': spikes} 
        netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

        simConfig = specs.SimConfig()      
        simConfig.duration = 1000         
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        output_spkt_times = get_central_spike_times(sim, 'ECell', (0, simConfig.duration))
        allPopRates = sim.allSimData['popRates']
        freqs = []
        for l in range(1, numNeurons):
            freqs.append(allPopRates[f'ECell{l}'])
        output_freq = statistics.mean(freqs)
        ys_freq_all.append(output_freq) 
        if f > 100:
            PLProb = compute_plp2(output_spkt_times, spikes[4])
            ys_PLP.append(PLProb)

            y = output_freq / f
            ys_SPC_all.append(y)
    
    #Симуляция с отключенными PSDC слоями
    ys_SPC_offPSDC= [] #Spike per cycle
    ys_freq_offPSDC= [] #Freq output
    netParams.connParams = connParamsPSDCoff
    for f in t_matrix[1]:
        with open(f'inputs/{f}HzPattern.json', 'r') as json_file:
            spikes = json.load(json_file)
        netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

        simConfig = specs.SimConfig()      
        simConfig.duration = 1000         
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        output_spkt_times = get_central_spike_times(sim, 'ECell', (0, simConfig.duration))
        allPopRates = sim.allSimData['popRates']
        freqs = []
        for l in range(1, numNeurons):
            freqs.append(allPopRates[f'ECell{l}'])
        output_freq = statistics.mean(freqs)
        ys_freq_offPSDC.append(output_freq) 
        if f > 100:
            y = output_freq / f
            ys_SPC_offPSDC.append(y)
    
    PSDCisOff = [False, True]
    freqResponse = {}
    for PSDCstatus in PSDCisOff:
        freqResponse_temp = []
        for f in t_matrix[5]:
            with open(f'inputs/{f}mN.json', 'r') as json_file:
                spikes = json.load(json_file)

            if (PSDCstatus):
                netParams.popParams['stimLW'] = {'cellModel': 'VecStim', 'numCells': len(maskLongWays), 'spkTimes': spikes}
            netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

            if (PSDCstatus):
                netParams.connParams = connParamsAllPathActive
            else:
                netParams.connParams = connParamsPSDCoff

            simConfig = specs.SimConfig()      
            simConfig.duration = 600         
            simConfig.dt = 0.025 
            simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

            sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

            allPopRates = sim.allSimData['popRates']
            freqs = []
            for l in range(1, numNeurons):
                freqs.append(allPopRates[f'ECell{l}'])

            freqResponse_temp.append(statistics.mean(freqs))

        freqResponse[PSDCstatus] = np.array(freqResponse_temp)
    ys_pressure = freqResponse[True] / freqResponse[False]

    y_final = [
        ys_SPC_all, ys_SPC_offPSDC, 
        ys_PLP, 
        ys_freq_all, ys_freq_offPSDC, 
        ys_pressure.tolist()
    ]

    return  y_final


def megaTargetFuncOnlySyn(t_matrix, 
                    ItoE_weight, InterPSDC_weight, PANtoE_weight, PANtoI_weight,
                    tauExcLoop, tauExcProj, tauExcInh, tauExcPSDC, tauPSDCtoProj, tauPSDCtoInh, tauInh,
                    tau_recExcLoop, tau_recExcProj, tau_recExcInh, tau_recExcPSDC, tau_recPSDCtoProj, tau_recPSDCtoInh, tau_rec_Inh,
                    UExcLoop, UExcProj, UExcInh, UExcPSDC, UPSDCtoProj, UPSDCtoInh, UInh,
                    tau_facilExcLoop, tau_facilExcProj, tau_facilExcInh, tau_facilExcPSDC, tau_facilPSDCtoProj, tau_facilPSDCtoInh, tau_facilInh):
    netParams = specs.NetParams() 
    netParams.ItoE_weight = ItoE_weight
    netParams.InterPSDC_weight = InterPSDC_weight
    netParams.PANtoE_weight = PANtoE_weight
    netParams.PANtoI_weight = PANtoI_weight

    IzhiPSDC = {'secs': {}}
    IzhiPSDC['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiPSDC['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiPSDC['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 5,
        'k': 1.5,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.1,
        'b': 0.26,
        'c': -65,
        'd': 2,
        'celltype': 2
    } 

    IzhiProj = {'secs': {}}
    IzhiProj['secs']['soma'] = {'geom': {}, 'pointps': {}}                        
    IzhiProj['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiProj['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 2,
        'k': 2.0,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.01,
        'b': 4,
        'c': -42,
        'd': 0.6,
        'celltype': 3
    }

    IzhiInh = {'secs': {}}
    IzhiInh['secs']['soma'] = {'geom': {}, 'pointps': {}}                       
    IzhiInh ['secs']['soma']['geom'] = {'diam': 10, 'L': 100, 'Ra': 100.0}
    IzhiInh['secs']['soma']['pointps']['Izhi'] = {
        'mod': 'Izhi2007b',
        'C': 5,
        'k': 1.5,
        'vr': -60,
        'vt': -45,
        'vpeak': 25,
        'a': 0.1,
        'b': 0.26,
        'c': -65,
        'd': 2,
        'celltype': 2
    } 

    netParams.cellParams['IzhiPSDC'] = IzhiPSDC 
    netParams.cellParams['IzhiProj'] = IzhiProj
    netParams.cellParams['IzhiInh'] = IzhiInh

    levels = 2 #Если > 2 - то слоев PSDC слоев больше
    numNeurons = 11; #10 проекционных клеток и 10 ингибиторов. Выбрано для ускорения оптимизации
    #Ранее замечено, что если применять scale=1/10 фактор к синаптическим весам при 100 нейронах, то поведение будет аналогично, что и при 10.
    #Сравнивал времена спайков при фиксированых delay - практически идеальные совпадения.
    numRightWay = int(numNeurons * 0.6) #Путей идущих на прямую в к проэкционным нейронам 60% (взято из Nature статьи)
    numLongWay = numNeurons - numRightWay;

    #Маски чтобы сделать более равномерное распределение PSDC путей и прямых
    step = (numNeurons - 1) / (numLongWay - 1)
    maskLongWays = np.arange(1, numNeurons, step).astype(int)
    maskRightWays = np.arange(1, numNeurons, 1)
    maskRightWays = np.setdiff1d(maskRightWays, maskLongWays)

    #Создание PSDC слоев
    for l in range(1, levels):
        for i in range(1, numLongWay):
            netParams.popParams[f'EInterCell1C{i}'] = {'cellType': 'IzhiPSDC', 'numCells': 1}

    for i in range(1, numNeurons):
        netParams.popParams[f'ECell{i}'] = {'cellType': 'IzhiProj', 'numCells': 1}
        netParams.popParams[f'ICell{i}'] = {'cellType': 'IzhiInh', 'numCells': 1}

    ## Tsodyks-Markram Pressets ::::::::::::::::::::::::::::::::::::::::::::::::::::::
    netParams.synMechParams['ExcLoopTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcLoop),          
        'tau_rec': abs(tau_recExcLoop),    
        'tau_facil': abs(tau_facilExcLoop),    
        'U': abs(UExcLoop),          
        'u0': 0.0,         
        'e': 0.0            
    }

    netParams.synMechParams['ExcProjTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcProj),           #спад после спайка
        'tau_rec': abs(tau_recExcProj),    #восстановление после депрессии
        'tau_facil': abs(tau_facilExcProj),      
        'U': abs(UExcProj),           #высокий уровень высвобождения
        'u0': 0.0,          #начальное значение u
        'e': 0.0             
    }

    netParams.synMechParams['ExcInhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcInh),           #спад после спайка
        'tau_rec': abs(tau_recExcInh),    #восстановление после депрессии
        'tau_facil': abs(tau_facilExcInh),      
        'U': abs(UExcInh),           #высокий уровень высвобождения
        'u0': 0.0,          #начальное значение u
        'e': 0.0             
    }

    netParams.synMechParams['ExcPSDCTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauExcPSDC),          
        'tau_rec': abs(tau_recExcPSDC),    
        'tau_facil': abs(tau_facilExcPSDC),
        'U': abs(UExcPSDC),          
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCtoProjTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoProj),        
        'tau_rec': abs(tau_recPSDCtoProj),   
        'tau_facil': abs(tau_facilPSDCtoProj),   
        'U': abs(UPSDCtoProj),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['PSDCtoInhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauPSDCtoInh),        
        'tau_rec': abs(tau_recPSDCtoInh),   
        'tau_facil': abs(tau_facilPSDCtoInh),   
        'U': abs(UPSDCtoInh),         
        'u0': 0.0,          
        'e': 0.0            
    }

    netParams.synMechParams['InhTMG'] = {
        'mod': 'tmgsyn',
        'tau': abs(tauInh),           #спад после спайка
        'tau_rec': abs(tau_rec_Inh),    #восстановление после депрессии
        'tau_facil': abs(tau_facilInh),  
        'U': abs(UInh),           #высокий уровень высвобождения
        'u0': 0.0,          #начальное значение u
        'e': -70             
    }
    ## Tsodyks-Markram Pressets ::::::::::::::::::::::::::::::::::::::::::::::::::::::

    typePSDCExc = 'ExcPSDCTMG'
    typePSDCtoInh = 'PSDCtoInhTMG'
    typePSDCtoProj = 'PSDCtoProjTMG'
    typeProjExcSyn = 'ExcProjTMG'
    typeInhExcSyn = 'ExcInhTMG'
    typeInhSyn = 'InhTMG'
    ## Synaptic mechanism parameters ------------------------------------------ 
    
    #Расчет задержен для нейронов с отклонением в 10%
    delta = 0.1
    #PSDC-output delays
    def_delay_for_PSDC_output = 2
    lower_bound = def_delay_for_PSDC_output - def_delay_for_PSDC_output*delta
    upper_bound = def_delay_for_PSDC_output + def_delay_for_PSDC_output*delta
    delay_for_PSDC_output = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    def_delay_for_PSDC_inh = 2
    lower_bound = def_delay_for_PSDC_inh - def_delay_for_PSDC_inh*delta
    upper_bound = def_delay_for_PSDC_inh + def_delay_for_PSDC_inh*delta
    delay_for_PSDC_inh = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    #Inhs_Right-output delays
    def_delay_for_InhsRight_output = 2.0
    lower_bound = def_delay_for_InhsRight_output - def_delay_for_InhsRight_output*delta
    upper_bound = def_delay_for_InhsRight_output + def_delay_for_InhsRight_output*delta
    delay_for_InhsRight_output = np.random.uniform(lower_bound, upper_bound, numRightWay*numNeurons)

    #Inhs_Right-output delays
    def_delay_for_InhsLong_output = 3.0
    lower_bound = def_delay_for_InhsLong_output - def_delay_for_InhsLong_output*delta
    upper_bound = def_delay_for_InhsLong_output + def_delay_for_InhsLong_output*delta
    delay_for_InhsLong_output = np.random.uniform(lower_bound, upper_bound, numLongWay*numNeurons)

    #Шаблон для системы с PSDC путями
    connParamsAllPathActive = {}
    ## Connections parameters -------------------------------------------------
    # PSDC Levels=========================================
    # Input
    for i in range(1, numLongWay):
        connParamsAllPathActive[f'STIMLW->EInterCell1C{i}'] = {
            'preConds': {'pop': 'stimLW'}, 
            'postConds': {'pop': f'EInterCell1C{i}'},  
            'probability': 1,         
            'weight': netParams.PANtoE_weight,           
            'delay': 0,                 
            'sec': 'dend',              
            'loc': 1.0,                 
            'synMech': typePSDCExc,           
    }
            
    # Inter - Output
    for i in range(1, numLongWay): 
        d = 0
        for j in range(1, numNeurons):
            connParamsAllPathActive[f'EInterCell{levels - 1}C{i}->ECell{j}'] = {
                'preConds': {'pop': f'EInterCell{levels - 1}C{i}'}, 
                'postConds': {'pop': f'ECell{j}'},  
                'probability': 1,         
                'weight': netParams.InterPSDC_weight,           
                'delay': delay_for_PSDC_output[d],                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typePSDCtoProj,           
            }
            d += 1
        d = 0
        for j in maskLongWays:
            connParamsAllPathActive[f'EInterCell{levels - 1}C{i}->ICell{j}'] = {
                'preConds': {'pop': f'EInterCell{levels - 1}C{i}'}, 
                'postConds': {'pop': f'ICell{j}'},  
                'probability': 1,         
                'weight': netParams.InterPSDC_weight,           
                'delay': delay_for_PSDC_inh[d],                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typePSDCtoInh,           
            }
            d += 1
    # PSDC Levels=========================================  

    # Fully connected Levels==============================
    for i in range(1, numNeurons):
        connParamsAllPathActive[f'STIMRW->ECell{i}'] = {
                'preConds': {'pop': 'stimRW'}, 
                'postConds': {'pop': f'ECell{i}'},  
                'probability': 1,         
                'weight': netParams.PANtoE_weight,           
                'delay': 0,                 
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeProjExcSyn,           
            }

        if i < numNeurons - 1:
            connParamsAllPathActive[f'ECell{i}->ECell{i + 1}'] = {
                    'preConds': {'pop': f'ECell{i}'}, 
                    'postConds': {'pop': f'ECell{i + 1}'},  
                    'probability': 1,         
                    'weight': netParams.PANtoE_weight * 0.85,           
                    'delay': 3,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ExcLoopTMG',           
                }

            connParamsAllPathActive[f'ECell{i+1}->ECell{i}'] = {
                    'preConds': {'pop': f'ECell{i + 1}'}, 
                    'postConds': {'pop': f'ECell{i}'},  
                    'probability': 1,         
                    'weight': netParams.PANtoE_weight * 0.85,           
                    'delay': 3,                 
                    'sec': 'dend',              
                    'loc': 1.0,                 
                    'synMech': 'ExcLoopTMG'           
                }
               
    for i in maskRightWays:
        connParamsAllPathActive[f'STIMRW->ICell{i}'] = {
            'preConds': {'pop': 'stimRW'}, 
            'postConds': {'pop': f'ICell{i}'},  
            'probability': 1,         
            'weight': netParams.PANtoE_weight,           
            'delay': 0,                 
            'sec': 'dend',              
            'loc': 1.0,                 
            'synMech': typeInhExcSyn,           
        }
    # Fully connected Levels==============================
            
    #EXC
    for i in range(1, numNeurons):
        #INH
        d = 0
        for j in maskLongWays:
            connParamsAllPathActive[f'ICell4ms{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': netParams.ItoE_weight,   
                'delay': delay_for_InhsLong_output[d],                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeInhSyn,
            }
            d += 1    

        #INH
        d = 0
        for j in maskRightWays:
            connParamsAllPathActive[f'ICell2ms{j}->ECell{i}'] = {
                'preConds': {'pop': f'ICell{j}'}, 
                'postConds': {'pop': f'ECell{i}'}, 
                'probability': 1,         
                'weight': netParams.ItoE_weight,   
                'delay': delay_for_InhsRight_output[d],                
                'sec': 'dend',              
                'loc': 1.0,                 
                'synMech': typeInhSyn,
            }   
            d += 1  
    ## Connections parameters -------------------------------------------------
    
    #Шаблон без PSDC
    connParamsPSDCoff = connParamsAllPathActive.copy()

    #Удаление PSDC слоев
    for i in range(1, numLongWay):
        del connParamsPSDCoff[f'STIMLW->EInterCell1C{i}']
    for i in range(1, numLongWay): 
        for j in range(1, numNeurons):
            del connParamsPSDCoff[f'EInterCell{levels - 1}C{i}->ECell{j}']
        for j in maskLongWays:
            del connParamsPSDCoff[f'EInterCell{levels - 1}C{i}->ICell{j}']

    #Симуляция с PSDC слоями
    ys_SPC_all= [] #Spike per cycle
    ys_freq_all= [] #Freq output
    ys_PLP= [] #Phase Locking Probability
    netParams.connParams = connParamsAllPathActive
    for f in t_matrix[3]:
        with open(f'inputs/{f}HzPattern.json', 'r') as json_file:
            spikes = json.load(json_file)
        netParams.popParams['stimLW'] = {'cellModel': 'VecStim', 'numCells': len(maskLongWays), 'spkTimes': spikes} 
        netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

        simConfig = specs.SimConfig()      
        simConfig.duration = 1000         
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        output_spkt_times = get_central_spike_times(sim, 'ECell', (0, simConfig.duration))
        allPopRates = sim.allSimData['popRates']
        freqs = []
        for l in range(1, numNeurons):
            freqs.append(allPopRates[f'ECell{l}'])
        output_freq = statistics.mean(freqs)
        ys_freq_all.append(output_freq) 
        if f > 100:
            #PLProb = compute_plp2(output_spkt_times, spikes[4])
            PLProb, _ = compute_plp(output_spkt_times, spikes[4])
            ys_PLP.append(PLProb)

            y = output_freq / f
            ys_SPC_all.append(y)
    
    #Симуляция с отключенными PSDC слоями
    ys_SPC_offPSDC= [] #Spike per cycle
    ys_freq_offPSDC= [] #Freq output
    netParams.connParams = connParamsPSDCoff
    for f in t_matrix[4]:
        with open(f'inputs/{f}HzPattern.json', 'r') as json_file:
            spikes = json.load(json_file)
        netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

        simConfig = specs.SimConfig()      
        simConfig.duration = 1000         
        simConfig.dt = 0.025 
        simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

        sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

        output_spkt_times = get_central_spike_times(sim, 'ECell', (0, simConfig.duration))
        allPopRates = sim.allSimData['popRates']
        freqs = []
        for l in range(1, numNeurons):
            freqs.append(allPopRates[f'ECell{l}'])
        output_freq = statistics.mean(freqs)
        ys_freq_offPSDC.append(output_freq) 
        if f > 100:
            y = output_freq / f
            ys_SPC_offPSDC.append(y)
    
    PSDCisOff = [False, True]
    freqResponse = {}
    for PSDCstatus in PSDCisOff:
        freqResponse_temp = []
        for f in t_matrix[5]:
            with open(f'inputs/{f}mN.json', 'r') as json_file:
                spikes = json.load(json_file)

            if (PSDCstatus):
                netParams.popParams['stimLW'] = {'cellModel': 'VecStim', 'numCells': len(maskLongWays), 'spkTimes': spikes}
            netParams.popParams['stimRW'] = {'cellModel': 'VecStim', 'numCells': len(maskRightWays), 'spkTimes': spikes}

            if (PSDCstatus):
                netParams.connParams = connParamsAllPathActive
            else:
                netParams.connParams = connParamsPSDCoff

            simConfig = specs.SimConfig()      
            simConfig.duration = 600         
            simConfig.dt = 0.025 
            simConfig.printPopAvgRates = [ 0, simConfig.duration ]  

            sim.createSimulateAnalyze(netParams = netParams, simConfig = simConfig)

            allPopRates = sim.allSimData['popRates']
            freqs = []
            for l in range(1, numNeurons):
                freqs.append(allPopRates[f'ECell{l}'])

            freqResponse_temp.append(statistics.mean(freqs))

        freqResponse[PSDCstatus] = np.array(freqResponse_temp)
    ys_pressure = freqResponse[True] / freqResponse[False]

    y_final = [
        ys_SPC_all, ys_SPC_offPSDC, 
        ys_PLP, 
        ys_freq_all, ys_freq_offPSDC, 
        ys_pressure.tolist()
    ]

    return  y_final


#True values targer funcs------------------------------------------------------------------------
y_true_matrix = []
t_matrix = []

with open('characteristics/freq_sps_brainstem.json', 'r') as json_file:
    y_true_freq_stim_SPC_outputAllActive = json.load(json_file)
    y_values = list(y_true_freq_stim_SPC_outputAllActive['y'])
    x_values = list(y_true_freq_stim_SPC_outputAllActive['x'])
    y_true_matrix.append(y_values)
    t_matrix.append(x_values)
print(f'y_true_freq_stim_SPC_output={y_true_freq_stim_SPC_outputAllActive}')

y_true_matrix.append(y_values)
t_matrix.append(x_values)

with open('characteristics/freq_PLP_brainstem.json', 'r') as json_file:
    y_true_freq_stim_PLP_outputAllActive = json.load(json_file)
    y_values = list(y_true_freq_stim_PLP_outputAllActive['y'])
    x_values = list(y_true_freq_stim_PLP_outputAllActive['x'])
    y_true_matrix.append(y_values)
    t_matrix.append(x_values)

with open('characteristics/freq_freq_brainstem.json', 'r') as json_file:
    y_true_freq_freq_outputAllActive = json.load(json_file)
    y_values = list(y_true_freq_freq_outputAllActive['y'])
    x_values = list(y_true_freq_freq_outputAllActive['x'])
    y_true_matrix.append(y_values)
    t_matrix.append(x_values)
y_true_matrix.append(y_values)
t_matrix.append(x_values)

with open('characteristics/mN5_30_80.json', 'r') as json_file:
    y_true_mN5_30_80 = json.load(json_file)
    y_values = list(y_true_mN5_30_80['y'])
    x_values = list(y_true_mN5_30_80['x'])
    y_true_matrix.append(y_values)
    t_matrix.append(x_values)
print(f'y_true_mN5_30_80={y_true_mN5_30_80}')

print(y_true_matrix)
print(t_matrix)
#True values targer funcs------------------------------------------------------------------------

'''
Для оптимизации всего
param_bounds = {
    'C_PSDC': (1, 20), 
    'C_Proj': (1, 20), 
    'C_Inh': (1, 20),  

    'k_PSDC': (0.8, 3), 
    'k_Proj': (0.8, 3),  
    'k_Inh': (0.8, 3), 

    'vr_PSDC': (-80, -50), 
    'vr_Proj': (-80, -50),  
    'vr_Inh': (-80, -50), 

    'vt_PSDC': (-60, -20), 
    'vt_Proj': (-60, -20), 
    'vt_Inh': (-60, -20), 

    'a_PSDC': (0.001, 20), 
    'a_Proj': (0.001, 20),  
    'a_Inh': (0.001, 20), 

    'b_PSDC': (0.001, 20), 
    'b_Proj': (0.001, 20),  
    'b_Inh': (0.001, 20), 

    'c_PSDC': (-80, -40), 
    'c_Proj': (-80, -40),  
    'c_Inh': (-80, -40), 

    'd_PSDC': (1, 100), 
    'd_Proj': (1, 100),  
    'd_Inh': (1, 100), 

    'celltype_PSDC': (1, 7), 
    'celltype_Proj': (1, 7),  
    'celltype_Inh': (1, 7), 

    'ItoE_weight': (0.4, 1.0), 
    'InterPSDC_weight': (0.4, 1.0), 
    'PANtoE_weight': (0.4, 1.0), 
    'PANtoI_weight': (0.4, 1.0),

    'tauExcLoop': (2.0, 150.0),
    'tauExcProj': (2.0, 150.0),
    'tauExcInh': (2.0, 150.0),
    'tauExcPSDC': (2.0, 150.0),
    'tauPSDCtoProj': (2.0, 150.0),
    'tauPSDCtoInh': (2.0, 150.0),
    'tauInh': (2.0, 150.0),

    'tau_recExcLoop': (2.0, 1000.0),
    'tau_recExcProj': (2.0, 1000.0),
    'tau_recExcInh': (2.0, 1000.0),
    'tau_recExcPSDC': (2, 1000.0),
    'tau_recPSDCtoProj': (2, 1000.0),
    'tau_recPSDCtoInh': (2, 1000.0),
    'tau_rec_Inh': (2, 1000.0),

    'UExcLoop': (0.05, 0.8),
    'UExcProj': (0.05, 0.8),
    'UExcInh': (0.05, 0.8),
    'UExcPSDC': (0.05, 0.8),
    'UPSDCtoProj': (0.05, 0.8),
    'UPSDCtoInh': (0.05, 0.8),
    'UInh': (0.05, 0.8)
}

#Загружаем парамтры из предыдущей оптимизации
with open('initParams/init_7.json', 'r') as json_file:
    init_params = json.load(json_file)
'''

#Для оптимизации только синапса
param_bounds = {
    'ItoE_weight': (0.3, 1.0), 
    'InterPSDC_weight': (0.3, 1.0), 
    'PANtoE_weight': (0.3, 1.0), 
    'PANtoI_weight': (0.3, 1.0),

    'tauExcLoop': (2.0, 500.0),
    'tauExcProj': (2.0, 500.0),
    'tauExcInh': (2.0, 500.0),
    'tauExcPSDC': (2.0, 500.0),
    'tauPSDCtoProj': (2.0, 500.0),
    'tauPSDCtoInh': (2.0, 500.0),
    'tauInh': (2.0, 500.0),

    'tau_recExcLoop': (2.0, 1000.0),
    'tau_recExcProj': (2.0, 1000.0),
    'tau_recExcInh': (2.0, 1000.0),
    'tau_recExcPSDC': (2, 1000.0),
    'tau_recPSDCtoProj': (2, 1000.0),
    'tau_recPSDCtoInh': (2, 1000.0),
    'tau_rec_Inh': (2, 1000.0),

    'UExcLoop': (0.05, 0.8),
    'UExcProj': (0.05, 0.8),
    'UExcInh': (0.05, 0.8),
    'UExcPSDC': (0.05, 0.8),
    'UPSDCtoProj': (0.05, 0.8),
    'UPSDCtoInh': (0.05, 0.8),
    'UInh': (0.05, 0.8),

    'tau_facilExcLoop': (2.0, 1000.0),
    'tau_facilExcProj': (2.0, 1000.0),
    'tau_facilExcInh': (2.0, 1000.0),
    'tau_facilExcPSDC': (2, 1000.0),
    'tau_facilPSDCtoProj': (2, 1000.0),
    'tau_facilPSDCtoInh': (2, 1000.0),
    'tau_facilInh': (2, 1000.0),
}

with open('initParams/init_7_osyn.json', 'r') as json_file:
    init_params = json.load(json_file)

init_params['tau_facilExcLoop'] = 100
init_params['tau_facilExcProj'] = 100
init_params['tau_facilExcInh'] = 100
init_params['tau_facilExcPSDC'] = 100
init_params['tau_facilPSDCtoProj'] = 100
init_params['tau_facilPSDCtoInh'] = 100
init_params['tau_facilInh'] = 100


'''
ys = megaTargetFuncOnlySyn(t_matrix, **init_params)
print('======================================')
print(len(ys))
print(len(y_true_matrix))

print('======================================')
print(ys)
#'''

#'''
#param_names, results, pareto = run_nsga3_optimization_with_init_values2(
param_names, results, pareto = run_nsga3_optimization_with_init_values2_min_max_logging(
        mega_func=megaTargetFuncOnlySyn,
        t_matrix=t_matrix,
        y_true_matrix=y_true_matrix,
        param_bounds=param_bounds,
        n_gen=100,
        pop_size=100,
        log_path="optimisation_log.csv",
        checkpoint_path='save_state_optim.json', 
        checkpoint_every=5,
        priorities=[1, 1.5, 1, 1, 1.5, 2],
        #priorities=[1, 1.5, 1, 1.5, 2],
        init_params=init_params, 
        delta_init_params=0.01,
        loss_func=huber
)
#'''

'''
param_names, results, pareto = run_nsga3_optimization_with_init_values(        
        target_funcs=[targerFreqFuncAllPathActive, targerFreqFuncPSDCPathOff, 
                      targerFreqFuncAllPathActivePLP, 
                      targerFreqFuncAllPathActiveFreq, targerFreqFuncPSDCPathOffFreq,
                      targerPressureFunc],
        t_matrix=t_matrix,
        y_true_matrix=y_true_matrix,
        param_bounds=param_bounds,
        n_gen=200,
        pop_size=200,
        log_path="optimisation_log.csv",
        checkpoint_path='save_state_optim.json', 
        checkpoint_every=5,
        #priorities=[1, 1.5, 1, 1.5, 2], #Если отсуствуют чисто частотные ЦФ
        priorities=[1, 1.5, 1, 1, 1.5, 2],
        init_params=init_params, 
        delta_init_params=0.01,
        loss_func=huber
)
'''











