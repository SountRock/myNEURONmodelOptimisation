import numpy as np
import statistics

def compute_plp(spike_times, stim_times):
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)

    #Определение периода T из стимуляций
    isi_stim = np.diff(stim_times)  
    T = np.median(isi_stim) 
    stim_freq = 1.0 / T #частота стимула

    #Привязка спайков к фазе стимула
    phases = spike_times % T #фаза спайка в пределах периода
    in_window = np.abs(phases - T/2) <= T/12 #окно середины периода (+-T/12)

    #PLP = доля спайков в окне
    plp = np.mean(in_window.astype(float))

    return plp, stim_freq                   

''' Еще раз все проверить
Phase-locked spiking
To determine whether a nerve fiber or a neuron was entrained by the sinusoidal vibration, we calculated standardized inter-spike intervals. We chose this measure because it is immune to variability in response onset times across different stimulus repetitions. For a
given vibration frequency, the inter-spike intervals (ISI) of all possible spike pairs that occurred during stimulation were calculated and
grouped across stimulus repetitions. Because entrained spiking should yield an ISI distribution that peaks at integer multiples of the
sinusoidal stimulus period T, values were converted to standardized inter-spike intervals (SISI) according to SISI = T + (ISI - nT),
where nT is the integer multiple of T closest to ISI. Entrainment probability was defined as the percentage of ISIs in the [nT - T/12, nT + T/12] interval. 
Given that standardized ISIs are distributed between nT - T/2 and nT + T/2, entrainment probability should
equal unity in the case of perfect entrainment and be equal to 1/6 in the case of chance entrainment. For each neuron and at each
frequency we repeated the calculation of entrainment probability 1,999 times with randomly sampled ISIs (with replacement). We
then measured whether the lower 99th percentile of the repeated measures was less than 1/6, which constitutes a one-tailed bootstrap test at significance level p < 0.01. For complementary analysis of spiking regularity, we computed the coefficient of variation
(CV), which is the standard deviation of the inter-spike intervals (ISI) divided by the mean ISI
'''
def compute_plp2(spike_times, stim_times):
    spike_list = [np.asarray(s, dtype=float).ravel() for s in spike_times]
    stim_list = [np.asarray(s, dtype=float).ravel() for s in stim_times]

    total_pairs = 0
    total_in_window = 0
    for spikes, stims in zip(spike_list, stim_list):
        #Нужно минимум 2 спайка и минимум 2 стимула для определения T
        if spikes.size < 2 or stims.size < 2:
            continue

        #T как медиана положительных дельт стимов
        diffs = np.diff(stims)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue
        
        T = float(np.median(diffs))
        if not np.isfinite(T) or T <= 0:
            continue

        #Последовательные ISI данной клетки
        isis = np.diff(spikes)
        if isis.size == 0:
            continue

        #Ближайшее целое число периодов и отклонение от n*T
        n_mult = np.rint(isis / T)
        phase_offset = isis - n_mult * T

        in_window = np.abs(phase_offset) <= (T / 12.0)

        total_pairs += in_window.size
        total_in_window += int(in_window.sum())

    if total_pairs == 0:
        return 0.0

    return float(total_in_window) / float(total_pairs)


def get_central_spike_times(sim, pop_name, time_range=(0, 5000)):
    gid2pop = {c['gid']: c['tags']['pop'] for c in sim.net.allCells}
    gids = [gid for gid, pop in gid2pop.items() if pop.startswith(pop_name)]
    if not gids:
        return []

    gids = sorted(gids)
    central_gid = gids[len(gids) // 2]  
    spk_times = [t for t, gid in zip(sim.simData['spkt'], sim.simData['spkid'])
                 if gid == central_gid and time_range[0] <= t <= time_range[1]]

    return sorted(spk_times)
