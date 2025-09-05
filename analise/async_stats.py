import numpy as np
import statistics

def generate_async_pattern(num_chs, duration_ms, maxISI, delay_arr, lower_bound=0.5, upper_bound=1.0):
    async_matrix = []
    for i in range(num_chs):
        temp = generate_spikes_with_asynchrony4(maxISI, duration_ms, seed=None, delay=delay_arr[i], 
                                                lower_bound=lower_bound, upper_bound=upper_bound)
        async_matrix.append(temp)
    
    return async_matrix

def generate_spikes_with_asynchrony4(maxISI, duration_ms, seed=None, delay=0, lower_bound=0.5, upper_bound=1.0):
    if seed is not None:
        np.random.seed(seed)

    #Генерация случайных чисел для всех временных меток
    num_spikes = int(duration_ms // maxISI) + 1# Оценка максимального количества спайков
    p_values = np.random.uniform(lower_bound, upper_bound, num_spikes)

    #Накопление временных меток
    spike_times = np.cumsum(maxISI * p_values) + delay

    #Фильтрация временных меток, которые превышают длительность
    spike_times = spike_times[spike_times < duration_ms]

    #Округление временных меток до целых чисел
    spike_times = np.round(spike_times).astype(float)

    return [float(t) for t in spike_times]

def jitter_rms_deviation_fixed_max_isi(spike_times, max_isi_ms):
    """
    Оценка притяжения спайков к равномерной сетке, построенной с шагом maxISI,
    через RMS (среднеквадратичное отклонение) относительно ожидаемого времени следующего спайка.
    """
    spike_times = np.sort(np.array(spike_times))
    if len(spike_times) == 0:
        raise ValueError("Список спайков пуст.")

    expected_spikes = []
    deviations = []

    #Начинаем с первого спайка и ожидаем каждый следующий через max_isi_ms
    expected_time = spike_times[0]
    for actual_time in spike_times:
        expected_spikes.append(expected_time)
        deviations.append((actual_time - expected_time) ** 2)
        expected_time += max_isi_ms

    rms_deviation = np.sqrt(np.mean(deviations)) if deviations else np.nan
    freq_deviation = 1000 / rms_deviation if rms_deviation > 0 else 0.0
    precision = len(spike_times) / len(expected_spikes) if len(expected_spikes) > 0 else 0.0

    return {
        'max_isi_ms': float(max_isi_ms),
        'rms_deviation_ms': float(rms_deviation),
        'expected_spikes': [float(t) for t in expected_spikes],
        'in_rhythm_spikes': [float(t) for t in spike_times],
        'deviation_score': float(freq_deviation / (1000.0 / max_isi_ms)) if max_isi_ms > 0 else 0.0,
        'precision': float(precision)
    }

def jitter_rms_deviation_fixed_max_isi2(spike_times, max_isi_ms):
    """
    Оценка притяжения спайков к равномерной сетке, построенной с шагом maxISI,
    через RMS (среднеквадратичное отклонение) относительно ожидаемого времени следующего спайка.
    """
    spike_times = np.sort(np.array(spike_times))
    if len(spike_times) == 0:
        raise ValueError("Список спайков пуст.")

    expected_spikes = []
    deviations = []

    #Начинаем с первого спайка и ожидаем каждый следующий через max_isi_ms
    expected_time = spike_times[0]
    for actual_time in spike_times:
        expected_spikes.append(expected_time)
        deviations.append((actual_time - expected_time) ** 2)
        expected_time += max_isi_ms

    rms_deviation = np.sqrt(np.mean(deviations)) if deviations else np.nan
    freq_deviation = 1000 / rms_deviation if rms_deviation > 0 else 0.0
    precision = len(spike_times) / len(expected_spikes) if len(expected_spikes) > 0 else 0.0

    return {
        'max_isi_ms': float(max_isi_ms),
        'rms_deviation_ms': float(rms_deviation),
        'expected_spikes': [float(t) for t in expected_spikes],
        'in_rhythm_spikes': [float(t) for t in spike_times],
        'deviation_score': float(freq_deviation),
        'precision': float(precision)
    }

def jitter_analise_for_matrix(matrix, maxISI=20):
    result = []
    alienation_score = []
    for m in matrix:
        temp = jitter_rms_deviation_fixed_max_isi(m, maxISI)
        result.append(temp)
        alienation_score.append(temp['deviation_score'])
    
    return statistics.median(alienation_score), result

def jitter_analise_for_matrix2(matrix, maxISI=20):
    result = []
    alienation_score = []
    for m in matrix:
        temp = jitter_rms_deviation_fixed_max_isi2(m, maxISI)
        result.append(temp)
        alienation_score.append(temp['deviation_score'])
    
    return statistics.median(alienation_score), result

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

''' 
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