import json
import numpy as np

def generate_spike_times(freq_hz, duration_ms, delay_ms):
    isi = 1000.0 / freq_hz

    num_spikes = int((duration_ms + delay_ms) / isi) + 1

    times = np.arange(num_spikes) * isi + delay_ms
    times = times[times <= duration_ms + delay_ms]

    return times.round(4).tolist()

def estimate_frequency(spike_times_ms):
    isis = [spike_times_ms[i+1] - spike_times_ms[i] for i in range(len(spike_times_ms)-1)]
    mean_isi = sum(isis) / len(isis)  
    freq_hz = 1000.0 / mean_isi      
    return round(freq_hz, 4)

num_chs = 10
duration_ms = 1000

delta = 0.5
def_delay = 2
lower_bound = def_delay - def_delay*delta
upper_bound = def_delay + def_delay*delta
delays = np.random.uniform(lower_bound, upper_bound, num_chs)

with open('characteristics\\freq_freq_nerve.json', 'r') as json_file:
    data = json.load(json_file)

freq_stim = data['x']
freq_response = data['y']

for i in range(len(freq_stim)):
    matrix_pattern = []
    for ch in range(num_chs):
        channel = generate_spike_times(freq_response[i], duration_ms, delays[ch])
        matrix_pattern.append(channel)
    
    with open(f'inputs\\{freq_stim[i]}HzPattern.json', 'w') as json_file:
        json.dump(matrix_pattern, json_file)

for i in range(len(freq_stim)):
    with open(f'inputs\\{freq_stim[i]}HzPattern.json', 'r') as json_file:
        data = json.load(json_file)
    true_freq = estimate_frequency(data[int(num_chs/2)])

    print(f'stim: {freq_stim[i]} true: {true_freq} expect: {freq_response[i]}')
