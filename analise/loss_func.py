import pandas as pd
import matplotlib.pyplot as plt

'''
Скрипт для отрисовки динамики loss нескольких ЦФ
'''

df = pd.read_csv("optimisation_logNNNew.csv")
cols = ['tf1_loss','tf2_loss','tf3_loss','tf4_loss','tf5_loss','tf6_loss','tf7_loss']
result = df.groupby("generation")[cols].min().reset_index()

group1 = ['tf1_loss','tf2_loss','tf3_loss','tf4_loss','tf7_loss']      
group2 = ['tf5_loss','tf6_loss']                 
group3 = ['tf2_loss','tf7_loss']                

color_map = {
    'tf1_loss': 'red',
    'tf2_loss': 'blue',
    'tf3_loss': 'green',
    'tf4_loss': 'orange',
    'tf5_loss': 'purple',
    'tf6_loss': 'brown',
    'tf7_loss': 'cyan'
}

plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
for c in group1:
    plt.plot(result["generation"], result[c], label=c, color=color_map[c])
plt.title("Spikes per cycle(tf1,tf2), Phase locking(tf4), Pressure (tf7)")
plt.xlabel("Generation")
plt.ylabel("Min loss")
plt.legend()
plt.grid(True)

plt.subplot(1,3,2)
for c in group2:
    plt.plot(result["generation"], result[c], label=c, color=color_map[c])
plt.title("Freq Optimisation(tf5,tf6)")
plt.xlabel("Generation")
plt.ylabel("Min loss")
plt.legend()
plt.grid(True)

plt.subplot(1,3,3)
for c in group3:
    plt.plot(result["generation"], result[c], label=c, color=color_map[c])
plt.xlabel("Generation")
plt.ylabel("Min loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()