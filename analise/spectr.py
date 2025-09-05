import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from analise_optim_data import get_row_as_dict, drop_columns_dict, plot_iteration_spectrogram_paginated
import json

'''
Скрипт для отрисвки спектрограммы по loss функциям по каждым предложеным параметрам из лога +
Отбор наиболее выгодных параметров и сохрание их в json для дальнейшей подстановки
'''


df = pd.read_csv("OnlySynNNNew01.csv")
df2 = df[['tf1_loss', 'tf2_loss', 'tf3_loss', 'tf4_loss', 'tf5_loss', 'tf6_loss', 'tf7_loss']]
print(df2)

#Генерация страниц спектрограммы
plot_iteration_spectrogram_paginated(
    df2,
    cell_height_px =25,
    cell_width_px = 100,
    vmin=0.0001,
    vmax=0.5,
    save_path="NNNew01_2/NNNew01_2"
    )

#Сохранение выбранного параметра
row_dict = get_row_as_dict(df, 2)
row_dict = drop_columns_dict(row_dict, ['generation', 'id', 'hypervolume'], ["_loss"])
#print(row_dict)

with open('init_7_osyn2.json', 'w') as json_file:
    json.dump(row_dict, json_file)
