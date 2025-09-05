import pandas as pd

'''
Скрипт для быстрой фильтрации лога
'''

def filter_and_save(df: pd.DataFrame, filters: dict, output_path: str) -> pd.DataFrame:
    filtered_df = df.copy()
    for col, threshold in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] < threshold]

    # добавляем новый уникальный индекс-столбец
    filtered_df = filtered_df.reset_index(drop=True) #сбрасываем старый индекс
    filtered_df.insert(0, "id", range(1, len(filtered_df) + 1))  #новый id с 1

    filtered_df.to_csv(output_path, index=False)
    return filtered_df

# tf1_loss,tf2_loss,tf3_loss,tf4_loss tf7_loss

#==============================================================
df = pd.read_csv("optimisation_logNNNew.csv")
filters = {"tf1_loss": 0.2, "tf2_loss": 0.2, "tf3_loss": 0.2, "tf7_loss": 0.2}
filtered = filter_and_save(df, filters, "NNNew01_2.csv")
print(filtered[['id', 'tf1_loss', 'tf2_loss', 'tf3_loss', 'tf4_loss', 'tf5_loss', 'tf6_loss', 'tf7_loss']].head())