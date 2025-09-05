import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.patches import Rectangle
import math
import gc

def filter_and_save(df: pd.DataFrame, filters: dict, output_path: str) -> pd.DataFrame:
    filtered_df = df.copy()
    for col, threshold in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col] < threshold]

    # добавляем новый уникальный индекс-столбец
    filtered_df = filtered_df.reset_index(drop=True)  # сбрасываем старый индекс
    filtered_df.insert(0, "id", range(1, len(filtered_df) + 1))  # новый id с 1

    filtered_df.to_csv(output_path, index=False)
    return filtered_df

def get_row_as_dict(df: pd.DataFrame, row_id: int) -> dict:
    df = df.rename(columns=lambda x: x.strip())

    if "id" not in df.columns:
        raise ValueError(f"id столбец не был найден, columns: {list(df.columns)}")

    row = df[df["id"] == row_id]

    if row.empty:
        raise ValueError(f"Строка с id={row_id} не найдена")

    return row.iloc[0].to_dict()

def drop_columns_df(df: pd.DataFrame, exact_keys: list[str] = None, patterns: list[str] = None) -> pd.DataFrame:
    cols_to_drop = set()
    if exact_keys:
        cols_to_drop.update([col for col in df.columns if col in exact_keys])
    if patterns:
        for pat in patterns:
            cols_to_drop.update([col for col in df.columns if pat in col])
    return df.drop(columns=list(cols_to_drop))


def drop_columns_dict(data: dict, exact_keys: list[str] = None, patterns: list[str] = None) -> dict:
    cleaned_dict = dict(data)  
    if exact_keys:
        for key in exact_keys:
            cleaned_dict.pop(key, None)
    if patterns:
        keys_to_remove = [k for k in cleaned_dict.keys() if any(pat in k for pat in patterns)]
        for k in keys_to_remove:
            cleaned_dict.pop(k, None)
    return cleaned_dict

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import math
import gc

def plot_iteration_spectrogram_paginated(
    df,
    iteration_col: str = None,
    cell_height_px: int = 20,
    cell_width_px: int = 40,
    vmin: float = None,
    vmax: float = None,
    save_path: str = None, 
    cmap_name: str = "yellow_red"
):
    num_df = df.select_dtypes(include="number").copy()
    if iteration_col and iteration_col in df.columns:
        y_labels = df[iteration_col].astype(str).tolist()
    else:
        y_labels = num_df.index.astype(str).tolist()

    data = num_df.values
    nrows, ncols = data.shape

    if cmap_name == "yellow_red":
        cmap = LinearSegmentedColormap.from_list("yellow_red", ["yellow", "red"])
    else:
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    dpi = 100
    #Количество строк на страницу
    rows_per_page = max(1, math.floor(800 / cell_height_px))
    num_pages = math.ceil(nrows / rows_per_page)

    if save_path:
        save_dir = os.path.dirname(save_path)
        base_name = os.path.basename(save_path)
        name, ext = os.path.splitext(base_name)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None
        name = "spectrogram_page"
        ext = ".png"

    for page_idx in range(num_pages):
        start_row = page_idx * rows_per_page
        end_row = min((page_idx + 1) * rows_per_page, nrows)

        page_data = data[start_row:end_row, :]
        page_labels = y_labels[start_row:end_row]

        nrows_page = page_data.shape[0]
        fig_width = (ncols * cell_width_px) / dpi
        fig_height = (nrows_page * cell_height_px) / dpi

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        im = ax.imshow(
            page_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect='auto',
            interpolation='nearest',
            origin='upper',
            extent=(0, ncols, 0, nrows_page)
        )

        x_centers = np.arange(ncols) + 0.5
        y_centers = np.arange(nrows_page) + 0.5

        x_labels = num_df.columns.tolist()
        ax.set_xticks(x_centers)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_yticks(y_centers)
        ax.set_yticklabels(page_labels)

        for row_idx in range(nrows_page):
            row = page_data[row_idx]
            if np.all(np.isnan(row)):
                continue
            min_pos = int(np.nanargmin(row))
            rect = Rectangle((min_pos, row_idx), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

        ax.set_xlim(0, ncols)
        ax.set_ylim(nrows_page, 0)

        ax.set_xlabel("metrics")
        ax.set_ylabel("iterations")
        ax.set_title(f"min=yellow, max=red (rows {start_row}-{end_row})")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("value", rotation=270, labelpad=12)

        plt.tight_layout()

        #Формируем путь для страницы
        if save_path:
            page_save_path = os.path.join(save_dir, f"{name}_page{page_idx+1}{ext}")
        else:
            page_save_path = f"{name}_page{page_idx+1}{ext}"

        fig.savefig(page_save_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

        #Очистка памяти
        del fig, ax, im, page_data
        gc.collect()