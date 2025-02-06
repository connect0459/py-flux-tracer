import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from py_flux_tracer import setup_plot_params


def plot_gas_stats(
    input_csv: str | Path,
    output_dirpath: str,
    output_filename: str = "gas_stats.png",
    figsize: tuple[float, float] = (10, 6),
    dpi: float | None = 350,
    save_fig: bool = True,
    show_fig: bool = False,
) -> None:
    """ガス販売量の積み上げグラフを作成する

    Parameters
    ----------
        input_csv: str | Path, optional
            入力CSVファイルのパス, by default "workspace/senior_thesis/private/gas_stats.csv"
        output_dirpath: str, optional
            出力ディレクトリ, by default "workspace/senior_thesis/figures/gas_stats"
        output_filename: str, optional
            出力ファイル名, by default "gas_stats.png"
        figsize: tuple[float, float], optional
            図のサイズ, by default (10, 6)
        dpi: float | None, optional
            図のdpi。デフォルトは350。
        save_fig: bool, optional
            図を保存するかどうか, by default True
        show_fig: bool, optional
            図を表示するかどうか, by default False
    """
    # データの読み込み
    df = pd.read_csv(input_csv)

    # 千m³からMm³に変換(1000で割る)
    for col in ["家庭用", "工業用", "商業用", "公用・医療用"]:
        df[col] = df[col] / 1000

    # プロットする項目と色の設定
    categories = ["家庭用", "工業用", "商業用", "公用・医療用"]
    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

    # 積み上げグラフの作成
    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(df))
    for category, color in zip(categories, colors, strict=True):
        ax.bar(
            df.index,
            df[category],
            bottom=bottom,
            label=category,
            color=color,
            width=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += df[category]

    # x軸の設定
    months = ["4", "5", "6", "7", "8", "9", "10", "11", "12", "1", "2", "3"]
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(months)

    # 軸ラベルの設定
    ax.set_xlabel("月")
    ax.set_ylabel("ガス販売量 (Mm³)")

    # グリッド線の追加
    ax.grid(True, alpha=0.3, axis="y")

    # 凡例の設定
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.5, -0.2),  # 位置を下に調整
        ncol=len(categories),
        frameon=False,
    )

    # レイアウトの調整
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()

    # 出力ディレクトリの作成
    os.makedirs(output_dirpath, exist_ok=True)
    output_filepath = os.path.join(output_dirpath, output_filename)

    # 図の保存と表示
    if save_fig:
        plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
    if show_fig:
        plt.show()
    plt.close()


if __name__ == "__main__":
    # フォントファイルを登録
    font_paths: list[str | Path] = [
        "/home/connect0459/.local/share/fonts/arial.ttf",  # 英語のデフォルト
        "/home/connect0459/.local/share/fonts/msgothic.ttc",  # 日本語のデフォルト
    ]
    # プロットの書式を設定
    setup_plot_params(
        font_family=["Arial", "MS Gothic"],
        font_paths=font_paths,
        font_size=24,
        tick_size=24,
    )
    plot_gas_stats(
        input_csv="workspace/senior_thesis/private/gas_stats.csv",
        output_dirpath="workspace/senior_thesis/private/outputs/gas_stats",
    )
