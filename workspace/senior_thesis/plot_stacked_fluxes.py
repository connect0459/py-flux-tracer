import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def plot_stacked_fluxes(
    input_filepath: str,
    output_dir: str,
    output_filename: str = "ch4_flux_stacked_bar_directions.png",
    concentration_unit: str = "nano",
    figsize: tuple[float, float] = (20, 13),
    ylim: float | None = None,
):
    """
    CH4フラックスの積み上げ棒グラフを作成する関数

    Args:
        input_filepath (str): 入力データのCSVファイルパス
        output_dir (str): 出力画像を保存するディレクトリ
        output_filename (str, optional): 出力画像のファイル名。デフォルトは"ch4_flux_stacked_bar_directions.png"
        concentration_unit (str, optional): 濃度の単位。'nano'または'micro'を指定。デフォルトは'nano'
        figsize (tuple[float, float], optional): 図のサイズ。デフォルトは(20, 13)
        ylim (float | None, optional): y軸の上限。Noneの場合は自動設定

    Raises:
        ValueError: concentration_unitが'nano'または'micro'以外の場合
    """
    flux_unit: str = "nmol m$^{-2}$ s$^{-1}$"
    flux_magnification: float = 1
    if concentration_unit == "micro":
        flux_unit = "μmol m$^{-2}$ s$^{-1}$"
        flux_magnification = 1 / 1000
    elif concentration_unit != "nano":
        raise ValueError(
            "concentration_unitには`micro`または`nano`を指定する必要があります。"
        )

    # データの読み込み
    df: pd.DataFrame = pd.read_csv(input_filepath, skiprows=[1])

    # 方角の配置順序を定義（左上から時計回り）
    directions_order: list[str] = ["nw", "ne", "sw", "se"]
    titles: dict[str, str] = {"nw": "北西", "ne": "北東", "sw": "南西", "se": "南東"}

    # サブプロットを含む大きな図を作成
    fig = plt.figure(figsize=figsize)

    # 各方角についてサブプロットを作成
    for idx, direction in enumerate(directions_order, 1):
        # サブプロットの位置を設定
        ax = fig.add_subplot(2, 2, idx)

        # 文字列を数値に変換
        diurnal = pd.to_numeric(df[f"diurnal_{direction}"], errors="coerce")
        gasratio = pd.to_numeric(df[f"gasratio_{direction}"], errors="coerce")

        # 単位によって倍率を補正
        diurnal *= flux_magnification

        # diurnalが10以下のデータをマスク
        valid_mask = diurnal > 10

        # gas由来とbio由来のCH4フラックスを計算（信頼性の低いデータは0に設定）
        gas = np.where(valid_mask, diurnal * gasratio / 100, 0)
        bio = np.where(valid_mask, diurnal * (100 - gasratio) / 100, 0)

        # 積み上げ棒グラフの作成
        width = 0.8
        ax.bar(
            df["month"],
            bio,
            width,
            label="bio",
            color="blue",
            alpha=0.6,
        )
        ax.bar(
            df["month"],
            gas,
            width,
            bottom=bio,
            label="gas",
            color="red",
            alpha=0.6,
        )

        # x軸の設定
        ax.set_xticks(df["month"])  # すべての月を目盛りとして設定
        ax.set_xticklabels(df["month"])  # すべての月をラベルとして表示

        # y軸の上限を設定
        if ylim is not None:
            ax.set_ylim(0, ylim)

        # gas比率の表示（信頼性の低いデータは表示しない）
        for i, (g, b, is_valid) in enumerate(zip(gas, bio, valid_mask)):
            if is_valid:
                total = g + b
                ratio = g / total * 100
                ax.text(
                    df["month"][i], total, f"{ratio:.0f}%", ha="center", va="bottom"
                )

        # x軸とy軸のラベルを各サブプロットに設定
        ax.set_xlabel("Month")
        ax.set_ylabel(f"CH$_4$ flux ({flux_unit})")

        # グラフの装飾
        ax.set_title(titles[direction])

    # サブプロット間の間隔を調整
    plt.tight_layout()

    # 図全体の凡例を下部に配置
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=len(handles)
    )

    # 凡例のためのスペースを確保
    plt.subplots_adjust(bottom=0.15)

    # グラフの保存
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, output_filename),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


"""
Ubuntu環境でのフォントの手動設定
不要な方はコメントアウトして実行してください。
ここでは日本語フォントを読み込んでいます。

1. インストール : `sudo apt update && sudo apt install -y fonts-ipafont`
2. キャッシュ削除 : `fc-cache -fv`
3. パスを確認 : `fc-list | grep -i ipa`

得られたパスを`font_path`に記述して実行
これでも読み込まれない場合は、matplotlibのキャッシュを削除する

4. `rm ~/.cache/matplotlib/fontlist-v390.json` # 実際のファイル名に変更
"""
# フォントファイルを登録
font_paths: list[str] = [
    "/home/connect0459/.local/share/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/.local/share/fonts/msgothic.ttc",  # 日本語のデフォルト
]
for path in font_paths:
    fm.fontManager.addfont(path)

# rcParamsでの全体的な設定
plt.rcParams.update(
    {
        # "font.family": ["Dejavu Sans"],
        "font.family": ["Arial", "MS Gothic"],
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
    }
)

tag: str = "average-10_16"
project_files_dir: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private"
)

if __name__ == "__main__":
    plot_stacked_fluxes(
        input_filepath=f"{project_files_dir}/analyze_monthly-2024.12.18.csv",
        output_dir=os.path.join(project_files_dir, "outputs", "stacked_fluxes"),
        output_filename=f"ch4_flux_stacked_bar_directions-{tag}.png",
        ylim=100,
    )
