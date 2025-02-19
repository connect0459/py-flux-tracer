import os
import re
from pathlib import Path
from typing import Literal

from modules.eddy_data_figures_generator import EddyDataFiguresGenerator
from modules.eddy_data_preprocessor import EddyDataPreprocessor
from modules.utilities import setup_plot_params
from tqdm import tqdm  # プログレスバー用

"""
------ config start ------
"""

# フォントファイルを登録
font_filepaths: list[str | Path] = [
    "/home/connect0459/.local/share/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/.local/share/fonts/msgothic.ttc",  # 日本語のデフォルト
]
# プロットの書式を設定
setup_plot_params(
    font_family=["Arial", "MS Gothic"],
    font_filepaths=font_filepaths,
    # font_size=24,
    # tick_size=24,
    font_size=32,
    tick_size=32,
)

output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/campbell/private/outputs"
)

"""
------ config end ------
"""

if __name__ == "__main__":
    edfg = EddyDataFiguresGenerator(fs=10)

    # 乱流データの設定
    data_dir = "/home/connect0459/labo/py-flux-tracer/workspace/campbell/private/eddy_csv-resampled-for_turb"
    turbulence_configs: list[
        dict[Literal["filename", "ch4_offset", "c2h6_offset"], str | float]
    ] = [
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_105_2024_10_08_1200-resampled.csv",
            "ch4_offset": 0.012693983,
            "c2h6_offset": -13.1381285,
        },
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_106_2024_10_09_0830-resampled.csv",
            "ch4_offset": 0.009960667,
            "c2h6_offset": -13.19275367,
        },
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_106_2024_10_09_2000-resampled.csv",
            "ch4_offset": 0.0095262,
            "c2h6_offset": -13.35212183,
        },
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_0000-resampled.csv",
            "ch4_offset": 0.009106433,
            "c2h6_offset": -13.35047267,
        },
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_107_2024_10_10_0200-resampled.csv",
            "ch4_offset": 0.009106433,
            "c2h6_offset": -13.35047267,
        },
        {
            "filename": "TOA5_37477.SAC_Ultra.Eddy_109_2024_10_12_1400-resampled.csv",
            "ch4_offset": 0.011030083,
            "c2h6_offset": -11.82567127,
        },
    ]

    # 各設定に対して処理を実行
    for config in tqdm(turbulence_configs, desc="Processing"):
        target_filename: str = str(config["filename"])
        ch4_offset = config["ch4_offset"]
        c2h6_offset = config["c2h6_offset"]
        # ディレクトリ内の全てのCSVファイルを取得
        filepath = os.path.join(data_dir, target_filename)
        # ファイル名から日時を抽出
        filename = os.path.basename(filepath)
        try:
            # ファイル名をアンダースコアで分割し、日時部分を取得
            parts = filename.split("_")
            # 年、月、日、時刻の部分を見つける
            for i, part in enumerate(parts):
                if part == "2024":  # 年を見つけたら、そこから4つの要素を取得
                    date = "_".join(
                        [
                            parts[i],  # 年
                            parts[i + 1],  # 月
                            parts[i + 2],  # 日
                            re.sub(
                                r"(\+|-resampled\.csv)", "", parts[i + 3]
                            ),  # 時刻から+と-resampled.csvを削除
                        ]
                    )
                    break

            # データの読み込みと処理
            edp = EddyDataPreprocessor(10)
            df_for_turb, _ = edp.get_resampled_df(filepath=filepath)
            df_for_turb = edp.add_uvw_columns(df_for_turb)
            df_for_turb["ch4_ppm_cal"] = df_for_turb["Ultra_CH4_ppm_C"] - ch4_offset
            df_for_turb["c2h6_ppb_cal"] = df_for_turb["Ultra_C2H6_ppb"] - c2h6_offset

            # 平均からの偏差(プライム)を計算
            w_mean = df_for_turb["edp_wind_w"].mean()
            ch4_mean = df_for_turb["ch4_ppm_cal"].mean()
            c2h6_mean = df_for_turb["c2h6_ppb_cal"].mean()

            df_for_turb["w_prime"] = df_for_turb["edp_wind_w"] - w_mean
            df_for_turb["ch4_prime"] = df_for_turb["ch4_ppm_cal"] - ch4_mean
            df_for_turb["c2h6_prime"] = df_for_turb["c2h6_ppb_cal"] - c2h6_mean

            # 図の作成と保存
            edfg.plot_turbulence(
                df=df_for_turb,
                col_uz="w_prime",  # 鉛直風速の偏差
                col_ch4="ch4_prime",  # メタン濃度の偏差
                col_c2h6="c2h6_prime",  # エタン濃度の偏差
                output_dirpath=(
                    os.path.join(output_dirpath, "turbulences", "for_turb")
                ),
                output_filename=f"turbulence_prime-{date}.png",
                add_serial_labels=False,
                figsize=(20, 10),
                show_fig=False,
                ylabel_uz=r"$w'$ (m s$^{-1}$)",  # y軸ラベルを偏差用に変更
                ylabel_ch4=r"$\mathrm{CH_4}'$ (ppm)",
                ylabel_c2h6=r"$\mathrm{C_2H_6}'$ (ppb)",
            )

        except (IndexError, ValueError) as e:
            print(f"ファイル名'{filename}'から日時を抽出できませんでした: {e!s}")
            continue
