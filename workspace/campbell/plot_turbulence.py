import os
import re
from pathlib import Path
from typing import Literal

from tqdm import tqdm  # プログレスバー用

from py_flux_tracer import (
    EddyDataFiguresGenerator,
    EddyDataPreprocessor,
    setup_plot_params,
)

"""
------ config start ------
"""

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

output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/outputs"
)

"""
------ config end ------
"""

if __name__ == "__main__":
    edfg = EddyDataFiguresGenerator(fs=10)

    # 乱流データの設定
    data_dir = "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/data/eddy_csv-resampled-for_turb"
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
    for config in tqdm(turbulence_configs, desc="Processing turbulences"):
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

            # 図の作成と保存
            edfg.plot_turbulence(
                df=df_for_turb,
                col_uz="edp_wind_w",
                col_ch4="ch4_ppm_cal",
                col_c2h6="c2h6_ppb_cal",
                output_dirpath=(
                    os.path.join(output_dirpath, "turbulences", "for_turb")
                ),
                output_filename=f"turbulence-{date}.png",
                add_serial_labels=False,
                figsize=(20, 10),
            )
            # mfg.logger.info(f"'{date}'の'turbulences'を作成しました。")

        except (IndexError, ValueError) as e:
            print(f"ファイル名'{filename}'から日時を抽出できませんでした: {e!s}")
            continue
