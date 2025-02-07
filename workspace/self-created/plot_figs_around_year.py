import os
from pathlib import Path

import pandas as pd

from py_flux_tracer import (
    MonthlyFiguresGenerator,
    setup_plot_params,
)

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
)

output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/private/self-created/outputs"
)

"""
------ config end ------
"""

if __name__ == "__main__":
    mfg = MonthlyFiguresGenerator()
    # 正月の平日/休日
    mfg.logger.info("年末年始の処理を開始します。")
    # 月ごとのDataFrameを作成
    df_around_year: pd.DataFrame = pd.read_csv(
        "/home/connect0459/labo/py-flux-tracer/workspace/private/self-created/around_year_2025.csv",
        skiprows=[1],
        na_values=[
            "#DIV/0!",
            "#VALUE!",
            "#REF!",
            "#N/A",
            "#NAME?",
            "NAN",
            "nan",
        ],
    )
    df_around_year_for_diurnal = df_around_year.copy()
    mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
        df=df_around_year_for_diurnal,
        y_col_ch4="Fch4_ultra",
        y_col_c2h6="Fc2h6_ultra",
        output_dirpath=os.path.join(output_dirpath, "around_year"),
        output_filename="diurnal_by_date_around_year.png",
        add_label=True,
        subplot_label_ch4=None,
        subplot_label_c2h6=None,
        plot_holiday=False,
        ax1_ylim=(-20, 150),
        ax2_ylim=(-1, 8),
    )
    mfg.plot_source_contributions_diurnal_by_date(
        df=df_around_year_for_diurnal,
        output_dirpath=(os.path.join(output_dirpath, "around_year")),
        output_filename="source_contributions_by_date_around_year.png",
        col_ch4_flux="Fch4_ultra",
        col_c2h6_flux="Fc2h6_ultra",
        subplot_fontsize=24,
        y_max=100,
        # add_xlabel=False,
        # add_ylabel=False,
        add_legend=True,
    )
