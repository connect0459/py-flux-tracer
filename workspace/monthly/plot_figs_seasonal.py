import os
from pathlib import Path

import numpy as np
import pandas as pd

from py_flux_tracer import (
    MonthlyConverter,
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
    font_size=28,
    tick_size=28,
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-12-31"  # yyyy-MM-ddで指定
output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/monthly/private/outputs"
)
diurnal_subplot_fontsize: float = 36

"""
------ config end ------
"""

if __name__ == "__main__":
    # Ultra
    with MonthlyConverter(
        "/home/connect0459/labo/py-flux-tracer/workspace/monthly/private",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        df_ultra = converter.read_sheets(
            sheet_names=["Final", "Final.SA", "eddy"],
            columns=[
                "Fch4_ultra",
                "Fc2h6_ultra",
                "CH4_ultra_cal",
                "C2H6_ultra_cal",
                "Fch4_open",
                "slope",
                "intercept",
                "r_value",
                "p_value",
                "stderr",
                "RSSI",
                "Wind direction",
                "WS vector",
                " TF-wch4",
                " TF-wc2h6",
            ],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )

    # Picarro
    with MonthlyConverter(
        "/home/connect0459/labo/py-flux-tracer/workspace/monthly/private",
        file_pattern="SA.Picaro.*.xlsx",
    ) as converter:
        df_picarro = converter.read_sheets(
            sheet_names=["Final", "eddy"],
            columns=["Fch4_picaro", " TFwch4_c"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )
        # print(df_picarro.head(10))

    # 両方を結合したDataFrameを明示的に作成
    df_combined = MonthlyConverter.merge_dataframes(df1=df_ultra, df2=df_picarro)

    # 濃度データはキャリブレーション後から使用
    # Dateカラムをインデックスに設定
    df_combined["Date"] = pd.to_datetime(df_combined["Date"])
    df_combined["Date_index"] = df_combined["Date"]
    df_combined.set_index("Date_index", inplace=True)

    # 濃度データのみを2024-10-01以降に切り出し
    filter_date = pd.to_datetime("2024-10-01")
    mask = df_combined.index < filter_date
    df_combined.loc[mask, "CH4_ultra_cal"] = np.nan
    df_combined.loc[mask, "C2H6_ultra_cal"] = np.nan

    # RSSIが40未満のデータは信頼性が低いため、Fch4_openをnanに置換
    df_combined.loc[df_combined["RSSI"] < 40, "Fch4_open"] = np.nan

    mfg = MonthlyFiguresGenerator()

    seasons: list[list[int]] = [[6, 7, 8], [9, 10, 11]]
    seasons_tags: list[str] = ["summer", "fall"]
    seasons_subplot_labels: list[str] = ["(a)", "(b)"]
    for season, tag, subplot_label in zip(
        seasons, seasons_tags, seasons_subplot_labels, strict=True
    ):
        # 季節ごとのDataFrameを作成(extract_period_dataを使用)
        start_date = f"2024-{season[0]:02d}-01"
        end_date = f"2024-{season[-1]:02d}-{MonthlyConverter.get_last_day_of_month(2024, season[-1])}"
        df_season: pd.DataFrame = MonthlyConverter.extract_period_data(
            df=df_combined,
            start_date=start_date,
            end_date=end_date,
        )

        mfg.logger.info(f"season:'{tag}'の図を作成します。")
        # 日変化パターンを月ごとに作成
        mfg.plot_c1c2_fluxes_diurnal_patterns(
            df=df_season,
            y_cols_ch4=["Fch4_ultra", "Fch4_open", "Fch4_picaro"],
            y_cols_c2h6=["Fc2h6_ultra"],
            labels_ch4=["Ultra", "Open Path", "G2401"],
            labels_c2h6=["Ultra"],
            legend_only_ch4=True,
            add_label=True,
            # add_legend=True,
            # add_label=False,
            add_legend=False,
            subplot_fontsize=diurnal_subplot_fontsize,
            colors_ch4=["black", "red", "blue"],
            colors_c2h6=["black"],
            output_dirpath=(os.path.join(output_dirpath, "diurnal")),
            output_filename=f"diurnal-{tag}.png",  # タグ付けしたファイル名
        )
        # ultraのみ
        mfg.plot_c1c2_fluxes_diurnal_patterns(
            df=df_season,
            y_cols_ch4=["Fch4_ultra"],
            y_cols_c2h6=["Fc2h6_ultra"],
            labels_ch4=["Ultra"],
            labels_c2h6=["Ultra"],
            legend_only_ch4=True,
            # add_label=True,
            # add_legend=True,
            # add_label=False,
            add_legend=False,
            # figsize=(10, 6),
            figsize=(12, 5),
            ax1_ylim=(0, 100),
            ax2_ylim=(0, 5),
            subplot_fontsize=diurnal_subplot_fontsize,
            subplot_label_ch4=None,
            subplot_label_c2h6=None,
            colors_ch4=["red"],
            colors_c2h6=["orange"],
            output_dirpath=(os.path.join(output_dirpath, "diurnal")),
            output_filename=f"diurnal_ultra-{tag}.png",  # タグ付けしたファイル名
        )
        # # 凡例のみ
        # mfg.plot_c1c2_fluxes_diurnal_patterns(
        #     df=df_season,
        #     y_cols_ch4=["Fch4_ultra"],
        #     y_cols_c2h6=["Fc2h6_ultra"],
        #     labels_ch4=[r"CH$_4$ flux"],
        #     labels_c2h6=[r"C$_2$H$_6$ flux"],
        #     # legend_only_ch4=True,
        #     # add_label=True,
        #     add_legend=True,
        #     # add_label=False,
        #     # add_legend=False,
        #     # figsize=(10, 6),
        #     figsize=(12, 5),
        #     ax1_ylim=(0, 100),
        #     ax2_ylim=(0, 5),
        #     subplot_fontsize=diurnal_subplot_fontsize,
        #     subplot_label_ch4=None,
        #     subplot_label_c2h6=None,
        #     colors_ch4=["red"],
        #     colors_c2h6=["orange"],
        #     output_dirpath=(os.path.join(output_dirpath, "diurnal")),
        #     output_filename="diurnal_ultra-legend.png",  # タグ付けしたファイル名
        # )

        mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
            df=df_season,
            y_col_ch4="Fch4_ultra",
            y_col_c2h6="Fc2h6_ultra",
            legend_only_ch4=True,
            add_label=True,
            # add_legend=True,
            # add_label=False,
            add_legend=False,
            subplot_fontsize=diurnal_subplot_fontsize,
            plot_holiday=False,
            output_dirpath=(os.path.join(output_dirpath, "diurnal_by_date")),
            output_filename=f"diurnal_by_date-{tag}.png",  # タグ付けしたファイル名
        )
        mfg.logger.info("'diurnals-seasons'を作成しました。")

        mfg.plot_source_contributions_diurnal(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "sources")),
            output_filename=f"source_contributions_seasons-{tag}.png",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            subplot_label=subplot_label,
            subplot_fontsize=24,
            y_max=110,
            print_summary=False,
        )
        # スライド用のサイズ
        mfg.plot_source_contributions_diurnal(
            df=df_season,
            figsize=(10, 8),
            output_dirpath=(os.path.join(output_dirpath, "sources")),
            output_filename=f"source_contributions_seasons-slide-{tag}.png",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            subplot_label=subplot_label,
            subplot_fontsize=24,
            y_max=100,
            label_bio="生物起源",
            label_gas="都市ガス起源",
            add_legend=False,
            print_summary=False,
        )
        mfg.plot_source_contributions_diurnal(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "sources")),
            output_filename="source_contributions-legend-ja.png",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            subplot_label=subplot_label,
            subplot_fontsize=24,
            y_max=110,
            label_bio="生物起源",
            label_gas="都市ガス起源",
            add_legend=True,
            print_summary=False,
        )

        season_mono_config: dict[str, str | float | bool] = {
            # "color_bio": "gray",
            # "color_gas": "black",
            "color_bio": "black",
            "color_gas": "gray",
            "label_bio": "生物起源",
            "label_gas": "都市ガス起源",
            "flux_alpha": 0.7,
        }
        mfg.plot_source_contributions_diurnal(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "sources")),
            output_filename=f"source_contributions_seasons-mono-{tag}.png",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            subplot_fontsize=24,
            figsize=(6, 5),
            y_max=100,
            color_bio=str(season_mono_config["color_bio"]),
            color_gas=str(season_mono_config["color_gas"]),
            flux_alpha=float(season_mono_config["flux_alpha"]),
            label_bio=str(season_mono_config["label_bio"]),
            label_gas=str(season_mono_config["label_gas"]),
            # add_legend=(tag == "fall"),
            add_xlabel=False,
            add_ylabel=False,
            add_legend=False,
            print_summary=False,
        )
        mfg.plot_source_contributions_diurnal_by_date(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "sources")),
            output_filename=f"source_contributions_seasons_by_date-mono-{tag}.png",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            subplot_fontsize=24,
            y_max=110,
            color_bio=str(season_mono_config["color_bio"]),
            color_gas=str(season_mono_config["color_gas"]),
            flux_alpha=float(season_mono_config["flux_alpha"]),
            label_bio=str(season_mono_config["label_bio"]),
            label_gas=str(season_mono_config["label_gas"]),
            add_xlabel=False,
            add_ylabel=False,
            add_legend=False,
            # print_summary=False,
        )

        # 2024年7月1日から2024年7月31日までのデータを除外
        df_season_filtered = df_season.copy()
        mask = ~(
            (df_season_filtered["Date"] >= "2024-07-01")
            & (df_season_filtered["Date"] <= "2024-07-31")
            & (df_season_filtered["Wind direction"] >= 90)
            & (df_season_filtered["Wind direction"] <= 180)
        )
        df_season = df_season_filtered[mask]

        df_season = df_season[
            (df_season["Date"].dt.hour >= 10) & (df_season["Date"].dt.hour < 16)
        ]

        mfg.plot_wind_rose_sources(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "wind_rose")),
            output_filename=f"wind_rose_stacked-mono-{tag}.png",
            col_datetime="Date",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            col_wind_dir="Wind direction",
            ymax=100,
            color_bio=str(season_mono_config["color_bio"]),
            color_gas=str(season_mono_config["color_gas"]),
            flux_alpha=float(season_mono_config["flux_alpha"]),
            label_bio=str(season_mono_config["label_bio"]),
            label_gas=str(season_mono_config["label_gas"]),
            gap_degrees=3.0,
            num_directions=8,  # 方位の数(8方位)
            subplot_label=subplot_label,
            print_summary=False,  # 統計情報を表示するかどうか
            add_legend=False,
            stack_bars=True,
            save_fig=True,
            show_fig=False,
        )

        mfg.plot_wind_rose_sources(
            df=df_season,
            output_dirpath=(os.path.join(output_dirpath, "wind_rose")),
            output_filename="wind_rose_stacked-mono-legend-ja.png",
            col_datetime="Date",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            col_wind_dir="Wind direction",
            ymax=100,
            color_bio=str(season_mono_config["color_bio"]),
            color_gas=str(season_mono_config["color_gas"]),
            flux_alpha=float(season_mono_config["flux_alpha"]),
            label_bio=str(season_mono_config["label_bio"]),
            label_gas=str(season_mono_config["label_gas"]),
            num_directions=8,  # 方位の数(8方位)
            subplot_label=subplot_label,
            print_summary=False,  # 統計情報を表示するかどうか
            add_legend=True,
            stack_bars=True,
            save_fig=True,
            show_fig=False,
        )
