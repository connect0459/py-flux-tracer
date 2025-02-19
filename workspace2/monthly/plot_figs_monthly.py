import os
from pathlib import Path

import numpy as np
import pandas as pd
from modules.monthly_converter import MonthlyConverter
from modules.monthly_figures_generator import MonthlyFiguresGenerator
from modules.utilities import setup_plot_params

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

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-12-31"  # yyyy-MM-ddで指定
months: list[int] = [5, 6, 7, 8, 9, 10, 11, 12]
subplot_labels: list[list[str | None]] = [
    ["(a)", None],
    ["(b)", None],
    ["(c)", None],
    ["(d)", None],
    ["(e)", None],
    ["(f)", None],
    ["(g)", None],
    ["(h)", None],
]
output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/monthly/private/outputs"
)

# フラグ
# plot_timeseries: bool = True
# plot_comparison: bool = True
# plot_diurnals: bool = True
# diurnal_subplot_fontsize: float = 36
# plot_scatter: bool = True
# plot_sources: bool = True
plot_timeseries: bool = False
plot_comparison: bool = False
plot_diurnals: bool = False
diurnal_subplot_fontsize: float = 36
plot_scatter: bool = False
plot_sources: bool = True

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

    # CH4_ultraをppb単位に直したカラムを作成
    df_combined["CH4_ultra_cal_ppb"] = df_combined["CH4_ultra_cal"] * 1000

    # # 伝達関数の補正量が max_ratio を超えるときにnanとする
    # max_ratio: float = 2.5
    # # ultra
    # df_combined["Fch4_ultra"] = np.where(
    #     df_combined[" TF-wch4"] > max_ratio, np.nan, df_combined["Fch4_ultra"]
    # )
    # df_combined["Fc2h6_ultra"] = np.where(
    #     df_combined[" TF-wc2h6"] > max_ratio, np.nan, df_combined["Fc2h6_ultra"]
    # )
    # # g2401
    # df_combined["Fch4_picaro"] = np.where(
    #     df_combined[" TFwch4_c"] > max_ratio, np.nan, df_combined["Fch4_picaro"]
    # )

    # print("----------")
    # print(df_combined.head(10))  # DataFrameの先頭10行を表示

    mfg = MonthlyFiguresGenerator()

    if plot_timeseries:
        df_combined_without_fleeze = df_combined.copy()
        freeze_mask = (
            (df_combined_without_fleeze.index >= "2024-09-02")
            & (df_combined_without_fleeze.index <= "2024-09-10")
        ) | (
            (df_combined_without_fleeze.index >= "2024-12-04")
            & (df_combined_without_fleeze.index <= "2024-12-17")
        )
        df_combined_without_fleeze.loc[freeze_mask, "CH4_ultra_cal"] = np.nan
        df_combined_without_fleeze.loc[freeze_mask, "C2H6_ultra_cal"] = np.nan
        df_combined_without_fleeze.loc[freeze_mask, "Fch4_ultra"] = np.nan
        df_combined_without_fleeze.loc[freeze_mask, "Fc2h6_ultra"] = np.nan
        mfg.plot_c1c2_concs_and_fluxes_timeseries(
            df=df_combined_without_fleeze,
            col_ch4_conc="CH4_ultra_cal",
            col_ch4_flux="Fch4_ultra",
            col_c2h6_conc="C2H6_ultra_cal",
            col_c2h6_flux="Fc2h6_ultra",
            output_dirpath=(os.path.join(output_dirpath, "timeseries")),
            # print_summary=False,
        )
        mfg.plot_c1c2_timeseries(
            df=df_combined_without_fleeze,
            output_dirpath=os.path.join(output_dirpath, "tests"),
            col_ch4_flux="Fch4_ultra",
            col_c2h6_flux="Fc2h6_ultra",
            ch4_ylim=(-1, 90),
            c2h6_ylim=(-0.1, 6),
            start_date="2024-05-15",
            end_date="2024-12-31",
            figsize=(20, 6),
        )
        mfg.logger.info("'timeseries'を作成しました。")

    # 単一月のデータ抽出も同様に変更
    for month, subplot_label in zip(months, subplot_labels, strict=True):
        month_str = f"{month:02d}"
        mfg.logger.info(f"{month_str}の処理を開始します。")

        # 月ごとのDataFrameを作成(extract_period_dataを使用)
        start_date = f"2024-{month:02d}-01"
        end_date = (
            f"2024-{month:02d}-{MonthlyConverter.get_last_day_of_month(2024, month)}"
        )
        df_month: pd.DataFrame = MonthlyConverter.extract_period_data(
            df=df_combined,
            start_date=start_date,
            end_date=end_date,
        )
        if month == 10 or month == 11 or month == 12:
            df_month["Fch4_open"] = np.nan

        if plot_diurnals:
            df_month_for_diurnanls = df_month.copy()
            conc_label: str | None = (
                "(a)"
                if month == 10
                else "(b)"
                if month == 11
                else "(c)"
                if month == 12
                else None
            )
            mfg.plot_diurnal_concentrations(
                df=df_month_for_diurnanls,
                output_dirpath=os.path.join(output_dirpath, "diurnal_conc"),
                output_filename=f"diurnal_conc-{month_str}.png",
                col_ch4_conc="CH4_ultra_cal",
                col_c2h6_conc="C2H6_ultra_cal",
                show_std=True,
                alpha_std=0.2,
                ch4_ylim=(1.9, 2.3),
                c2h6_ylim=(-3, 11),
                subplot_label_ch4=conc_label,
                # add_legend=True if month == 11 else False,
                add_legend=False,
                interval="30min",
                print_summary=False,
            )

            # flux_slope/conc_slope
            try:
                if month == 10 or month == 11 or month == 12:
                    df_month["gas_ratio_conc"] = df_month["slope"] / 0.076 * 100
                    df_month["gas_ratio_conc"] = np.where(
                        (df_month["gas_ratio_conc"] >= 0)
                        & (df_month["gas_ratio_conc"] <= 100),
                        df_month["gas_ratio_conc"],
                        np.nan,
                    )
                    df_month["gas_ratio_flux"] = (
                        df_month["Fc2h6_ultra"] / df_month["Fch4_ultra"] / 0.076 * 100
                    )
                    df_month["gas_ratio_flux"] = np.where(
                        (df_month["gas_ratio_flux"] >= 0)
                        & (df_month["gas_ratio_flux"] <= 100),
                        df_month["gas_ratio_flux"],
                        np.nan,
                    )
                    # 日変化パターン
                    mfg.plot_gas_ratio_diurnal(
                        df=df_month,
                        output_dirpath=os.path.join(output_dirpath, "gas_ratio"),
                        col_ratio_1="gas_ratio_conc",
                        col_ratio_2="gas_ratio_flux",
                        label_1="濃度勾配",
                        label_2="フラックス",
                        color_1="purple",
                        color_2="red",
                        output_filename=f"gas_ratio_diurnal-{month_str}.png",
                        subplot_label=conc_label,
                        subplot_fontsize=24,
                        y_max=100,
                        xlabel="Hour",
                        ylabel="都市ガス比率 (%)",
                        save_fig=True,
                        show_fig=False,
                    )
                    # 散布図
                    mfg.plot_scatter(
                        df=df_month,
                        col_x="gas_ratio_conc",
                        col_y="gas_ratio_flux",
                        xlabel="都市ガス比率 (濃度) (%)",
                        ylabel="都市ガス比率 (フラックス) (%)",
                        output_dirpath=(os.path.join(output_dirpath, "gas_ratio")),
                        output_filename=f"scatter-ultra_slopes-{month_str}.png",
                        x_axis_range=(0, 100),
                        y_axis_range=(0, 100),
                    )
            except Exception:
                pass

            # 日変化パターンを月ごとに作成
            mfg.plot_c1c2_fluxes_diurnal_patterns(
                df=df_month_for_diurnanls,
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
                subplot_label_ch4=subplot_label[0],
                subplot_label_c2h6=subplot_label[1],
                colors_ch4=["red", "black", "blue"],
                colors_c2h6=["orange"],
                output_dirpath=(os.path.join(output_dirpath, "diurnal")),
                output_filename=f"diurnal-{month_str}.png",  # タグ付けしたファイル名
                ax1_ylim=(-20, 150),
                ax2_ylim=(0, 6),
            )
            df_month_for_g2401_ultra = df_month_for_diurnanls.copy()
            # 2024-09-02から2024-09-10のデータをnanに設定
            mask = (df_month_for_g2401_ultra["Date"] >= "2024-09-02") & (
                df_month_for_g2401_ultra["Date"] <= "2024-09-10"
            )
            df_month_for_g2401_ultra.loc[mask] = np.nan
            mfg.plot_c1c2_fluxes_diurnal_patterns(
                df=df_month_for_g2401_ultra,
                y_cols_ch4=["Fch4_ultra", "Fch4_picaro"],
                y_cols_c2h6=["Fc2h6_ultra"],
                labels_ch4=["Ultra", "G2401"],
                labels_c2h6=["Ultra"],
                legend_only_ch4=True,
                add_label=True,
                # add_legend=True,
                # add_label=False,
                add_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                subplot_label_ch4=None,
                subplot_label_c2h6=None,
                colors_ch4=["red", "blue"],
                colors_c2h6=["orange"],
                output_dirpath=(os.path.join(output_dirpath, "diurnal")),
                output_filename=f"diurnal_g2401_ultra-{month_str}.png",  # タグ付けしたファイル名
                ax1_ylim=(-20, 150),
                ax2_ylim=(0, 6),
            )

            mfg.logger.info("'diurnals'を作成しました。")

        if plot_scatter:
            # c1c2 flux
            mfg.plot_scatter(
                df=df_month,
                col_x="Fch4_ultra",
                col_y="Fc2h6_ultra",
                xlabel=r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dirpath=(os.path.join(output_dirpath, "scatter")),
                output_filename=f"scatter-ultra_c1c2_f-{month_str}.png",
                x_axis_range=(-50, 400),
                y_axis_range=(-5, 25),
                show_fixed_slope=True,
            )
            try:
                # open_ultra
                mfg.plot_scatter(
                    df=df_month,
                    col_x="Fch4_open",
                    col_y="Fch4_ultra",
                    xlabel=r"Open Path CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                    ylabel=r"Ultra CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                    output_dirpath=(os.path.join(output_dirpath, "scatter")),
                    output_filename=f"scatter-open_ultra-{month_str}.png",
                    x_axis_range=(-50, 200),
                    y_axis_range=(-50, 200),
                )
            except Exception as e:
                print(e)

            # g2401_ultra
            mfg.plot_scatter(
                df=df_month,
                col_x="Fch4_picaro",
                col_y="Fch4_ultra",
                xlabel=r"G2401 CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                ylabel=r"Ultra CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                output_dirpath=(os.path.join(output_dirpath, "scatter")),
                output_filename=f"scatter-g2401_ultra-{month_str}.png",
                x_axis_range=(-50, 200),
                y_axis_range=(-50, 200),
            )

            mfg.logger.info("'scatters'を作成しました。")

        if plot_sources:
            mfg.plot_source_contributions_diurnal(
                df=df_month,
                output_dirpath=(os.path.join(output_dirpath, "sources")),
                output_filename=f"source_contributions-{month_str}.png",
                col_ch4_flux="Fch4_ultra",
                col_c2h6_flux="Fc2h6_ultra",
                y_max=110,
                subplot_label=subplot_label[0],
                subplot_fontsize=24,
                add_legend=False,
                show_fig=False,
            )
            mfg.plot_source_contributions_diurnal_with_slope(
                df=df_month,
                output_dirpath=(os.path.join(output_dirpath, "sources")),
                output_filename=f"source_contributions_with_slope-{month_str}.png",
                col_ch4_flux="Fch4_ultra",
                col_c2h6_flux="Fc2h6_ultra",
                col_slope="slope",
                y_max=110,
                # subplot_label=subplot_label[0],
                subplot_fontsize=24,
                show_fig=False,
                color_bio_conc="#00FFFF",
                color_gas_conc="#FF00FF",
            )
            mfg.logger.info("'sources'を作成しました。")

        if plot_comparison:
            # 開始日から1か月後の日付を計算
            start_date = f"2024-{month_str}-01"
            end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=1)).strftime(
                "%Y-%m-%d"
            )

            mfg.plot_fluxes_comparison(
                df=df_combined,
                output_dirpath=os.path.join(output_dirpath, "comparison"),
                output_filename=f"timeseries-g2401_ultra-{month_str}.png",
                cols_flux=["Fch4_picaro", "Fch4_ultra"],
                labels=["G2401", "Ultra"],
                colors=["blue", "red"],
                # ylim=(10, 60),
                start_date=start_date,
                end_date=end_date,
                include_end_date=False,
                show_ci=False,
                apply_ma=False,
                hourly_mean=True,
                x_interval="10days",  # "month"または"10days"を指定
                save_fig=True,
                show_fig=False,
            )

    # 2ヶ月毎
    months_two: list[list[int]] = [[5, 6], [7, 8], [9, 10], [11, 12]]
    subplot_labels_two: list[list[str | None]] = [
        ["(a)", None],
        ["(b)", None],
        ["(c)", None],
        ["(d)", None],
    ]
    for month_two, subplot_label in zip(months_two, subplot_labels_two, strict=True):
        # monthを0埋めのMM形式に変換
        month_str = "_".join(f"{month:02d}" for month in month_two)
        mfg.logger.info(f"{month_str}の処理を開始します。")

        # 月ごとのDataFrameを作成(extract_period_dataを使用)
        start_date = f"2024-{month_two[0]:02d}-01"
        end_date = f"2024-{month_two[1]:02d}-{MonthlyConverter.get_last_day_of_month(2024, month_two[1])}"
        df_month_two: pd.DataFrame = MonthlyConverter.extract_period_data(
            df=df_combined,
            start_date=start_date,
            end_date=end_date,
        )
        if month == 11 or month == 12:
            df_month_two["Fch4_open"] = np.nan

        if plot_diurnals:
            df_month_for_diurnanls = df_month_two.copy()
            # 日変化パターンを2ヶ月ごとに作成
            mfg.plot_c1c2_fluxes_diurnal_patterns_by_date(
                df=df_month_for_diurnanls,
                y_col_ch4="Fch4_ultra",
                y_col_c2h6="Fc2h6_ultra",
                plot_holiday=False,
                add_label=True,
                add_legend=False,
                subplot_fontsize=diurnal_subplot_fontsize,
                subplot_label_ch4=subplot_label[0],
                subplot_label_c2h6=subplot_label[1],
                output_dirpath=(os.path.join(output_dirpath, "diurnal_by_date")),
                output_filename=f"diurnal_by_date_two-{month_str}.png",
                ax1_ylim=(-20, 150),
                ax2_ylim=(-1, 8),
            )
            mfg.logger.info("'diurnals_by_date_two'を作成しました。")
            mfg.plot_source_contributions_diurnal_by_date(
                df=df_month_for_diurnanls,
                output_dirpath=(os.path.join(output_dirpath, "sources")),
                output_filename=f"source_contributions_by_date_two-{month_str}.png",
                col_ch4_flux="Fch4_ultra",
                col_c2h6_flux="Fc2h6_ultra",
                add_legend=False,
                # subplot_label_weekday=None,
                subplot_label_weekday=subplot_label[0],
                y_max=125,
            )
            mfg.logger.info("'source_contributions_by_date_two'を作成しました。")

        if plot_sources:
            setup_plot_params(font_size=32, tick_size=32)
            mfg.plot_source_contributions_diurnal_with_slope(
                df=df_month_two,
                output_dirpath=(os.path.join(output_dirpath, "sources")),
                output_filename=f"source_contributions_with_slope_two-{month_str}.png",
                col_ch4_flux="Fch4_ultra",
                col_c2h6_flux="Fc2h6_ultra",
                col_slope="slope",
                y_max=110,
                add_legend=False,
                figsize=(10, 8),
                # subplot_label=subplot_label[0],
                subplot_fontsize=24,
                show_fig=False,
                color_bio_conc="#00FFFF",
                color_gas_conc="#FF00FF",
            )
