import os
from logging import DEBUG, INFO, Logger
from pathlib import Path
from typing import Literal

import jpholiday
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullLocator
from scipy import linalg, stats

from ..commons.utilities import setup_logger


# 移動平均の計算関数
def calculate_rolling_stats(data: pd.Series, window: int, confidence_interval) -> tuple:
    """移動平均と信頼区間を計算する。

    Parameters
    ------
        data: pd.Series
            入力データ系列
        window: int
            移動平均の窓サイズ

    Returns
    ------
        tuple
            (移動平均, 下側信頼区間, 上側信頼区間)
    """
    # データ数が少なすぎる場合は警告
    if len(data) < window:
        window = len(data) // 4  # データ長の1/4を窓サイズとして使用
        raise ValueError(f"データ数が少ないため、窓サイズを{window}に調整しました")

    # 最小窓サイズの設定
    window = max(3, min(window, len(data)))

    # NaNを含むデータの処理(線形補間を行わない)
    data_cleaned = data.copy()

    # 移動平均の計算(NaNを含む場合はその期間の移動平均もNaNになる)
    rolling_mean = data_cleaned.rolling(
        window=window,
        center=True,
        min_periods=3,  # 最低3点あれば計算する
    ).mean()

    rolling_std = data_cleaned.rolling(window=window, center=True, min_periods=3).std()

    # 信頼区間の計算
    z_score = stats.norm.ppf((1 + confidence_interval) / 2)
    ci_lower = rolling_mean - z_score * rolling_std
    ci_upper = rolling_mean + z_score * rolling_std

    return rolling_mean, ci_lower, ci_upper


class MonthlyFiguresGenerator:
    def __init__(
        self,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ) -> None:
        """
        Monthlyシートから作成したDataFrameを解析して図を作成するクラス

        Parameters
        ------
            logger: Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug: bool
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

    def plot_c1c2_concs_and_fluxes_timeseries(
        self,
        df: pd.DataFrame,
        output_dirpath: str | Path | None = None,
        output_filename: str = "conc_flux_timeseries.png",
        col_datetime: str = "Date",
        col_ch4_conc: str = "CH4_ultra",
        col_ch4_flux: str = "Fch4_ultra",
        col_c2h6_conc: str = "C2H6_ultra",
        col_c2h6_flux: str = "Fc2h6_ultra",
        ylim_ch4_conc: tuple = (1.8, 2.6),
        ylim_ch4_flux: tuple = (-100, 600),
        ylim_c2h6_conc: tuple = (-12, 20),
        ylim_c2h6_flux: tuple = (-20, 40),
        figsize: tuple[float, float] = (12, 16),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
        print_summary: bool = False,
    ) -> None:
        """CH4とC2H6の濃度とフラックスの時系列プロットを作成します。

        Parameters
        ----------
            df: pd.DataFrame
                月別データを含むDataFrameを指定します。
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"conc_flux_timeseries.png"です。
            col_datetime: str, optional
                日付列の名前を指定します。デフォルト値は"Date"です。
            col_ch4_conc: str, optional
                CH4濃度列の名前を指定します。デフォルト値は"CH4_ultra"です。
            col_ch4_flux: str, optional
                CH4フラックス列の名前を指定します。デフォルト値は"Fch4_ultra"です。
            col_c2h6_conc: str, optional
                C2H6濃度列の名前を指定します。デフォルト値は"C2H6_ultra"です。
            col_c2h6_flux: str, optional
                C2H6フラックス列の名前を指定します。デフォルト値は"Fc2h6_ultra"です。
            ylim_ch4_conc: tuple, optional
                CH4濃度のy軸範囲を指定します。デフォルト値は(1.8, 2.6)です。
            ylim_ch4_flux: tuple, optional
                CH4フラックスのy軸範囲を指定します。デフォルト値は(-100, 600)です。
            ylim_c2h6_conc: tuple, optional
                C2H6濃度のy軸範囲を指定します。デフォルト値は(-12, 20)です。
            ylim_c2h6_flux: tuple, optional
                C2H6フラックスのy軸範囲を指定します。デフォルト値は(-20, 40)です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 16)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                図を保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                図を表示するかどうかを指定します。デフォルト値はTrueです。
            print_summary: bool, optional
                解析情報をprintするかどうかを指定します。デフォルト値はFalseです。

        Examples
        --------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_c1c2_concs_and_fluxes_timeseries(
        ...     df=monthly_data,
        ...     output_dirpath="output",
        ...     ylim_ch4_conc=(1.5, 3.0),
        ...     print_summary=True
        ... )
        """
        # dfのコピー
        df_internal: pd.DataFrame = df.copy()
        if print_summary:
            # 統計情報の計算と表示
            for name, col in [
                ("CH4 concentration", col_ch4_conc),
                ("CH4 flux", col_ch4_flux),
                ("C2H6 concentration", col_c2h6_conc),
                ("C2H6 flux", col_c2h6_flux),
            ]:
                # NaNを除外してから統計量を計算
                valid_data = df_internal[col].dropna()

                if len(valid_data) > 0:
                    # quantileで計算(0-1の範囲)
                    quantile_05 = valid_data.quantile(0.05)
                    quantile_95 = valid_data.quantile(0.95)
                    mean_value = np.nanmean(valid_data)
                    positive_ratio = (valid_data > 0).mean() * 100

                    print(f"\n{name}:")
                    # パーセンタイルで表示(0-100の範囲)
                    print(
                        f"90パーセンタイルレンジ: {quantile_05:.2f} - {quantile_95:.2f}"
                    )
                    print(f"平均値: {mean_value:.2f}")
                    print(f"正の値の割合: {positive_ratio:.1f}%")
                else:
                    print(f"\n{name}: データが存在しません")

        # プロットの作成
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)

        # CH4濃度のプロット
        ax1.scatter(
            df_internal[col_datetime],
            df_internal[col_ch4_conc],
            color="red",
            alpha=0.5,
            s=20,
        )
        ax1.set_ylabel("CH$_4$ Concentration\n(ppm)")
        ax1.set_ylim(*ylim_ch4_conc)  # 引数からy軸範囲を設定
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # CH4フラックスのプロット
        ax2.scatter(
            df_internal[col_datetime],
            df_internal[col_ch4_flux],
            color="red",
            alpha=0.5,
            s=20,
        )
        ax2.set_ylabel("CH$_4$ flux\n(nmol m$^{-2}$ s$^{-1}$)")
        ax2.set_ylim(*ylim_ch4_flux)  # 引数からy軸範囲を設定
        ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", fontsize=20)
        ax2.grid(True, alpha=0.3)

        # C2H6濃度のプロット
        ax3.scatter(
            df_internal[col_datetime],
            df_internal[col_c2h6_conc],
            color="orange",
            alpha=0.5,
            s=20,
        )
        ax3.set_ylabel("C$_2$H$_6$ Concentration\n(ppb)")
        ax3.set_ylim(*ylim_c2h6_conc)  # 引数からy軸範囲を設定
        ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, va="top", fontsize=20)
        ax3.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax4.scatter(
            df_internal[col_datetime],
            df_internal[col_c2h6_flux],
            color="orange",
            alpha=0.5,
            s=20,
        )
        ax4.set_ylabel("C$_2$H$_6$ flux\n(nmol m$^{-2}$ s$^{-1}$)")
        ax4.set_ylim(*ylim_c2h6_flux)  # 引数からy軸範囲を設定
        ax4.text(0.02, 0.98, "(d)", transform=ax4.transAxes, va="top", fontsize=20)
        ax4.grid(True, alpha=0.3)

        # x軸の設定
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        plt.setp(ax4.get_xticklabels(), rotation=0, ha="right")
        ax4.set_xlabel("Month")

        # レイアウトの調整と保存
        plt.tight_layout()

        # 図の保存
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close()

        if print_summary:

            def analyze_top_values(df, column_name, top_percent=20):
                print(f"\n{column_name}の上位{top_percent}%の分析:")
                # DataFrameのコピーを作成し、日時関連の列を追加
                df_analysis = df.copy()
                df_analysis["hour"] = pd.to_datetime(df_analysis[col_datetime]).dt.hour
                df_analysis["month"] = pd.to_datetime(
                    df_analysis[col_datetime]
                ).dt.month
                df_analysis["weekday"] = pd.to_datetime(
                    df_analysis[col_datetime]
                ).dt.dayofweek

                # 上位20%のしきい値を計算
                threshold = df_analysis[column_name].quantile(1 - top_percent / 100)
                high_values = df_analysis[df_analysis[column_name] > threshold]

                # 月ごとの分析
                print("\n月別分布:")
                monthly_counts = high_values.groupby("month").size()
                total_counts = df_analysis.groupby("month").size()
                monthly_percentages = (monthly_counts / total_counts * 100).round(1)

                # 月ごとのデータを安全に表示
                available_months = set(monthly_counts.index) & set(total_counts.index)
                for month in sorted(available_months):
                    print(
                        f"月{month}: {monthly_percentages[month]}% ({monthly_counts[month]}件/{total_counts[month]}件)"
                    )

                # 時間帯ごとの分析(3時間区切り)
                print("\n時間帯別分布:")
                # copyを作成して新しい列を追加
                high_values = high_values.copy()
                high_values["time_block"] = high_values["hour"] // 3 * 3
                time_blocks = high_values.groupby("time_block").size()
                total_time_blocks = df_analysis.groupby(
                    df_analysis["hour"] // 3 * 3
                ).size()
                time_percentages = (time_blocks / total_time_blocks * 100).round(1)

                # 時間帯ごとのデータを安全に表示
                available_blocks = set(time_blocks.index) & set(total_time_blocks.index)
                for block in sorted(available_blocks):
                    print(
                        f"{block:02d}:00-{block + 3:02d}:00: {time_percentages[block]}% ({time_blocks[block]}件/{total_time_blocks[block]}件)"
                    )

                # 曜日ごとの分析
                print("\n曜日別分布:")
                weekday_names = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
                weekday_counts = high_values.groupby("weekday").size()
                total_weekdays = df_analysis.groupby("weekday").size()
                weekday_percentages = (weekday_counts / total_weekdays * 100).round(1)

                # 曜日ごとのデータを安全に表示
                available_days = set(weekday_counts.index) & set(total_weekdays.index)
                for day in sorted(available_days):
                    if 0 <= day <= 6:  # 有効な曜日インデックスのチェック
                        print(
                            f"{weekday_names[day]}: {weekday_percentages[day]}% ({weekday_counts[day]}件/{total_weekdays[day]}件)"
                        )

            # 濃度とフラックスそれぞれの分析を実行
            print("\n=== 上位値の時間帯・曜日分析 ===")
            analyze_top_values(df_internal, col_ch4_conc)
            analyze_top_values(df_internal, col_ch4_flux)
            analyze_top_values(df_internal, col_c2h6_conc)
            analyze_top_values(df_internal, col_c2h6_flux)

    def plot_c1c2_timeseries(
        self,
        df: pd.DataFrame,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        output_dirpath: str | Path | None = None,
        output_filename: str = "timeseries_year.png",
        col_datetime: str = "Date",
        window_size: int = 24 * 7,
        confidence_interval: float = 0.95,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        subplot_fontsize: int = 20,
        show_ci: bool = True,
        ch4_ylim: tuple[float, float] | None = None,
        c2h6_ylim: tuple[float, float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        figsize: tuple[float, float] = (16, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4とC2H6フラックスの時系列変動をプロットします。

        Parameters
        ----------
            df: pd.DataFrame
                プロットするデータを含むDataFrameを指定します。
            col_ch4_flux: str
                CH4フラックスのカラム名を指定します。
            col_c2h6_flux: str
                C2H6フラックスのカラム名を指定します。
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"timeseries_year.png"です。
            col_datetime: str, optional
                日時カラムの名前を指定します。デフォルト値は"Date"です。
            window_size: int, optional
                移動平均の窓サイズを指定します。デフォルト値は24*7(1週間)です。
            confidence_interval: float, optional
                信頼区間を指定します。0から1の間の値で、デフォルト値は0.95(95%)です。
            subplot_label_ch4: str | None, optional
                CH4プロットのラベルを指定します。デフォルト値は"(a)"です。
            subplot_label_c2h6: str | None, optional
                C2H6プロットのラベルを指定します。デフォルト値は"(b)"です。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            show_ci: bool, optional
                信頼区間を表示するかどうかを指定します。デフォルト値はTrueです。
            ch4_ylim: tuple[float, float] | None, optional
                CH4のy軸範囲を指定します。未指定の場合は自動で設定されます。
            c2h6_ylim: tuple[float, float] | None, optional
                C2H6のy軸範囲を指定します。未指定の場合は自動で設定されます。
            start_date: str | None, optional
                開始日を"YYYY-MM-DD"形式で指定します。未指定の場合はデータの最初の日付が使用されます。
            end_date: str | None, optional
                終了日を"YYYY-MM-DD"形式で指定します。未指定の場合はデータの最後の日付が使用されます。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(16, 6)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_c1c2_timeseries(
        ...     df=monthly_data,
        ...     col_ch4_flux="Fch4_ultra",
        ...     col_c2h6_flux="Fc2h6_ultra",
        ...     output_dirpath="output",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
        """
        # データの準備
        df_internal = df.copy()
        if not isinstance(df_internal.index, pd.DatetimeIndex):
            df_internal[col_datetime] = pd.to_datetime(df_internal[col_datetime])
            df_internal.set_index(col_datetime, inplace=True)

        # 日付範囲の処理
        if start_date is not None:
            start_dt = pd.to_datetime(start_date).normalize()  # 時刻を00:00:00に設定
            # df_min_date = (
            #     df_internal.index.normalize().min().normalize()
            # )  # 日付のみの比較のため正規化
            df_min_date = pd.to_datetime(df_internal.index.min()).normalize()

            # データの最小日付が指定開始日より後の場合にのみ警告
            if df_min_date.date() > start_dt.date():
                self.logger.warning(
                    f"指定された開始日{start_date}がデータの開始日{df_min_date.strftime('%Y-%m-%d')}より前です。"
                    f"データの開始日を使用します。"
                )
                start_dt = df_min_date
        else:
            # start_dt = df_internal.index.normalize().min()
            start_dt = pd.to_datetime(df_internal.index.min()).normalize()

        if end_date is not None:
            end_dt = (
                pd.to_datetime(end_date).normalize()
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            # df_max_date = (
            #     df_internal.index.normalize().max().normalize()
            # )  # 日付のみの比較のため正規化
            df_max_date = pd.to_datetime(df_internal.index.max()).normalize()

            # データの最大日付が指定終了日より前の場合にのみ警告
            if df_max_date.date() < pd.to_datetime(end_date).date():
                self.logger.warning(
                    f"指定された終了日{end_date}がデータの終了日{df_max_date.strftime('%Y-%m-%d')}より後です。"
                    f"データの終了日を使用します。"
                )
                end_dt = df_internal.index.max()
        else:
            end_dt = df_internal.index.max()

        # 指定された期間のデータを抽出
        mask = (df_internal.index >= start_dt) & (df_internal.index <= end_dt)
        df_internal = df_internal[mask]

        # CH4とC2H6の移動平均と信頼区間を計算
        ch4_mean, ch4_lower, ch4_upper = calculate_rolling_stats(
            df_internal[col_ch4_flux], window_size, confidence_interval
        )
        c2h6_mean, c2h6_lower, c2h6_upper = calculate_rolling_stats(
            df_internal[col_c2h6_flux], window_size, confidence_interval
        )

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CH4プロット
        ax1.plot(df_internal.index, ch4_mean, "red", label="CH$_4$")
        if show_ci:
            ax1.fill_between(
                df_internal.index, ch4_lower, ch4_upper, color="red", alpha=0.2
            )
        if subplot_label_ch4:
            ax1.text(
                0.02,
                0.98,
                subplot_label_ch4,
                transform=ax1.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )
        ax1.set_ylabel("CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        if ch4_ylim is not None:
            ax1.set_ylim(ch4_ylim)
        ax1.grid(True, alpha=0.3)

        # C2H6プロット
        ax2.plot(df_internal.index, c2h6_mean, "orange", label="C$_2$H$_6$")
        if show_ci:
            ax2.fill_between(
                df_internal.index, c2h6_lower, c2h6_upper, color="orange", alpha=0.2
            )
        if subplot_label_c2h6:
            ax2.text(
                0.02,
                0.98,
                subplot_label_c2h6,
                transform=ax2.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )
        ax2.set_ylabel("C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)")
        if c2h6_ylim is not None:
            ax2.set_ylim(c2h6_ylim)
        ax2.grid(True, alpha=0.3)

        # x軸の設定
        for ax in [ax1, ax2]:
            ax.set_xlabel("Month")
            # x軸の範囲を設定
            ax.set_xlim(start_dt, end_dt)

            # 1ヶ月ごとの主目盛り
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            # カスタムフォーマッタの作成(数字を通常フォントで表示)
            def date_formatter(x, p):
                date = mdates.num2date(x)
                return f"{date.strftime('%m')}"

            ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))

            # 補助目盛りの設定
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            # ティックラベルの回転と位置調整
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

        plt.tight_layout()

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_fluxes_comparison(
        self,
        df: pd.DataFrame,
        cols_flux: list[str],
        labels: list[str],
        colors: list[str],
        output_dirpath: str | Path | None,
        output_filename: str = "ch4_flux_comparison.png",
        col_datetime: str = "Date",
        window_size: int = 24 * 7,
        confidence_interval: float = 0.95,
        subplot_label: str | None = None,
        subplot_fontsize: int = 20,
        show_ci: bool = True,
        y_lim: tuple[float, float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_end_date: bool = True,
        legend_loc: str = "upper right",
        apply_ma: bool = True,
        hourly_mean: bool = False,
        x_interval: Literal["month", "10days"] = "month",
        xlabel: str = "Month",
        ylabel: str = "CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
        figsize: tuple[float, float] = (12, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """複数のCH4フラックスの時系列変動を比較するプロットを作成します。

        Parameters
        ----------
            df: pd.DataFrame
                プロットするデータを含むDataFrameを指定します。
            cols_flux: list[str]
                比較するフラックスのカラム名のリストを指定します。
            labels: list[str]
                凡例に表示する各フラックスのラベルのリストを指定します。
            colors: list[str]
                各フラックスの色のリストを指定します。
            output_dirpath: str | Path | None
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"ch4_flux_comparison.png"です。
            col_datetime: str, optional
                日時カラムの名前を指定します。デフォルト値は"Date"です。
            window_size: int, optional
                移動平均の窓サイズを指定します。デフォルト値は24*7(1週間)です。
            confidence_interval: float, optional
                信頼区間を指定します。0から1の間の値で、デフォルト値は0.95(95%)です。
            subplot_label: str | None, optional
                プロットのラベルを指定します。デフォルト値はNoneです。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            show_ci: bool, optional
                信頼区間を表示するかどうかを指定します。デフォルト値はTrueです。
            y_lim: tuple[float, float] | None, optional
                y軸の範囲を指定します。デフォルト値はNoneです。
            start_date: str | None, optional
                開始日をYYYY-MM-DD形式で指定します。デフォルト値はNoneです。
            end_date: str | None, optional
                終了日をYYYY-MM-DD形式で指定します。デフォルト値はNoneです。
            include_end_date: bool, optional
                終了日を含めるかどうかを指定します。デフォルト値はTrueです。
            legend_loc: str, optional
                凡例の位置を指定します。デフォルト値は"upper right"です。
            apply_ma: bool, optional
                移動平均を適用するかどうかを指定します。デフォルト値はTrueです。
            hourly_mean: bool, optional
                1時間平均を適用するかどうかを指定します。デフォルト値はFalseです。
            x_interval: Literal["month", "10days"], optional
                x軸の目盛り間隔を指定します。"month"(月初めのみ)または"10days"(10日刻み)を指定できます。デフォルト値は"month"です。
            xlabel: str, optional
                x軸のラベルを指定します。デフォルト値は"Month"です。
            ylabel: str, optional
                y軸のラベルを指定します。デフォルト値は"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)"です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 6)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                図を保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                図を表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        -------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_fluxes_comparison(
        ...     df=monthly_data,
        ...     cols_flux=["Fch4_ultra", "Fch4_picarro"],
        ...     labels=["Ultra", "Picarro"],
        ...     colors=["red", "blue"],
        ...     output_dirpath="output",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
        """
        # データの準備
        df_internal = df.copy()

        # インデックスを日時型に変換
        df_internal.index = pd.to_datetime(df_internal.index)

        # 1時間平均の適用
        if hourly_mean:
            # 時間情報のみを使用してグループ化
            df_internal = df_internal.groupby(
                [df_internal.index.strftime("%Y-%m-%d"), df_internal.index.hour]
            ).mean()
            # マルチインデックスを日時インデックスに変換
            df_internal.index = pd.to_datetime(
                [f"{date} {hour:02d}:00:00" for date, hour in df_internal.index]
            )

        # 日付範囲の処理
        if start_date is not None:
            start_dt = pd.to_datetime(start_date).normalize()  # 時刻を00:00:00に設定
            df_min_date = (
                df_internal.index.normalize().min().normalize()
            )  # 日付のみの比較のため正規化

            # データの最小日付が指定開始日より後の場合にのみ警告
            if df_min_date.date() > start_dt.date():
                self.logger.warning(
                    f"指定された開始日{start_date}がデータの開始日{df_min_date.strftime('%Y-%m-%d')}より前です。"
                    f"データの開始日を使用します。"
                )
                start_dt = df_min_date
        else:
            start_dt = df_internal.index.normalize().min()

        if end_date is not None:
            if include_end_date:
                end_dt = (
                    pd.to_datetime(end_date).normalize()
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )
            else:
                # 終了日を含まない場合、終了日の前日の23:59:59まで
                end_dt = pd.to_datetime(end_date).normalize() - pd.Timedelta(seconds=1)

            df_max_date = (
                df_internal.index.normalize().max().normalize()
            )  # 日付のみの比較のため正規化

            # データの最大日付が指定終了日より前の場合にのみ警告
            compare_date = pd.to_datetime(end_date).date()
            if not include_end_date:
                compare_date = compare_date - pd.Timedelta(days=1)

            if df_max_date.date() < compare_date:
                self.logger.warning(
                    f"指定された終了日{end_date}がデータの終了日{df_max_date.strftime('%Y-%m-%d')}より後です。"
                    f"データの終了日を使用します。"
                )
                end_dt = df_internal.index.max()
        else:
            end_dt = df_internal.index.max()

        # 指定された期間のデータを抽出
        mask = (df_internal.index >= start_dt) & (df_internal.index <= end_dt)
        df_internal = df_internal[mask]

        # プロットの作成
        fig, ax = plt.subplots(figsize=figsize)

        # 各フラックスのプロット
        for flux_col, label, color in zip(cols_flux, labels, colors, strict=True):
            if apply_ma:
                # 移動平均の計算
                mean, lower, upper = calculate_rolling_stats(
                    df_internal[flux_col], window_size, confidence_interval
                )
                ax.plot(df_internal.index, mean, color, label=label, alpha=0.7)
                if show_ci:
                    ax.fill_between(
                        df_internal.index, lower, upper, color=color, alpha=0.2
                    )
            else:
                # 生データのプロット
                ax.plot(
                    df_internal.index,
                    df_internal[flux_col],
                    color,
                    label=label,
                    alpha=0.7,
                )

        # プロットの設定
        if subplot_label:
            ax.text(
                0.02,
                0.98,
                subplot_label,
                transform=ax.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_loc)

        # x軸の設定
        ax.set_xlim(float(mdates.date2num(start_dt)), float(mdates.date2num(end_dt)))

        if x_interval == "month":
            # 月初めにメジャー線のみ表示
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(NullLocator())  # マイナー線を非表示
        elif x_interval == "10days":
            # 月初め(1日)、10日、20日、30日に目盛りを表示
            class Custom10DayLocator(mdates.DateLocator):
                def __call__(self):
                    dmin, dmax = self.viewlim_to_dt()
                    dates = []
                    current = pd.to_datetime(dmin).normalize()
                    end = pd.to_datetime(dmax).normalize()

                    while current <= end:
                        # その月の1日、10日、20日、30日を追加
                        for day in [1, 10, 20, 30]:
                            try:
                                date = current.replace(day=day)
                                if dmin <= date <= dmax:
                                    dates.append(date)
                            except ValueError:
                                # 30日が存在しない月(2月など)の場合は
                                # その月の最終日を使用
                                if day == 30:
                                    last_day = (
                                        current + pd.DateOffset(months=1)
                                    ).replace(day=1) - pd.Timedelta(days=1)
                                    if dmin <= last_day <= dmax:
                                        dates.append(last_day)

                        # 次の月へ
                        current = (current + pd.DateOffset(months=1)).replace(day=1)

                    return self.raise_if_exceeds(
                        [float(mdates.date2num(date)) for date in dates]
                    )

            ax.xaxis.set_major_locator(Custom10DayLocator())
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            ax.grid(True, which="minor", alpha=0.1)

        # カスタムフォーマッタの作成
        def date_formatter(x, p):
            date = mdates.num2date(x)
            if x_interval == "month":
                # 月初めの1日の場合のみ月を表示
                if date.day == 1:
                    return f"{date.strftime('%m')}"
                return ""
            else:  # "10days"の場合
                # MM/DD形式で表示し、/を中心に配置
                month = f"{date.strftime('%m'):>2}"  # 右寄せで2文字
                day = f"{date.strftime('%d'):<2}"  # 左寄せで2文字
                return f"{month}/{day}"

        ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))
        plt.setp(
            ax.xaxis.get_majorticklabels(), ha="center", rotation=0
        )  # 中央揃えに変更

        plt.tight_layout()

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_c1c2_fluxes_diurnal_patterns(
        self,
        df: pd.DataFrame,
        y_cols_ch4: list[str],
        y_cols_c2h6: list[str],
        labels_ch4: list[str],
        labels_c2h6: list[str],
        colors_ch4: list[str],
        colors_c2h6: list[str],
        output_dirpath: str | Path | None = None,
        output_filename: str = "diurnal.png",
        legend_only_ch4: bool = False,
        add_label: bool = True,
        add_legend: bool = True,
        show_std: bool = False,
        std_alpha: float = 0.2,
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        ax1_ylim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4とC2H6の日変化パターンを1つの図に並べてプロットします。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレームを指定します。
            y_cols_ch4: list[str]
                CH4のプロットに使用するカラム名のリストを指定します。
            y_cols_c2h6: list[str]
                C2H6のプロットに使用するカラム名のリストを指定します。
            labels_ch4: list[str]
                CH4の各ラインに対応するラベルのリストを指定します。
            labels_c2h6: list[str]
                C2H6の各ラインに対応するラベルのリストを指定します。
            colors_ch4: list[str]
                CH4の各ラインに使用する色のリストを指定します。
            colors_c2h6: list[str]
                C2H6の各ラインに使用する色のリストを指定します。
            output_dirpath: str | Path | None, optional
                出力先ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"diurnal.png"です。
            legend_only_ch4: bool, optional
                CH4の凡例のみを表示するかどうかを指定します。デフォルト値はFalseです。
            add_label: bool, optional
                サブプロットラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_legend: bool, optional
                凡例を表示するかどうかを指定します。デフォルト値はTrueです。
            show_std: bool, optional
                標準偏差を表示するかどうかを指定します。デフォルト値はFalseです。
            std_alpha: float, optional
                標準偏差の透明度を指定します。デフォルト値は0.2です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            subplot_label_ch4: str | None, optional
                CH4プロットのラベルを指定します。デフォルト値は"(a)"です。
            subplot_label_c2h6: str | None, optional
                C2H6プロットのラベルを指定します。デフォルト値は"(b)"です。
            ax1_ylim: tuple[float, float] | None, optional
                CH4プロットのy軸の範囲を指定します。デフォルト値はNoneです。
            ax2_ylim: tuple[float, float] | None, optional
                C2H6プロットのy軸の範囲を指定します。デフォルト値はNoneです。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_c1c2_fluxes_diurnal_patterns(
        ...     df=monthly_data,
        ...     y_cols_ch4=["CH4_flux1", "CH4_flux2"],
        ...     y_cols_c2h6=["C2H6_flux1", "C2H6_flux2"],
        ...     labels_ch4=["CH4 1", "CH4 2"],
        ...     labels_c2h6=["C2H6 1", "C2H6 2"],
        ...     colors_ch4=["red", "blue"],
        ...     colors_c2h6=["green", "orange"],
        ...     output_dirpath="output",
        ...     show_std=True
        ... )
        """
        # データの準備
        df_internal: pd.DataFrame = df.copy()
        df_internal.index = pd.to_datetime(df_internal.index)
        target_columns = y_cols_ch4 + y_cols_c2h6
        hourly_means, time_points = self._prepare_diurnal_data(
            df_internal, target_columns
        )

        # 標準偏差の計算を追加
        hourly_stds = {}
        if show_std:
            hourly_stds = df_internal.groupby(df_internal.index.hour)[
                target_columns
            ].std()
            # 24時間目のデータ点を追加
            last_hour = hourly_stds.iloc[0:1].copy()
            last_hour.index = pd.Index([24])
            hourly_stds = pd.concat([hourly_stds, last_hour])

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CH4のプロット (左側)
        ch4_lines = []
        for col_y, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4, strict=True):
            mean_values = hourly_means["all"][col_y]
            line = ax1.plot(
                time_points,
                mean_values,
                "-o",
                label=label,
                color=color,
            )
            ch4_lines.extend(line)

            # 標準偏差の表示
            if show_std:
                std_values = hourly_stds[col_y]
                ax1.fill_between(
                    time_points,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=std_alpha,
                )

        # C2H6のプロット (右側)
        c2h6_lines = []
        for col_y, label, color in zip(
            y_cols_c2h6, labels_c2h6, colors_c2h6, strict=True
        ):
            mean_values = hourly_means["all"][col_y]
            line = ax2.plot(
                time_points,
                mean_values,
                "o-",
                label=label,
                color=color,
            )
            c2h6_lines.extend(line)

            # 標準偏差の表示
            if show_std:
                std_values = hourly_stds[col_y]
                ax2.fill_between(
                    time_points,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=std_alpha,
                )

        # 軸の設定
        for ax, ylabel, subplot_label in [
            (ax1, r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_ch4),
            (ax2, r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_c2h6),
        ]:
            self._setup_diurnal_axes(
                ax=ax,
                time_points=time_points,
                ylabel=ylabel,
                subplot_label=subplot_label,
                add_label=add_label,
                add_legend=False,  # 個別の凡例は表示しない
                subplot_fontsize=subplot_fontsize,
            )

        if ax1_ylim is not None:
            ax1.set_ylim(ax1_ylim)
        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))

        if ax2_ylim is not None:
            ax2.set_ylim(ax2_ylim)
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        plt.tight_layout()

        # 共通の凡例
        if add_legend:
            all_lines = ch4_lines
            all_labels = [line.get_label() for line in ch4_lines]
            if not legend_only_ch4:
                all_lines += c2h6_lines
                all_labels += [line.get_label() for line in c2h6_lines]
            fig.legend(
                all_lines,
                all_labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(all_lines),
            )
            plt.subplots_adjust(bottom=0.25)  # 下部に凡例用のスペースを確保

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_c1c2_fluxes_diurnal_patterns_by_date(
        self,
        df: pd.DataFrame,
        y_col_ch4: str,
        y_col_c2h6: str,
        output_dirpath: str | Path | None = None,
        output_filename: str = "diurnal_by_date.png",
        plot_all: bool = True,
        plot_weekday: bool = True,
        plot_weekend: bool = True,
        plot_holiday: bool = True,
        add_label: bool = True,
        add_legend: bool = True,
        show_std: bool = False,
        std_alpha: float = 0.2,
        legend_only_ch4: bool = False,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        ax1_ylim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
        print_summary: bool = False,
    ) -> None:
        """CH4とC2H6の日変化パターンを日付分類して1つの図に並べてプロットします。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレームを指定します。
            y_col_ch4: str
                CH4フラックスを含むカラム名を指定します。
            y_col_c2h6: str
                C2H6フラックスを含むカラム名を指定します。
            output_dirpath: str | Path | None, optional
                出力先ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"diurnal_by_date.png"です。
            plot_all: bool, optional
                すべての日をプロットするかどうかを指定します。デフォルト値はTrueです。
            plot_weekday: bool, optional
                平日をプロットするかどうかを指定します。デフォルト値はTrueです。
            plot_weekend: bool, optional
                週末をプロットするかどうかを指定します。デフォルト値はTrueです。
            plot_holiday: bool, optional
                祝日をプロットするかどうかを指定します。デフォルト値はTrueです。
            add_label: bool, optional
                サブプロットラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_legend: bool, optional
                凡例を表示するかどうかを指定します。デフォルト値はTrueです。
            show_std: bool, optional
                標準偏差を表示するかどうかを指定します。デフォルト値はFalseです。
            std_alpha: float, optional
                標準偏差の透明度を指定します。デフォルト値は0.2です。
            legend_only_ch4: bool, optional
                CH4の凡例のみを表示するかどうかを指定します。デフォルト値はFalseです。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            subplot_label_ch4: str | None, optional
                CH4プロットのラベルを指定します。デフォルト値は"(a)"です。
            subplot_label_c2h6: str | None, optional
                C2H6プロットのラベルを指定します。デフォルト値は"(b)"です。
            ax1_ylim: tuple[float, float] | None, optional
                CH4プロットのy軸の範囲を指定します。デフォルト値はNoneです。
            ax2_ylim: tuple[float, float] | None, optional
                C2H6プロットのy軸の範囲を指定します。デフォルト値はNoneです。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。
            print_summary: bool, optional
                統計情報を表示するかどうかを指定します。デフォルト値はFalseです。

        Examples
        -------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_c1c2_fluxes_diurnal_patterns_by_date(
        ...     df=monthly_data,
        ...     y_col_ch4="CH4_flux",
        ...     y_col_c2h6="C2H6_flux",
        ...     output_dirpath="output",
        ...     show_std=True,
        ...     print_summary=True
        ... )
        """
        # データの準備
        df_internal: pd.DataFrame = df.copy()
        df_internal.index = pd.to_datetime(df_internal.index)
        target_columns = [y_col_ch4, y_col_c2h6]
        hourly_means, time_points = self._prepare_diurnal_data(
            df_internal, target_columns, include_date_types=True
        )

        # 標準偏差の計算を追加
        hourly_stds = {}
        if show_std:
            for condition in ["all", "weekday", "weekend", "holiday"]:
                if condition == "all":
                    condition_data = df_internal
                elif condition == "weekday":
                    condition_data = df_internal[
                        ~(
                            df_internal.index.dayofweek.isin([5, 6])
                            | df_internal.index.map(
                                lambda x: jpholiday.is_holiday(x.date())
                            )
                        )
                    ]
                elif condition == "weekend":
                    condition_data = df_internal[
                        df_internal.index.dayofweek.isin([5, 6])
                    ]
                else:  # holiday
                    condition_data = df_internal[
                        df_internal.index.map(lambda x: jpholiday.is_holiday(x.date()))
                    ]

                hourly_stds[condition] = condition_data.groupby(
                    pd.to_datetime(condition_data.index).hour
                )[target_columns].std()
                # 24時間目のデータ点を追加
                last_hour = hourly_stds[condition].iloc[0:1].copy()
                last_hour.index = [24]
                hourly_stds[condition] = pd.concat([hourly_stds[condition], last_hour])

        # プロットスタイルの設定
        styles = {
            "all": {
                "color": "black",
                "linestyle": "-",
                "alpha": 1.0,
                "label": "All days",
            },
            "weekday": {
                "color": "blue",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekdays",
            },
            "weekend": {
                "color": "red",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekends",
            },
            "holiday": {
                "color": "green",
                "linestyle": "-",
                "alpha": 0.8,
                "label": "Weekends & Holidays",
            },
        }

        # プロット対象の条件を選択
        plot_conditions = {
            "all": plot_all,
            "weekday": plot_weekday,
            "weekend": plot_weekend,
            "holiday": plot_holiday,
        }
        selected_conditions = {
            col: means
            for col, means in hourly_means.items()
            if plot_conditions.get(col)
        }

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CH4とC2H6のプロット用のラインオブジェクトを保存
        ch4_lines = []
        c2h6_lines = []

        # CH4とC2H6のプロット
        for condition, means in selected_conditions.items():
            style = styles[condition].copy()

            # CH4プロット
            mean_values_ch4 = means[y_col_ch4]
            line_ch4 = ax1.plot(time_points, mean_values_ch4, marker="o", **style)
            ch4_lines.extend(line_ch4)

            if show_std and condition in hourly_stds:
                std_values = hourly_stds[condition][y_col_ch4]
                ax1.fill_between(
                    time_points,
                    mean_values_ch4 - std_values,
                    mean_values_ch4 + std_values,
                    color=style["color"],
                    alpha=std_alpha,
                )

            # C2H6プロット
            style["linestyle"] = "--"
            mean_values_c2h6 = means[y_col_c2h6]
            line_c2h6 = ax2.plot(time_points, mean_values_c2h6, marker="o", **style)
            c2h6_lines.extend(line_c2h6)

            if show_std and condition in hourly_stds:
                std_values = hourly_stds[condition][y_col_c2h6]
                ax2.fill_between(
                    time_points,
                    mean_values_c2h6 - std_values,
                    mean_values_c2h6 + std_values,
                    color=style["color"],
                    alpha=std_alpha,
                )

        # 軸の設定
        for ax, ylabel, subplot_label in [
            (ax1, r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_ch4),
            (ax2, r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)", subplot_label_c2h6),
        ]:
            self._setup_diurnal_axes(
                ax=ax,
                time_points=time_points,
                ylabel=ylabel,
                subplot_label=subplot_label,
                add_label=add_label,
                add_legend=False,
                subplot_fontsize=subplot_fontsize,
            )

        if ax1_ylim is not None:
            ax1.set_ylim(ax1_ylim)
        ax1.yaxis.set_major_locator(MultipleLocator(20))
        ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.0f}"))

        if ax2_ylim is not None:
            ax2.set_ylim(ax2_ylim)
        ax2.yaxis.set_major_locator(MultipleLocator(1))
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:.1f}"))

        plt.tight_layout()

        # 共通の凡例を図の下部に配置
        if add_legend:
            lines_to_show = (
                ch4_lines if legend_only_ch4 else ch4_lines[: len(selected_conditions)]
            )
            fig.legend(
                lines_to_show,
                [
                    style["label"]
                    for style in list(styles.values())[: len(lines_to_show)]
                ],
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=len(lines_to_show),
            )
            plt.subplots_adjust(bottom=0.25)  # 下部に凡例用のスペースを確保

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        # 日変化パターンの統計分析を追加
        if print_summary:
            # 平日と休日のデータを準備
            dates = pd.to_datetime(df_internal.index)
            is_weekend = dates.dayofweek.isin([5, 6])
            is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
            is_weekday = ~(is_weekend | is_holiday)

            weekday_data = df_internal[is_weekday]
            holiday_data = df_internal[is_weekend | is_holiday]

            def get_diurnal_stats(data, column):
                # 時間ごとの平均値を計算
                hourly_means = data.groupby(data.index.hour)[column].mean()

                # 8-16時の時間帯の統計
                daytime_means = hourly_means[
                    (hourly_means.index >= 8) & (hourly_means.index <= 16)
                ]

                if len(daytime_means) == 0:
                    return None

                return {
                    "mean": daytime_means.mean(),
                    "max": daytime_means.max(),
                    "max_hour": daytime_means.idxmax(),
                    "min": daytime_means.min(),
                    "min_hour": daytime_means.idxmin(),
                    "hours_count": len(daytime_means),
                }

            # CH4とC2H6それぞれの統計を計算
            for col, gas_name in [(y_col_ch4, "CH4"), (y_col_c2h6, "C2H6")]:
                print(f"\n=== {gas_name} フラックス 8-16時の統計分析 ===")

                weekday_stats = get_diurnal_stats(weekday_data, col)
                holiday_stats = get_diurnal_stats(holiday_data, col)

                if weekday_stats and holiday_stats:
                    print("\n平日:")
                    print(f"  平均値: {weekday_stats['mean']:.2f}")
                    print(
                        f"  最大値: {weekday_stats['max']:.2f} ({weekday_stats['max_hour']}時)"
                    )
                    print(
                        f"  最小値: {weekday_stats['min']:.2f} ({weekday_stats['min_hour']}時)"
                    )
                    print(f"  集計時間数: {weekday_stats['hours_count']}")

                    print("\n休日:")
                    print(f"  平均値: {holiday_stats['mean']:.2f}")
                    print(
                        f"  最大値: {holiday_stats['max']:.2f} ({holiday_stats['max_hour']}時)"
                    )
                    print(
                        f"  最小値: {holiday_stats['min']:.2f} ({holiday_stats['min_hour']}時)"
                    )
                    print(f"  集計時間数: {holiday_stats['hours_count']}")

                    # 平日/休日の比率を計算
                    print("\n平日/休日の比率:")
                    print(
                        f"  平均値比: {weekday_stats['mean'] / holiday_stats['mean']:.2f}"
                    )
                    print(
                        f"  最大値比: {weekday_stats['max'] / holiday_stats['max']:.2f}"
                    )
                    print(
                        f"  最小値比: {weekday_stats['min'] / holiday_stats['min']:.2f}"
                    )
                else:
                    print("十分なデータがありません")

    def plot_diurnal_concentrations(
        self,
        df: pd.DataFrame,
        col_ch4_conc: str = "CH4_ultra_cal",
        col_c2h6_conc: str = "C2H6_ultra_cal",
        col_datetime: str = "Date",
        output_dirpath: str | Path | None = None,
        output_filename: str = "diurnal_concentrations.png",
        show_std: bool = True,
        alpha_std: float = 0.2,
        add_legend: bool = True,
        print_summary: bool = False,
        subplot_label_ch4: str | None = None,
        subplot_label_c2h6: str | None = None,
        subplot_fontsize: int = 24,
        ch4_ylim: tuple[float, float] | None = None,
        c2h6_ylim: tuple[float, float] | None = None,
        interval: Literal["30min", "1H"] = "1H",
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4とC2H6の濃度の日内変動を描画します。

        Parameters
        ----------
            df: pd.DataFrame
                濃度データを含むDataFrameを指定します。
            col_ch4_conc: str, optional
                CH4濃度のカラム名を指定します。デフォルト値は"CH4_ultra_cal"です。
            col_c2h6_conc: str, optional
                C2H6濃度のカラム名を指定します。デフォルト値は"C2H6_ultra_cal"です。
            col_datetime: str, optional
                日時カラム名を指定します。デフォルト値は"Date"です。
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"diurnal_concentrations.png"です。
            show_std: bool, optional
                標準偏差を表示するかどうかを指定します。デフォルト値はTrueです。
            alpha_std: float, optional
                標準偏差の透明度を指定します。デフォルト値は0.2です。
            add_legend: bool, optional
                凡例を追加するかどうかを指定します。デフォルト値はTrueです。
            print_summary: bool, optional
                統計情報を表示するかどうかを指定します。デフォルト値はFalseです。
            subplot_label_ch4: str | None, optional
                CH4プロットのラベルを指定します。デフォルト値はNoneです。
            subplot_label_c2h6: str | None, optional
                C2H6プロットのラベルを指定します。デフォルト値はNoneです。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は24です。
            ch4_ylim: tuple[float, float] | None, optional
                CH4のy軸範囲を指定します。デフォルト値はNoneです。
            c2h6_ylim: tuple[float, float] | None, optional
                C2H6のy軸範囲を指定します。デフォルト値はNoneです。
            interval: Literal["30min", "1H"], optional
                時間間隔を指定します。"30min"または"1H"を指定できます。デフォルト値は"1H"です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_diurnal_concentrations(
        ...     df=monthly_data,
        ...     output_dirpath="output",
        ...     show_std=True,
        ...     interval="30min"
        ... )
        """
        # データの準備
        df_internal = df.copy()
        df_internal.index = pd.to_datetime(df_internal.index)
        if interval == "30min":
            # 30分間隔の場合、時間と30分を別々に取得
            df_internal["hour"] = pd.to_datetime(df_internal[col_datetime]).dt.hour
            df_internal["minute"] = pd.to_datetime(df_internal[col_datetime]).dt.minute
            df_internal["time_bin"] = df_internal["hour"] + df_internal["minute"].map(
                {0: 0, 30: 0.5}
            )
        else:
            # 1時間間隔の場合
            df_internal["time_bin"] = pd.to_datetime(df_internal[col_datetime]).dt.hour

        # 時間ごとの平均値と標準偏差を計算
        hourly_stats = df_internal.groupby("time_bin")[
            [col_ch4_conc, col_c2h6_conc]
        ].agg(["mean", "std"])

        # 最後のデータポイントを追加(最初のデータを使用)
        last_point = hourly_stats.iloc[0:1].copy()
        last_point.index = pd.Index(
            [hourly_stats.index[-1] + (0.5 if interval == "30min" else 1)]
        )
        hourly_stats = pd.concat([hourly_stats, last_point])

        # 時間軸の作成
        if interval == "30min":
            time_points = pd.date_range("2024-01-01", periods=49, freq="30min")
            x_ticks = [0, 6, 12, 18, 24]  # 主要な時間のティック
        else:
            time_points = pd.date_range("2024-01-01", periods=25, freq="1H")
            x_ticks = [0, 6, 12, 18, 24]

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CH4濃度プロット
        mean_ch4 = hourly_stats[col_ch4_conc]["mean"]
        if show_std:
            std_ch4 = hourly_stats[col_ch4_conc]["std"]
            ax1.fill_between(
                time_points,
                mean_ch4 - std_ch4,
                mean_ch4 + std_ch4,
                color="red",
                alpha=alpha_std,
            )
        ch4_line = ax1.plot(time_points, mean_ch4, "red", label="CH$_4$")[0]

        ax1.set_ylabel("CH$_4$ (ppm)")
        if ch4_ylim is not None:
            ax1.set_ylim(ch4_ylim)
        if subplot_label_ch4:
            ax1.text(
                0.02,
                0.98,
                subplot_label_ch4,
                transform=ax1.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # C2H6濃度プロット
        mean_c2h6 = hourly_stats[col_c2h6_conc]["mean"]
        if show_std:
            std_c2h6 = hourly_stats[col_c2h6_conc]["std"]
            ax2.fill_between(
                time_points,
                mean_c2h6 - std_c2h6,
                mean_c2h6 + std_c2h6,
                color="orange",
                alpha=alpha_std,
            )
        c2h6_line = ax2.plot(time_points, mean_c2h6, "orange", label="C$_2$H$_6$")[0]

        ax2.set_ylabel("C$_2$H$_6$ (ppb)")
        if c2h6_ylim is not None:
            ax2.set_ylim(c2h6_ylim)
        if subplot_label_c2h6:
            ax2.text(
                0.02,
                0.98,
                subplot_label_c2h6,
                transform=ax2.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # 両プロットの共通設定
        for ax in [ax1, ax2]:
            ax.set_xlabel("Time (hour)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=x_ticks))
            ax.set_xlim(time_points[0], time_points[-1])
            # 1時間ごとの縦線を表示
            ax.grid(True, which="major", alpha=0.3)

        # 共通の凡例を図の下部に配置
        if add_legend:
            fig.legend(
                [ch4_line, c2h6_line],
                ["CH$_4$", "C$_2$H$_6$"],
                loc="center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=2,
            )
        plt.subplots_adjust(bottom=0.2)

        plt.tight_layout()
        if save_fig:
            if output_dirpath is None:
                raise ValueError()
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        if print_summary:
            # 統計情報の表示
            for name, col in [("CH4", col_ch4_conc), ("C2H6", col_c2h6_conc)]:
                stats = hourly_stats[col]
                mean_vals = stats["mean"]

                print(f"\n{name}濃度の日内変動統計:")
                print(f"最小値: {mean_vals.min():.3f} (Hour: {mean_vals.idxmin()})")
                print(f"最大値: {mean_vals.max():.3f} (Hour: {mean_vals.idxmax()})")
                print(f"平均値: {mean_vals.mean():.3f}")
                print(f"日内変動幅: {mean_vals.max() - mean_vals.min():.3f}")
                print(f"最大/最小比: {mean_vals.max() / mean_vals.min():.3f}")

    def plot_flux_diurnal_patterns_with_std(
        self,
        df: pd.DataFrame,
        col_ch4_flux: str = "Fch4",
        col_c2h6_flux: str = "Fc2h6",
        ch4_label: str = r"$\mathregular{CH_{4}}$フラックス",
        c2h6_label: str = r"$\mathregular{C_{2}H_{6}}$フラックス",
        col_datetime: str = "Date",
        output_dirpath: str | Path | None = None,
        output_filename: str = "diurnal_patterns.png",
        window_size: int = 6,
        show_std: bool = True,
        alpha_std: float = 0.1,
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
        print_summary: bool = False,
    ) -> None:
        """CH4とC2H6フラックスの日変化パターンをプロットします。

        Parameters
        ----------
            df: pd.DataFrame
                プロットするデータを含むDataFrameを指定します。
            col_ch4_flux: str, optional
                CH4フラックスのカラム名を指定します。デフォルト値は"Fch4"です。
            col_c2h6_flux: str, optional
                C2H6フラックスのカラム名を指定します。デフォルト値は"Fc2h6"です。
            ch4_label: str, optional
                CH4フラックスの凡例ラベルを指定します。デフォルト値は"CH4フラックス"です。
            c2h6_label: str, optional
                C2H6フラックスの凡例ラベルを指定します。デフォルト値は"C2H6フラックス"です。
            col_datetime: str, optional
                日時カラムの名前を指定します。デフォルト値は"Date"です。
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"diurnal_patterns.png"です。
            window_size: int, optional
                移動平均の窓サイズを指定します。デフォルト値は6です。
            show_std: bool, optional
                標準偏差を表示するかどうかを指定します。デフォルト値はTrueです。
            alpha_std: float, optional
                標準偏差の透明度を0-1の範囲で指定します。デフォルト値は0.1です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。
            print_summary: bool, optional
                統計情報を表示するかどうかを指定します。デフォルト値はFalseです。

        Examples
        --------
        >>> generator = MonthlyFiguresGenerator()
        >>> df = pd.read_csv("flux_data.csv")
        >>> generator.plot_flux_diurnal_patterns_with_std(
        ...     df,
        ...     col_ch4_flux="CH4_flux",
        ...     col_c2h6_flux="C2H6_flux",
        ...     output_dirpath="output",
        ...     show_std=True
        ... )
        """
        # 日時インデックスの処理
        df_internal = df.copy()
        if not isinstance(df_internal.index, pd.DatetimeIndex):
            df_internal[col_datetime] = pd.to_datetime(df_internal[col_datetime])
            df_internal.set_index(col_datetime, inplace=True)
        df_internal.index = pd.to_datetime(df_internal.index)
        # 時刻データの抽出とグループ化
        df_internal["hour"] = df_internal.index.hour
        hourly_means = df_internal.groupby("hour")[[col_ch4_flux, col_c2h6_flux]].agg(
            ["mean", "std"]
        )

        # 24時間目のデータ点を追加(0時のデータを使用)
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = pd.Index([24])
        hourly_means = pd.concat([hourly_means, last_hour])

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 移動平均の計算と描画
        ch4_mean = (
            hourly_means[(col_ch4_flux, "mean")]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        c2h6_mean = (
            hourly_means[(col_c2h6_flux, "mean")]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        if show_std:
            ch4_std = (
                hourly_means[(col_ch4_flux, "std")]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            c2h6_std = (
                hourly_means[(col_c2h6_flux, "std")]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

            ax1.fill_between(
                time_points,
                ch4_mean - ch4_std,
                ch4_mean + ch4_std,
                color="blue",
                alpha=alpha_std,
            )
            ax2.fill_between(
                time_points,
                c2h6_mean - c2h6_std,
                c2h6_mean + c2h6_std,
                color="red",
                alpha=alpha_std,
            )

        # メインのラインプロット
        ax1.plot(time_points, ch4_mean, "blue", label=ch4_label)
        ax2.plot(time_points, c2h6_mean, "red", label=c2h6_label)

        # 軸の設定
        for ax, ylabel in [
            (ax1, r"CH$_4$ (nmol m$^{-2}$ s$^{-1}$)"),
            (ax2, r"C$_2$H$_6$ (nmol m$^{-2}$ s$^{-1}$)"),
        ]:
            ax.set_xlabel("Time")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
            ax.set_xlim(time_points[0], time_points[-1])
            ax.grid(True, alpha=0.3)
            ax.legend()

        # グラフの保存
        plt.tight_layout()

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=350, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        # 統計情報の表示(オプション)
        if print_summary:
            for col, name in [(col_ch4_flux, "CH4"), (col_c2h6_flux, "C2H6")]:
                mean_val = hourly_means[(col, "mean")].mean()
                min_val = hourly_means[(col, "mean")].min()
                max_val = hourly_means[(col, "mean")].max()
                min_time = hourly_means[(col, "mean")].idxmin()
                max_time = hourly_means[(col, "mean")].idxmax()

                self.logger.info(f"{name} Statistics:")
                self.logger.info(f"Mean: {mean_val:.2f}")
                self.logger.info(f"Min: {min_val:.2f} (Hour: {min_time})")
                self.logger.info(f"Max: {max_val:.2f} (Hour: {max_time})")
                self.logger.info(f"Max/Min ratio: {max_val / min_val:.2f}\n")

    def plot_gas_ratio_diurnal(
        self,
        df: pd.DataFrame,
        col_ratio_1: str,
        col_ratio_2: str,
        label_1: str,
        label_2: str,
        color_1: str,
        color_2: str,
        output_dirpath: str | Path | None = None,
        output_filename: str = "gas_ratio_diurnal.png",
        add_xlabel: bool = True,
        add_ylabel: bool = True,
        add_legend: bool = True,
        xlabel: str = "Hour",
        ylabel: str = "都市ガスが占める排出比率 (%)",
        subplot_fontsize: int = 20,
        subplot_label: str | None = None,
        y_max: float | None = 100,
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = False,
    ) -> None:
        """2つの比率の日変化を比較するプロットを作成します。

        Parameters
        ----------
            df: pd.DataFrame
                プロットするデータを含むDataFrameを指定します。
            col_ratio_1: str
                1つ目の比率データを含むカラム名を指定します。
            col_ratio_2: str
                2つ目の比率データを含むカラム名を指定します。
            label_1: str
                1つ目の比率データの凡例ラベルを指定します。
            label_2: str
                2つ目の比率データの凡例ラベルを指定します。
            color_1: str
                1つ目の比率データのプロット色を指定します。
            color_2: str
                2つ目の比率データのプロット色を指定します。
            output_dirpath: str | Path | None, optional
                出力先ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"gas_ratio_diurnal.png"です。
            add_xlabel: bool, optional
                x軸ラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_ylabel: bool, optional
                y軸ラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_legend: bool, optional
                凡例を表示するかどうかを指定します。デフォルト値はTrueです。
            xlabel: str, optional
                x軸のラベルを指定します。デフォルト値は"Hour"です。
            ylabel: str, optional
                y軸のラベルを指定します。デフォルト値は"都市ガスが占める排出比率 (%)"です。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            subplot_label: str | None, optional
                サブプロットのラベルを指定します。デフォルト値はNoneです。
            y_max: float | None, optional
                y軸の最大値を指定します。デフォルト値は100です。
            figsize: tuple[float, float], optional
                図のサイズを指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                図の解像度を指定します。デフォルト値は350です。
            save_fig: bool, optional
                図を保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                図を表示するかどうかを指定します。デフォルト値はFalseです。

        Examples
        -------
        >>> df = pd.DataFrame({
        ...     'ratio1': [80, 85, 90],
        ...     'ratio2': [70, 75, 80]
        ... })
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_gas_ratio_diurnal(
        ...     df=df,
        ...     col_ratio_1='ratio1',
        ...     col_ratio_2='ratio2',
        ...     label_1='比率1',
        ...     label_2='比率2',
        ...     color_1='blue',
        ...     color_2='red',
        ...     output_dirpath='output'
        ... )
        """
        df_internal: pd.DataFrame = df.copy()
        df_internal.index = pd.to_datetime(df_internal.index)

        # 時刻でグループ化して平均を計算
        hourly_means = df_internal.groupby(df_internal.index.hour)[
            [col_ratio_1, col_ratio_2]
        ].mean()
        hourly_stds = df_internal.groupby(df_internal.index.hour)[
            [col_ratio_1, col_ratio_2]
        ].std()

        # 24時間目のデータ点を追加(0時のデータを使用)
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = pd.Index([24])
        hourly_means = pd.concat([hourly_means, last_hour])

        last_hour_std = hourly_stds.iloc[0:1].copy()
        last_hour_std.index = pd.Index([24])
        hourly_stds = pd.concat([hourly_stds, last_hour_std])

        # 24時間分の時刻を生成
        time_points: pd.DatetimeIndex = pd.date_range(
            "2024-01-01", periods=25, freq="h"
        )

        # プロットの作成
        fig, ax = plt.subplots(figsize=figsize)

        # 1つ目の比率
        ax.plot(
            time_points,  # [:-1]を削除
            hourly_means[col_ratio_1],
            color=color_1,
            label=label_1,
            alpha=0.7,
        )
        ax.fill_between(
            time_points,  # [:-1]を削除
            hourly_means[col_ratio_1] - hourly_stds[col_ratio_1],
            hourly_means[col_ratio_1] + hourly_stds[col_ratio_1],
            color=color_1,
            alpha=0.2,
        )

        # 2つ目の比率
        ax.plot(
            time_points,  # [:-1]を削除
            hourly_means[col_ratio_2],
            color=color_2,
            label=label_2,
            alpha=0.7,
        )
        ax.fill_between(
            time_points,  # [:-1]を削除
            hourly_means[col_ratio_2] - hourly_stds[col_ratio_2],
            hourly_means[col_ratio_2] + hourly_stds[col_ratio_2],
            color=color_2,
            alpha=0.2,
        )

        # 軸の設定
        if add_xlabel:
            ax.set_xlabel(xlabel)
        if add_ylabel:
            ax.set_ylabel(ylabel)

        # y軸の範囲設定
        if y_max is not None:
            ax.set_ylim(0, y_max)

        # グリッド線の追加
        ax.grid(True, alpha=0.3)
        ax.grid(True, which="minor", alpha=0.1)

        # x軸の設定
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(
            float(mdates.date2num(time_points[0])),
            float(mdates.date2num(time_points[-1])),
        )
        ax.set_xticks(time_points[::6])
        ax.set_xticklabels(["0", "6", "12", "18", "24"])

        # サブプロットラベルの追加
        if subplot_label:
            ax.text(
                0.02,
                0.98,
                subplot_label,
                transform=ax.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # 凡例の追加
        if add_legend:
            # 凡例を図の下部中央に配置
            ax.legend(
                loc="center",
                bbox_to_anchor=(0.5, -0.25),  # 図の下部に配置
                ncol=2,  # 2列で表示
                frameon=False,  # 枠を非表示
            )
            # 凡例のために下部のマージンを調整
            plt.subplots_adjust(bottom=0.2)

        # プロットの保存と表示
        plt.tight_layout()
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_scatter(
        self,
        df: pd.DataFrame,
        col_x: str,
        col_y: str,
        output_dirpath: str | Path | None = None,
        output_filename: str = "scatter.png",
        add_label: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        x_axis_range: tuple | None = None,
        y_axis_range: tuple | None = None,
        x_scientific: bool = False,
        y_scientific: bool = False,
        fixed_slope: float = 0.076,
        show_fixed_slope: bool = False,
        figsize: tuple[float, float] = (6, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """散布図を作成し、TLS回帰直線を描画します。

        Parameters
        ----------
            df: pd.DataFrame
                プロットに使用するデータフレームを指定します。
            col_x: str
                x軸に使用する列名を指定します。
            col_y: str
                y軸に使用する列名を指定します。
            output_dirpath: str | Path | None, optional
                出力先ディレクトリを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"scatter.png"です。
            add_label: bool, optional
                軸ラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            xlabel: str | None, optional
                x軸のラベルを指定します。デフォルト値はNoneです。
            ylabel: str | None, optional
                y軸のラベルを指定します。デフォルト値はNoneです。
            x_axis_range: tuple | None, optional
                x軸の表示範囲を(最小値, 最大値)で指定します。デフォルト値はNoneです。
            y_axis_range: tuple | None, optional
                y軸の表示範囲を(最小値, 最大値)で指定します。デフォルト値はNoneです。
            x_scientific: bool, optional
                x軸を科学的記法で表示するかどうかを指定します。デフォルト値はFalseです。
            y_scientific: bool, optional
                y軸を科学的記法で表示するかどうかを指定します。デフォルト値はFalseです。
            fixed_slope: float, optional
                固定傾きの値を指定します。デフォルト値は0.076です。
            show_fixed_slope: bool, optional
                固定傾きの線を表示するかどうかを指定します。デフォルト値はFalseです。
            figsize: tuple[float, float], optional
                プロットのサイズを(幅, 高さ)で指定します。デフォルト値は(6, 6)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'x': [1, 2, 3, 4, 5],
        ...     'y': [2, 4, 6, 8, 10]
        ... })
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_scatter(
        ...     df=df,
        ...     col_x='x',
        ...     col_y='y',
        ...     xlabel='X軸',
        ...     ylabel='Y軸',
        ...     output_dirpath='output'
        ... )
        """
        # 有効なデータの抽出
        df_internal = MonthlyFiguresGenerator.get_valid_data(
            df=df, col_x=col_x, col_y=col_y
        )

        # データの準備
        x = df_internal[col_x].values
        y = df_internal[col_y].values

        # データの中心化
        x_array = np.array(x)
        y_array = np.array(y)
        x_mean = np.mean(x_array, axis=0)
        y_mean = np.mean(y_array, axis=0)
        x_c = x - x_mean
        y_c = y - y_mean

        # TLS回帰の計算
        data_matrix = np.vstack((x_c, y_c))
        cov_matrix = np.cov(data_matrix)
        _, eigenvecs = linalg.eigh(cov_matrix)
        largest_eigenvec = eigenvecs[:, -1]

        slope = largest_eigenvec[1] / largest_eigenvec[0]
        intercept = y_mean - slope * x_mean

        # R²とRMSEの計算
        y_pred = slope * x + intercept
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # プロットの作成
        fig, ax = plt.subplots(figsize=figsize)

        # データ点のプロット
        ax.scatter(x_array, y_array, color="black")

        # データの範囲を取得
        if x_axis_range is None:
            x_axis_range = (df_internal[col_x].min(), df_internal[col_x].max())
        if y_axis_range is None:
            y_axis_range = (df_internal[col_y].min(), df_internal[col_y].max())

        # 回帰直線のプロット
        x_range = np.linspace(x_axis_range[0], x_axis_range[1], 150)
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r", label="TLS regression")

        # 傾き固定の線を追加(フラグがTrueの場合)
        if show_fixed_slope:
            fixed_intercept = (
                y_mean - fixed_slope * x_mean
            )  # 中心点を通るように切片を計算
            y_fixed = fixed_slope * x_range + fixed_intercept
            ax.plot(x_range, y_fixed, "b--", label=f"Slope = {fixed_slope}", alpha=0.7)

        # 軸の設定
        ax.set_xlim(x_axis_range)
        ax.set_ylim(y_axis_range)

        # 指数表記の設定
        if x_scientific:
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_position((1.1, 0))  # 指数の位置調整
        if y_scientific:
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_position((0, 1.1))  # 指数の位置調整

        if add_label:
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)

        # 1:1の関係を示す点線(軸の範囲が同じ場合のみ表示)
        if (
            x_axis_range is not None
            and y_axis_range is not None
            and x_axis_range == y_axis_range
        ):
            ax.plot(
                [x_axis_range[0], x_axis_range[1]],
                [x_axis_range[0], x_axis_range[1]],
                "k--",
                alpha=0.5,
            )

        # 回帰情報の表示
        equation = (
            f"y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
        )
        position_x = 0.05
        fig_ha: str = "left"
        ax.text(
            position_x,
            0.95,
            equation,
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        ax.text(
            position_x,
            0.88,
            f"R² = {r_squared:.2f}",
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        ax.text(
            position_x,
            0.81,  # RMSEのための新しい位置
            f"RMSE = {rmse:.2f}",
            transform=ax.transAxes,
            va="top",
            ha=fig_ha,
            color="red",
        )
        # 目盛り線の設定
        ax.grid(True, alpha=0.3)

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            fig.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_source_contributions_diurnal(
        self,
        df: pd.DataFrame,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        col_datetime: str = "Date",
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "bio",
        label_gas: str = "gas",
        flux_alpha: float = 0.6,
        output_dirpath: str | Path | None = None,
        output_filename: str = "source_contributions.png",
        window_size: int = 6,
        print_summary: bool = False,
        add_xlabel: bool = True,
        add_ylabel: bool = True,
        add_legend: bool = True,
        smooth: bool = False,
        y_max: float = 100,
        subplot_label: str | None = None,
        subplot_fontsize: int = 20,
        figsize: tuple[float, float] = (10, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の日変化を積み上げグラフとして表示します。

        Parameters
        ----------
            df: pd.DataFrame
                CH4フラックスデータを含むデータフレームを指定します。
            col_ch4_flux: str
                CH4フラックスの列名を指定します。
            col_c2h6_flux: str
                C2H6フラックスの列名を指定します。
            col_datetime: str, optional
                日時の列名を指定します。デフォルト値は"Date"です。
            color_bio: str, optional
                生物起源の色を指定します。デフォルト値は"blue"です。
            color_gas: str, optional
                都市ガス起源の色を指定します。デフォルト値は"red"です。
            label_bio: str, optional
                生物起源のラベルを指定します。デフォルト値は"bio"です。
            label_gas: str, optional
                都市ガスのラベルを指定します。デフォルト値は"gas"です。
            flux_alpha: float, optional
                フラックスの透明度を指定します。デフォルト値は0.6です。
            output_dirpath: str | Path | None, optional
                出力先のディレクトリを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"source_contributions.png"です。
            window_size: int, optional
                移動平均の窓サイズを指定します。デフォルト値は6です。
            print_summary: bool, optional
                統計情報を表示するかどうかを指定します。デフォルト値はFalseです。
            add_xlabel: bool, optional
                x軸のラベルを追加するかどうかを指定します。デフォルト値はTrueです。
            add_ylabel: bool, optional
                y軸のラベルを追加するかどうかを指定します。デフォルト値はTrueです。
            add_legend: bool, optional
                凡例を追加するかどうかを指定します。デフォルト値はTrueです。
            smooth: bool, optional
                移動平均を適用するかどうかを指定します。デフォルト値はFalseです。
            y_max: float, optional
                y軸の上限値を指定します。デフォルト値は100です。
            subplot_label: str | None, optional
                サブプロットのラベルを指定します。デフォルト値はNoneです。
            subplot_fontsize: int, optional
                サブプロットラベルのフォントサイズを指定します。デフォルト値は20です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(10, 6)です。
            dpi: float | None, optional
                プロットのdpiを指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> df = pd.read_csv("flux_data.csv")
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_source_contributions_diurnal(
        ...     df=df,
        ...     col_ch4_flux="Fch4",
        ...     col_c2h6_flux="Fc2h6",
        ...     output_dirpath="output",
        ...     output_filename="diurnal_sources.png"
        ... )
        """
        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            col_ch4_flux=col_ch4_flux,
            col_c2h6_flux=col_c2h6_flux,
            col_datetime=col_datetime,
        )
        df_with_sources.index = pd.to_datetime(df_with_sources.index)

        # 時刻データの抽出とグループ化
        df_with_sources["hour"] = df_with_sources.index.hour
        hourly_means = df_with_sources.groupby("hour")[["ch4_gas", "ch4_bio"]].mean()

        # 24時間目のデータ点を追加(0時のデータを使用)
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = pd.Index([24])
        hourly_means = pd.concat([hourly_means, last_hour])

        # 移動平均の適用
        hourly_means_smoothed = hourly_means
        if smooth:
            hourly_means_smoothed = hourly_means.rolling(
                window=window_size, center=True, min_periods=1
            ).mean()

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        # プロットの作成
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        # サブプロットラベルの追加(subplot_labelが指定されている場合)
        if subplot_label:
            ax.text(
                0.02,  # x位置
                0.98,  # y位置
                subplot_label,
                transform=ax.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # 積み上げプロット
        ax.fill_between(
            time_points,
            0,
            hourly_means_smoothed["ch4_bio"],
            color=color_bio,
            alpha=flux_alpha,
            label=label_bio,
        )
        ax.fill_between(
            time_points,
            hourly_means_smoothed["ch4_bio"],
            hourly_means_smoothed["ch4_bio"] + hourly_means_smoothed["ch4_gas"],
            color=color_gas,
            alpha=flux_alpha,
            label=label_gas,
        )

        # 合計値のライン
        total_flux = hourly_means_smoothed["ch4_bio"] + hourly_means_smoothed["ch4_gas"]
        ax.plot(time_points, total_flux, "-", color="black", alpha=0.5)

        # 軸の設定
        if add_xlabel:
            ax.set_xlabel("Time (hour)")
        if add_ylabel:
            ax.set_ylabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(
            float(mdates.date2num(time_points[0])),
            float(mdates.date2num(time_points[-1])),
        )
        ax.set_ylim(0, y_max)  # y軸の範囲を設定
        ax.grid(True, alpha=0.3)

        # 凡例を図の下部に配置
        if add_legend:
            handles, labels = ax.get_legend_handles_labels()
            fig = plt.gcf()  # 現在の図を取得
            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(handles),
            )
            plt.subplots_adjust(bottom=0.2)  # 下部に凡例用のスペースを確保
        plt.tight_layout()

        # グラフの保存、表示
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        # 統計情報の表示
        if print_summary:
            # 昼夜の時間帯を定義
            daytime_range: list[int] = [6, 19]  # 6~18時
            daytime_hours = range(daytime_range[0], daytime_range[1])
            nighttime_hours = list(range(0, daytime_range[0])) + list(
                range(daytime_range[1], 24)
            )

            # 都市ガスと生物起源のデータを取得
            gas_flux = hourly_means["ch4_gas"]
            bio_flux = hourly_means["ch4_bio"]
            total_flux = gas_flux + bio_flux

            # 都市ガス比率を計算
            gas_ratio = (gas_flux / total_flux) * 100
            daytime_gas_ratio = (
                pd.Series(gas_flux).iloc[np.array(list(daytime_hours))].sum()
                / pd.Series(total_flux).iloc[np.array(list(daytime_hours))].sum()
            ) * 100
            nighttime_gas_ratio = (
                pd.Series(gas_flux).iloc[np.array(list(nighttime_hours))].sum()
                / pd.Series(total_flux).iloc[np.array(list(nighttime_hours))].sum()
            ) * 100

            stats = {
                "都市ガス起源": gas_flux,
                "生物起源": bio_flux,
                "合計": total_flux,
            }

            # 都市ガス比率の統計を出力
            self.logger.info("\n都市ガス比率の統計:")
            print(f"  全体の都市ガス比率: {gas_ratio.mean():.1f}%")
            print(
                f"  昼間({daytime_range[0]}~{daytime_range[1] - 1}時)の都市ガス比率: {daytime_gas_ratio:.1f}%"
            )
            print(
                f"  夜間({daytime_range[1] - 1}~{daytime_range[0]}時)の都市ガス比率: {nighttime_gas_ratio:.1f}%"
            )
            print(f"  最小比率: {gas_ratio.min():.1f}% (Hour: {gas_ratio.idxmin()})")
            print(f"  最大比率: {gas_ratio.max():.1f}% (Hour: {gas_ratio.idxmax()})")

            # 各フラックスの統計を出力
            for source, data in stats.items():
                mean_val = data.mean()
                min_val = data.min()
                max_val = data.max()
                min_time = data.idxmin()
                max_time = data.idxmax()

                # 昼間と夜間のデータを抽出
                daytime_data = pd.Series(data).iloc[np.array(list(daytime_hours))]
                nighttime_data = pd.Series(data).iloc[np.array(list(nighttime_hours))]

                daytime_mean = daytime_data.mean()
                nighttime_mean = nighttime_data.mean()

                self.logger.info(f"\n{source}の統計:")
                print(f"  平均値: {mean_val:.2f}")
                print(f"  最小値: {min_val:.2f} (Hour: {min_time})")
                print(f"  最大値: {max_val:.2f} (Hour: {max_time})")
                if min_val != 0:
                    print(f"  最大/最小比: {max_val / min_val:.2f}")
                print(
                    f"  昼間({daytime_range[0]}~{daytime_range[1] - 1}時)の平均: {daytime_mean:.2f}"
                )
                print(
                    f"  夜間({daytime_range[1] - 1}~{daytime_range[0]}時)の平均: {nighttime_mean:.2f}"
                )
                if nighttime_mean != 0:
                    print(f"  昼/夜比: {daytime_mean / nighttime_mean:.2f}")

    def plot_source_contributions_diurnal_by_date(
        self,
        df: pd.DataFrame,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        col_datetime: str = "Date",
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "bio",
        label_gas: str = "gas",
        flux_alpha: float = 0.6,
        output_dirpath: str | Path | None = None,
        output_filename: str = "source_contributions_by_date.png",
        add_xlabel: bool = True,
        add_ylabel: bool = True,
        add_legend: bool = True,
        print_summary: bool = False,
        subplot_fontsize: int = 20,
        subplot_label_weekday: str | None = None,
        subplot_label_weekend: str | None = None,
        y_max: float | None = None,
        figsize: tuple[float, float] = (12, 5),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の日変化を平日・休日別に表示します。

        Parameters
        ----------
            df: pd.DataFrame
                CH4フラックスとC2H6フラックスのデータを含むデータフレームを指定します。
            col_ch4_flux: str
                CH4フラックスのカラム名を指定します。
            col_c2h6_flux: str
                C2H6フラックスのカラム名を指定します。
            col_datetime: str, optional
                日時カラムの名前を指定します。デフォルト値は"Date"です。
            color_bio: str, optional
                生物起源のプロット色を指定します。デフォルト値は"blue"です。
            color_gas: str, optional
                都市ガス起源のプロット色を指定します。デフォルト値は"red"です。
            label_bio: str, optional
                生物起源の凡例ラベルを指定します。デフォルト値は"bio"です。
            label_gas: str, optional
                都市ガスの凡例ラベルを指定します。デフォルト値は"gas"です。
            flux_alpha: float, optional
                フラックスプロットの透明度を指定します。デフォルト値は0.6です。
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパスを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"source_contributions_by_date.png"です。
            add_xlabel: bool, optional
                x軸のラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_ylabel: bool, optional
                y軸のラベルを表示するかどうかを指定します。デフォルト値はTrueです。
            add_legend: bool, optional
                凡例を表示するかどうかを指定します。デフォルト値はTrueです。
            print_summary: bool, optional
                統計情報を表示するかどうかを指定します。デフォルト値はFalseです。
            subplot_fontsize: int, optional
                サブプロットのフォントサイズを指定します。デフォルト値は20です。
            subplot_label_weekday: str | None, optional
                平日グラフのラベルを指定します。デフォルト値はNoneです。
            subplot_label_weekend: str | None, optional
                休日グラフのラベルを指定します。デフォルト値はNoneです。
            y_max: float | None, optional
                y軸の上限値を指定します。デフォルト値はNoneです。
            figsize: tuple[float, float], optional
                プロットのサイズを(幅, 高さ)で指定します。デフォルト値は(12, 5)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルト値は350です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'Date': pd.date_range('2024-01-01', periods=48, freq='H'),
        ...     'Fch4': [1.2] * 48,
        ...     'Fc2h6': [0.1] * 48
        ... })
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_source_contributions_diurnal_by_date(
        ...     df=df,
        ...     col_ch4_flux='Fch4',
        ...     col_c2h6_flux='Fc2h6',
        ...     output_dirpath='output'
        ... )
        """
        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            col_ch4_flux=col_ch4_flux,
            col_c2h6_flux=col_c2h6_flux,
            col_datetime=col_datetime,
        )
        df_with_sources.index = pd.to_datetime(df_with_sources.index)

        # 日付タイプの分類
        dates = pd.to_datetime(df_with_sources.index)
        is_weekend = dates.dayofweek.isin([5, 6])
        is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
        is_weekday = ~(is_weekend | is_holiday)

        # データの分類
        data_weekday = df_with_sources[is_weekday]
        data_holiday = df_with_sources[is_weekend | is_holiday]

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 平日と休日それぞれのプロット
        for ax, data, _ in [
            (ax1, data_weekday, "Weekdays"),
            (ax2, data_holiday, "Weekends & Holidays"),
        ]:
            # 時間ごとの平均値を計算
            hourly_means = data.groupby(pd.to_datetime(data.index).hour)[
                ["ch4_gas", "ch4_bio"]
            ].mean()

            # 24時間目のデータ点を追加
            last_hour = hourly_means.iloc[0:1].copy()
            last_hour.index = pd.Index([24])
            hourly_means = pd.concat([hourly_means, last_hour])

            # 24時間分のデータポイントを作成
            time_points = pd.date_range("2024-01-01", periods=25, freq="h")

            # 積み上げプロット
            ax.fill_between(
                time_points,
                0,
                hourly_means["ch4_bio"],
                color=color_bio,
                alpha=flux_alpha,
                label=label_bio,
            )
            ax.fill_between(
                time_points,
                hourly_means["ch4_bio"],
                hourly_means["ch4_bio"] + hourly_means["ch4_gas"],
                color=color_gas,
                alpha=flux_alpha,
                label=label_gas,
            )

            # 合計値のライン
            total_flux = hourly_means["ch4_bio"] + hourly_means["ch4_gas"]
            ax.plot(time_points, total_flux, "-", color="black", alpha=0.5)

            # 軸の設定
            if add_xlabel:
                ax.set_xlabel("Time (hour)")
            if add_ylabel:
                if ax == ax1:  # 左側のプロットのラベル
                    ax.set_ylabel("CH$_4$ flux\n" r"(nmol m$^{-2}$ s$^{-1}$)")
                else:  # 右側のプロットのラベル
                    ax.set_ylabel("CH$_4$ flux\n" r"(nmol m$^{-2}$ s$^{-1}$)")

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
            ax.set_xlim(
                float(mdates.date2num(time_points[0])),
                float(mdates.date2num(time_points[-1])),
            )
            if y_max is not None:
                ax.set_ylim(0, y_max)
            ax.grid(True, alpha=0.3)

        # サブプロットラベルの追加
        if subplot_label_weekday:
            ax1.text(
                0.02,
                0.98,
                subplot_label_weekday,
                transform=ax1.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )
        if subplot_label_weekend:
            ax2.text(
                0.02,
                0.98,
                subplot_label_weekend,
                transform=ax2.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        # 凡例を図の下部に配置
        if add_legend:
            # 最初のプロットから凡例のハンドルとラベルを取得
            handles, labels = ax1.get_legend_handles_labels()
            # 図の下部に凡例を配置
            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.01),  # x=0.5で中央、y=0.01で下部に配置
                ncol=len(handles),  # ハンドルの数だけ列を作成(一行に表示)
            )
            # 凡例用のスペースを確保
            plt.subplots_adjust(bottom=0.2)  # 下部に30%のスペースを確保

        plt.tight_layout()
        # グラフの保存または表示
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        # 統計情報の表示
        if print_summary:
            for data, label in [
                (data_weekday, "Weekdays"),
                (data_holiday, "Weekends & Holidays"),
            ]:
                hourly_means = data.groupby(pd.to_datetime(data.index).hour)[
                    ["ch4_gas", "ch4_bio"]
                ].mean()

                print(f"\n{label}の統計:")

                # 都市ガス起源の統計
                gas_flux = hourly_means["ch4_gas"]
                bio_flux = hourly_means["ch4_bio"]

                # 昼夜の時間帯を定義
                daytime_range: list[int] = [6, 19]  # m~n時の場合、[m ,(n+1)]と定義
                daytime_hours = range(daytime_range[0], daytime_range[1])
                nighttime_hours = list(range(0, daytime_range[0])) + list(
                    range(daytime_range[1], 24)
                )

                # 昼間の統計
                daytime_gas = pd.Series(gas_flux).iloc[np.array(list(daytime_hours))]
                daytime_bio = pd.Series(bio_flux).iloc[np.array(list(daytime_hours))]
                daytime_total = daytime_gas + daytime_bio
                daytime_total = daytime_gas + daytime_bio
                daytime_ratio = (daytime_gas.sum() / daytime_total.sum()) * 100

                # 夜間の統計
                nighttime_gas = pd.Series(gas_flux).iloc[
                    np.array(list(nighttime_hours))
                ]
                nighttime_bio = pd.Series(bio_flux).iloc[
                    np.array(list(nighttime_hours))
                ]
                nighttime_total = nighttime_gas + nighttime_bio
                nighttime_ratio = (nighttime_gas.sum() / nighttime_total.sum()) * 100

                print("\n都市ガス起源:")
                print(f"  平均値: {gas_flux.mean():.2f}")
                print(f"  最小値: {gas_flux.min():.2f} (Hour: {gas_flux.idxmin()})")
                print(f"  最大値: {gas_flux.max():.2f} (Hour: {gas_flux.idxmax()})")
                if gas_flux.min() != 0:
                    print(f"  最大/最小比: {gas_flux.max() / gas_flux.min():.2f}")
                print(
                    f"  全体に占める割合: {(gas_flux.sum() / (gas_flux.sum() + hourly_means['ch4_bio'].sum()) * 100):.1f}%"
                )
                print(
                    f"  昼間({daytime_range[0]}~{daytime_range[1] - 1}時)の割合: {daytime_ratio:.1f}%"
                )
                print(
                    f"  夜間({daytime_range[1] - 1}~{daytime_range[0]}時)の割合: {nighttime_ratio:.1f}%"
                )

                # 生物起源の統計
                bio_flux = hourly_means["ch4_bio"]
                print("\n生物起源:")
                print(f"  平均値: {bio_flux.mean():.2f}")
                print(f"  最小値: {bio_flux.min():.2f} (Hour: {bio_flux.idxmin()})")
                print(f"  最大値: {bio_flux.max():.2f} (Hour: {bio_flux.idxmax()})")
                if bio_flux.min() != 0:
                    print(f"  最大/最小比: {bio_flux.max() / bio_flux.min():.2f}")
                print(
                    f"  全体に占める割合: {(bio_flux.sum() / (gas_flux.sum() + bio_flux.sum()) * 100):.1f}%"
                )

                # 合計フラックスの統計
                total_flux = gas_flux + bio_flux
                print("\n合計:")
                print(f"  平均値: {total_flux.mean():.2f}")
                print(f"  最小値: {total_flux.min():.2f} (Hour: {total_flux.idxmin()})")
                print(f"  最大値: {total_flux.max():.2f} (Hour: {total_flux.idxmax()})")
                if total_flux.min() != 0:
                    print(f"  最大/最小比: {total_flux.max() / total_flux.min():.2f}")

    def plot_wind_rose_sources(
        self,
        df: pd.DataFrame,
        output_dirpath: str | Path | None = None,
        output_filename: str = "edp_wind_rose.png",
        col_datetime: str = "Date",
        col_ch4_flux: str = "Fch4",
        col_c2h6_flux: str = "Fc2h6",
        col_wind_dir: str = "Wind direction",
        flux_unit: str = r"(nmol m$^{-2}$ s$^{-1}$)",
        ymax: float | None = None,
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "生物起源",
        label_gas: str = "都市ガス起源",
        flux_alpha: float = 0.4,
        num_directions: int = 8,
        gap_degrees: float = 0.0,
        center_on_angles: bool = True,
        subplot_label: str | None = None,
        add_legend: bool = True,
        stack_bars: bool = True,
        print_summary: bool = False,
        figsize: tuple[float, float] = (8, 8),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の風配図を作成します。

        Parameters
        ----------
            df: pd.DataFrame
                風配図を作成するためのデータフレーム
            output_dirpath: str | Path | None, optional
                生成された図を保存するディレクトリのパス。デフォルトはNone
            output_filename: str, optional
                保存するファイル名。デフォルトは"edp_wind_rose.png"
            col_datetime: str, optional
                日時を示すカラム名。デフォルトは"Date"
            col_ch4_flux: str, optional
                CH4フラックスを示すカラム名。デフォルトは"Fch4"
            col_c2h6_flux: str, optional
                C2H6フラックスを示すカラム名。デフォルトは"Fc2h6"
            col_wind_dir: str, optional
                風向を示すカラム名。デフォルトは"Wind direction"
            flux_unit: str, optional
                フラックスの単位。デフォルトは"(nmol m$^{-2}$ s$^{-1}$)"
            ymax: float | None, optional
                y軸の上限値。指定しない場合はデータの最大値に基づいて自動設定。デフォルトはNone
            color_bio: str, optional
                生物起源のフラックスに対する色。デフォルトは"blue"
            color_gas: str, optional
                都市ガス起源のフラックスに対する色。デフォルトは"red"
            label_bio: str, optional
                生物起源のフラックスに対するラベル。デフォルトは"生物起源"
            label_gas: str, optional
                都市ガス起源のフラックスに対するラベル。デフォルトは"都市ガス起源"
            flux_alpha: float, optional
                フラックスの透明度。デフォルトは0.4
            num_directions: int, optional
                風向の数。デフォルトは8
            gap_degrees: float, optional
                セクター間の隙間の大きさ(度数)。0の場合は隙間なし。デフォルトは0.0
            center_on_angles: bool, optional
                45度刻みの線を境界として扇形を描画するかどうか。Trueの場合は境界として、Falseの場合は中間(22.5度)を中心として描画。デフォルトはTrue
            subplot_label: str | None, optional
                サブプロットに表示するラベル。デフォルトはNone
            add_legend: bool, optional
                凡例を表示するかどうか。デフォルトはTrue
            stack_bars: bool, optional
                生物起源の上に都市ガス起源を積み上げるかどうか。Trueの場合は積み上げ、Falseの場合は両方を0から積み上げ。デフォルトはTrue
            print_summary: bool, optional
                統計情報を表示するかどうか。デフォルトはFalse
            figsize: tuple[float, float], optional
                プロットのサイズ。デフォルトは(8, 8)
            dpi: float | None, optional
                プロットのdpi。デフォルトは350
            save_fig: bool, optional
                図を保存するかどうか。デフォルトはTrue
            show_fig: bool, optional
                図を表示するかどうか。デフォルトはTrue

        Returns
        ----------
            None

        Examples
        ----------
        >>> # 基本的な使用方法
        >>> generator = MonthlyFiguresGenerator()
        >>> generator.plot_wind_rose_sources(
        ...     df=data,
        ...     output_dirpath="output/figures",
        ...     output_filename="wind_rose_2023.png"
        ... )
        
        >>> # カスタマイズした例
        >>> generator.plot_wind_rose_sources(
        ...     df=data,
        ...     num_directions=16,  # 16方位で表示
        ...     stack_bars=False,   # 積み上げない
        ...     color_bio="green",  # 色を変更
        ...     color_gas="orange"
        ... )
        """
        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            col_ch4_flux=col_ch4_flux,
            col_c2h6_flux=col_c2h6_flux,
            col_datetime=col_datetime,
        )

        # 方位の定義
        direction_ranges = self._define_direction_ranges(
            num_directions, center_on_angles
        )

        # 方位ごとのデータを集計
        direction_data = self._aggregate_direction_data(
            df_with_sources, col_wind_dir, direction_ranges
        )

        # プロットの作成
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="polar")

        # 方位の角度(ラジアン)を計算
        theta = np.array(
            [np.radians(angle) for angle in direction_data["center_angle"]]
        )

        # セクターの幅を計算(隙間を考慮)
        sector_width = np.radians((360.0 / num_directions) - gap_degrees)

        # 積み上げ方式に応じてプロット
        if stack_bars:
            # 生物起源を基準として描画
            ax.bar(
                theta,
                direction_data["bio_flux"],
                width=sector_width,  # 隙間を考慮した幅
                bottom=0.0,
                color=color_bio,
                alpha=flux_alpha,
                label=label_bio,
            )
            # 都市ガス起源を生物起源の上に積み上げ
            ax.bar(
                theta,
                direction_data["gas_flux"],
                width=sector_width,  # 隙間を考慮した幅
                bottom=direction_data["bio_flux"],
                color=color_gas,
                alpha=flux_alpha,
                label=label_gas,
            )
        else:
            # 両方を0から積み上げ
            ax.bar(
                theta,
                direction_data["bio_flux"],
                width=sector_width,  # 隙間を考慮した幅
                bottom=0.0,
                color=color_bio,
                alpha=flux_alpha,
                label=label_bio,
            )
            ax.bar(
                theta,
                direction_data["gas_flux"],
                width=sector_width,  # 隙間を考慮した幅
                bottom=0.0,
                color=color_gas,
                alpha=flux_alpha,
                label=label_gas,
            )

        # y軸の範囲を設定
        if ymax is not None:
            ax.set_ylim(0, ymax)
        else:
            # データの最大値に基づいて自動設定
            max_value = max(
                direction_data["bio_flux"].max(), direction_data["gas_flux"].max()
            )
            ax.set_ylim(0, max_value * 1.1)  # 最大値の1.1倍を上限に設定

        # 方位ラベルの設定
        # 北を上に設定
        ax.set_theta_zero_location("N")  # type:ignore
        # 時計回りに設定
        ax.set_theta_direction(-1)  # type:ignore

        # 方位ラベルの表示
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        angles = np.radians(np.linspace(0, 360, len(labels), endpoint=False))
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        # プロット領域の調整(上部と下部にスペースを確保)
        plt.subplots_adjust(
            top=0.8,  # 上部に20%のスペースを確保
            bottom=0.2,  # 下部に20%のスペースを確保(凡例用)
        )

        # サブプロットラベルの追加(デフォルトは左上)
        if subplot_label:
            ax.text(
                0.01,
                0.99,
                subplot_label,
                transform=ax.transAxes,
            )

        # 単位の追加(図の下部中央に配置)
        plt.figtext(
            0.5,  # x位置(中央)
            0.1,  # y位置(下部)
            flux_unit,
            ha="center",  # 水平方向の位置揃え
            va="bottom",  # 垂直方向の位置揃え
        )

        # 凡例の追加(単位の下に配置)
        if add_legend:
            # 最初のプロットから凡例のハンドルとラベルを取得
            handles, labels = ax.get_legend_handles_labels()
            # 図の下部に凡例を配置
            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.05),  # x=0.5で中央、y=0.05で下部に配置
                ncol=len(handles),  # ハンドルの数だけ列を作成(一行に表示)
            )

        # グラフの保存または表示
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

        # 統計情報の表示
        if print_summary:
            for source in ["gas", "bio"]:
                flux_data = direction_data[f"{source}_flux"]
                mean_val = flux_data.mean()
                max_val = flux_data.max()
                max_dir = direction_data.loc[flux_data.idxmax(), "name"]

                self.logger.info(
                    f"{label_gas if source == 'gas' else label_bio}の統計:"
                )
                print(f"  平均フラックス: {mean_val:.2f}")
                print(f"  最大フラックス: {max_val:.2f}")
                print(f"  最大フラックスの方位: {max_dir}")

    def _define_direction_ranges(
        self,
        num_directions: int = 8,
        center_on_angles: bool = False,
    ) -> pd.DataFrame:
        """方位の範囲を定義

        Parameters
        ----------
            num_directions: int
                方位の数(デフォルトは8)
            center_on_angles: bool
                Trueの場合、45度刻みの線を境界として扇形を描画します。
                Falseの場合、45度の中間(22.5度)を中心として扇形を描画します。

        Returns
        ----------
        pd.DataFrame
            方位の定義を含むDataFrame
        """
        if num_directions == 8:
            if center_on_angles:
                # 45度刻みの線を境界とする場合
                directions = pd.DataFrame(
                    {
                        "name": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                        "center_angle": [
                            22.5,
                            67.5,
                            112.5,
                            157.5,
                            202.5,
                            247.5,
                            292.5,
                            337.5,
                        ],
                    }
                )
            else:
                # 従来通り45度を中心とする場合
                directions = pd.DataFrame(
                    {
                        "name": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                        "center_angle": [0, 45, 90, 135, 180, 225, 270, 315],
                    }
                )
        else:
            raise ValueError(f"現在{num_directions}方位はサポートされていません")

        # 各方位の範囲を計算
        angle_range = 360 / num_directions
        directions["start_angle"] = directions["center_angle"] - angle_range / 2
        directions["end_angle"] = directions["center_angle"] + angle_range / 2

        # -180度から180度の範囲に正規化
        directions["start_angle"] = np.where(
            directions["start_angle"] > 180,
            directions["start_angle"] - 360,
            directions["start_angle"],
        )
        directions["end_angle"] = np.where(
            directions["end_angle"] > 180,
            directions["end_angle"] - 360,
            directions["end_angle"],
        )

        return directions

    def _aggregate_direction_data(
        self,
        df: pd.DataFrame,
        col_wind_dir: str,
        direction_ranges: pd.DataFrame,
    ) -> pd.DataFrame:
        """方位ごとのフラックスデータを集計

        Parameters
        ----------
            df: pd.DataFrame
                ソース分離済みのデータフレーム
            col_wind_dir: str
                風向のカラム名
            direction_ranges: pd.DataFrame
                方位の定義

        Returns
        ----------
            pd.DataFrame
                方位ごとの集計データ
        """
        df_internal: pd.DataFrame = df.copy()
        result_data = direction_ranges.copy()
        result_data["gas_flux"] = 0.0
        result_data["bio_flux"] = 0.0
        for idx, row in direction_ranges.iterrows():
            if row["start_angle"] < row["end_angle"]:
                mask = (df_internal[col_wind_dir] > row["start_angle"]) & (
                    df_internal[col_wind_dir] <= row["end_angle"]
                )
            else:  # 北方向など、-180度と180度をまたぐ場合
                mask = (df_internal[col_wind_dir] > row["start_angle"]) | (
                    df_internal[col_wind_dir] <= row["end_angle"]
                )
            result_data.at[idx, "gas_flux"] = np.nanmean(
                pd.to_numeric(df_internal.loc[mask, "ch4_gas"])
            )
            result_data.at[idx, "bio_flux"] = np.nanmean(
                pd.to_numeric(df_internal.loc[mask, "ch4_bio"])
            )
        # NaNを0に置換
        result_data = result_data.fillna(0)
        return result_data

    def _calculate_source_contributions(
        self,
        df: pd.DataFrame,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        gas_ratio_c1c2: float = 0.076,
        col_datetime: str = "Date",
    ) -> pd.DataFrame:
        """
        CH4フラックスの都市ガス起源と生物起源の寄与を計算する。
        このロジックでは、燃焼起源のCH4フラックスは考慮せず計算している。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム
            col_ch4_flux: str
                CH4フラックスのカラム名
            col_c2h6_flux: str
                C2H6フラックスのカラム名
            gas_ratio_c1c2: float
                ガスのC2H6/CH4比(無次元)
            col_datetime: str
                日時カラムの名前

        Returns
        ----------
            pd.DataFrame
                起源別のフラックス値を含むデータフレーム
        """
        df_internal = df.copy()

        # 日時インデックスの処理
        if not isinstance(df_internal.index, pd.DatetimeIndex):
            df_internal[col_datetime] = pd.to_datetime(df_internal[col_datetime])
            df_internal.set_index(col_datetime, inplace=True)

        # C2H6/CH4比の計算
        df_internal["c2c1_ratio"] = (
            df_internal[col_c2h6_flux] / df_internal[col_ch4_flux]
        )

        # 都市ガスの標準組成に基づく都市ガス比率の計算
        df_internal["gas_ratio"] = df_internal["c2c1_ratio"] / gas_ratio_c1c2 * 100

        # gas_ratioに基づいて都市ガス起源と生物起源の寄与を比例配分
        df_internal["ch4_gas"] = df_internal[col_ch4_flux] * np.clip(
            df_internal["gas_ratio"] / 100, 0, 1
        )
        df_internal["ch4_bio"] = df_internal[col_ch4_flux] * (
            1 - np.clip(df_internal["gas_ratio"] / 100, 0, 1)
        )

        return df_internal

    def _prepare_diurnal_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        include_date_types: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """
        日変化パターンの計算に必要なデータを準備する。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム
            target_columns: list[str]
                計算対象の列名のリスト
            include_date_types: bool
                日付タイプ(平日/休日など)の分類を含めるかどうか

        Returns
        ----------
            tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]
                - 時間帯ごとの平均値を含むDataFrameの辞書
                - 24時間分の時間点
        """
        df_internal = df.copy()
        df_internal["hour"] = pd.to_datetime(df_internal["Date"]).dt.hour

        # 時間ごとの平均値を計算する関数
        def calculate_hourly_means(data_df, condition=None):
            if condition is not None:
                data_df = data_df[condition]
            return data_df.groupby("hour")[target_columns].mean().reset_index()

        # 基本の全日データを計算
        hourly_means = {"all": calculate_hourly_means(df_internal)}

        # 日付タイプによる分類が必要な場合
        if include_date_types:
            dates = pd.to_datetime(df_internal["Date"])
            is_weekend = dates.dt.dayofweek.isin([5, 6])
            is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
            is_weekday = ~(is_weekend | is_holiday)

            hourly_means.update(
                {
                    "weekday": calculate_hourly_means(df_internal, is_weekday),
                    "weekend": calculate_hourly_means(df_internal, is_weekend),
                    "holiday": calculate_hourly_means(
                        df_internal, is_weekend | is_holiday
                    ),
                }
            )

        # 24時目のデータを追加
        for col in hourly_means:
            last_row = hourly_means[col].iloc[0:1].copy()
            last_row["hour"] = 24
            hourly_means[col] = pd.concat(
                [hourly_means[col], last_row], ignore_index=True
            )

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        return hourly_means, time_points

    def _setup_diurnal_axes(
        self,
        ax: Axes,
        time_points: pd.DatetimeIndex,
        ylabel: str,
        subplot_label: str | None = None,
        add_label: bool = True,
        add_legend: bool = True,
        subplot_fontsize: int = 20,
    ) -> None:
        """日変化プロットの軸の設定を行う

        Parameters
        ----------
            ax: plt.Axes
                設定対象の軸
            time_points: pd.DatetimeIndex
                時間軸のポイント
            ylabel: str
                y軸のラベル
            subplot_label: str | None
                サブプロットのラベル
            add_label: bool
                軸ラベルを表示するかどうか
            add_legend: bool
                凡例を表示するかどうか
            subplot_fontsize: int
                サブプロットのフォントサイズ
        """
        if add_label:
            ax.set_xlabel("Time (hour)")
            ax.set_ylabel(ylabel)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(
            float(mdates.date2num(time_points[0])),
            float(mdates.date2num(time_points[-1])),
        )
        ax.set_xticks(time_points[::6])
        ax.set_xticklabels(["0", "6", "12", "18", "24"])

        if subplot_label:
            ax.text(
                0.02,
                0.98,
                subplot_label,
                transform=ax.transAxes,
                va="top",
                fontsize=subplot_fontsize,
            )

        if add_legend:
            ax.legend()

    @staticmethod
    def get_valid_data(df: pd.DataFrame, col_x: str, col_y: str) -> pd.DataFrame:
        """指定された列の有効なデータ(NaNを除いた)を取得します。

        Parameters
        ----------
            df: pd.DataFrame
                データフレームを指定します。
            col_x: str
                X軸の列名を指定します。
            col_y: str
                Y軸の列名を指定します。

        Returns
        ----------
            pd.DataFrame
                有効なデータのみを含むDataFrameを返します。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'x': [1, 2, np.nan, 4],
        ...     'y': [1, np.nan, 3, 4]
        ... })
        >>> valid_df = MonthlyFiguresGenerator.get_valid_data(df, 'x', 'y')
        >>> print(valid_df)
           x  y
        0  1  1
        3  4  4
        """
        return df.copy().dropna(subset=[col_x, col_y])
