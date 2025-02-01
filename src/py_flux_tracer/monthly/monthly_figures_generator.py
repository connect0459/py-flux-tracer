import os
import glob
import jpholiday
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from scipy import linalg, stats
from matplotlib.ticker import FuncFormatter, MultipleLocator
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from ..campbell.eddy_data_preprocessor import EddyDataPreprocessor
from ..campbell.spectrum_calculator import SpectrumCalculator


# 移動平均の計算関数
def calculate_rolling_stats(data: pd.Series, window: int, confidence_interval) -> tuple:
    """移動平均と信頼区間を計算する。

    Parameters
    ------
        data : pd.Series
            入力データ系列
        window : int
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

    # NaNを含むデータの処理（線形補間を行わない）
    data_cleaned = data.copy()

    # 移動平均の計算（NaNを含む場合はその期間の移動平均もNaNになる）
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
        クラスのコンストラクタ

        Parameters
        ------
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug : bool
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MonthlyFiguresGenerator.setup_logger(logger, log_level)

    def plot_c1c2_fluxes_timeseries(
        self,
        df,
        output_dir: str,
        output_filename: str = "timeseries.png",
        col_datetime: str = "Date",
        col_c1_flux: str = "Fch4_ultra",
        col_c2_flux: str = "Fc2h6_ultra",
    ):
        """
        月別のフラックスデータを時系列プロットとして出力する

        Parameters
        ------
            df : pd.DataFrame
                月別データを含むDataFrame
            output_dir : str
                出力ファイルを保存するディレクトリのパス
            output_filename : str
                出力ファイルの名前
            col_datetime : str
                日付を含む列の名前。デフォルトは"Date"。
            col_c1_flux : str
                CH4フラックスを含む列の名前。デフォルトは"Fch4_ultra"。
            col_c2_flux : str
                C2H6フラックスを含む列の名前。デフォルトは"Fc2h6_ultra"。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 図の作成
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # CH4フラックスのプロット
        ax1.scatter(df[col_datetime], df[col_c1_flux], color="red", alpha=0.5, s=20)
        ax1.set_ylabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax1.set_ylim(-100, 600)
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax2.scatter(
            df[col_datetime],
            df[col_c2_flux],
            color="orange",
            alpha=0.5,
            s=20,
        )
        ax2.set_ylabel(r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)")
        ax2.set_ylim(-20, 60)
        ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", fontsize=20)
        ax2.grid(True, alpha=0.3)

        # x軸の設定
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        plt.setp(ax2.get_xticklabels(), rotation=0, ha="right")
        ax2.set_xlabel("Month")

        # 図の保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_c1c2_concentrations_and_fluxes_timeseries(
        self,
        df: pd.DataFrame,
        output_dir: str,
        output_filename: str = "conc_flux_timeseries.png",
        col_datetime: str = "Date",
        col_ch4_conc: str = "CH4_ultra",
        col_ch4_flux: str = "Fch4_ultra",
        col_c2h6_conc: str = "C2H6_ultra",
        col_c2h6_flux: str = "Fc2h6_ultra",
        ylim_ch4_conc: tuple = (1.8, 2.6),  # CH4濃度のy軸範囲
        ylim_ch4_flux: tuple = (-100, 600),  # CH4フラックスのy軸範囲
        ylim_c2h6_conc: tuple = (-12, 20),  # C2H6濃度のy軸範囲
        ylim_c2h6_flux: tuple = (-20, 40),  # C2H6フラックスのy軸範囲
        print_summary: bool = False,
    ) -> None:
        """
        CH4とC2H6の濃度とフラックスの時系列プロットを作成する

        Parameters
        ------
            df : pd.DataFrame
                月別データを含むDataFrame
            output_dir : str
                出力ディレクトリのパス
            output_filename : str
                出力ファイル名
            col_datetime : str
                日付列の名前
            col_ch4_conc : str
                CH4濃度列の名前
            col_ch4_flux : str
                CH4フラックス列の名前
            col_c2h6_conc : str
                C2H6濃度列の名前
            col_c2h6_flux : str
                C2H6フラックス列の名前
            ylim_ch4_conc : tuple
                CH4濃度のy軸範囲
            ylim_ch4_flux : tuple
                CH4フラックスのy軸範囲
            ylim_c2h6_conc : tuple
                C2H6濃度のy軸範囲
            ylim_c2h6_flux : tuple
                C2H6フラックスのy軸範囲
            print_summary : bool
                解析情報をprintするかどうか
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        if print_summary:
            # 統計情報の計算と表示
            for name, col in [
                ("CH4 concentration", col_ch4_conc),
                ("CH4 flux", col_ch4_flux),
                ("C2H6 concentration", col_c2h6_conc),
                ("C2H6 flux", col_c2h6_flux),
            ]:
                # NaNを除外してから統計量を計算
                valid_data = df[col].dropna()

                if len(valid_data) > 0:
                    # quantileで計算（0-1の範囲）
                    quantile_05 = valid_data.quantile(0.05)
                    quantile_95 = valid_data.quantile(0.95)
                    mean_value = np.nanmean(valid_data)
                    positive_ratio = (valid_data > 0).mean() * 100

                    print(f"\n{name}:")
                    # パーセンタイルで表示（0-100の範囲）
                    print(
                        f"90パーセンタイルレンジ: {quantile_05:.2f} - {quantile_95:.2f}"
                    )
                    print(f"平均値: {mean_value:.2f}")
                    print(f"正の値の割合: {positive_ratio:.1f}%")
                else:
                    print(f"\n{name}: データが存在しません")

        # プロットの作成
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

        # CH4濃度のプロット
        ax1.scatter(df[col_datetime], df[col_ch4_conc], color="red", alpha=0.5, s=20)
        ax1.set_ylabel("CH$_4$ Concentration\n(ppm)")
        ax1.set_ylim(*ylim_ch4_conc)  # 引数からy軸範囲を設定
        ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top", fontsize=20)
        ax1.grid(True, alpha=0.3)

        # CH4フラックスのプロット
        ax2.scatter(df[col_datetime], df[col_ch4_flux], color="red", alpha=0.5, s=20)
        ax2.set_ylabel("CH$_4$ flux\n(nmol m$^{-2}$ s$^{-1}$)")
        ax2.set_ylim(*ylim_ch4_flux)  # 引数からy軸範囲を設定
        ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", fontsize=20)
        ax2.grid(True, alpha=0.3)

        # C2H6濃度のプロット
        ax3.scatter(
            df[col_datetime], df[col_c2h6_conc], color="orange", alpha=0.5, s=20
        )
        ax3.set_ylabel("C$_2$H$_6$ Concentration\n(ppb)")
        ax3.set_ylim(*ylim_c2h6_conc)  # 引数からy軸範囲を設定
        ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, va="top", fontsize=20)
        ax3.grid(True, alpha=0.3)

        # C2H6フラックスのプロット
        ax4.scatter(
            df[col_datetime], df[col_c2h6_flux], color="orange", alpha=0.5, s=20
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
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
                threshold = df[column_name].quantile(1 - top_percent / 100)
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

                # 時間帯ごとの分析（3時間区切り）
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
            analyze_top_values(df, col_ch4_conc)
            analyze_top_values(df, col_ch4_flux)
            analyze_top_values(df, col_c2h6_conc)
            analyze_top_values(df, col_c2h6_flux)

    def plot_c1c2_timeseries(
        self,
        df: pd.DataFrame,
        output_dir: str,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        output_filename: str = "timeseries_year.png",
        col_datetime: str = "Date",
        window_size: int = 24 * 7,  # 1週間の移動平均のデフォルト値
        confidence_interval: float = 0.95,  # 95%信頼区間
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        subplot_fontsize: int = 20,
        show_ci: bool = True,
        ch4_ylim: tuple[float, float] | None = None,
        c2h6_ylim: tuple[float, float] | None = None,
        start_date: str | None = None,  # 追加："YYYY-MM-DD"形式
        end_date: str | None = None,  # 追加："YYYY-MM-DD"形式
        figsize: tuple[float, float] = (16, 6),
    ) -> None:
        """CH4とC2H6フラックスの時系列変動をプロット

        Parameters
        ------
            df : pd.DataFrame
                データフレーム
            output_dir : str
                出力ディレクトリのパス
            col_ch4_flux : str
                CH4フラックスのカラム名
            col_c2h6_flux : str
                C2H6フラックスのカラム名
            output_filename : str
                出力ファイル名
            col_datetime : str
                日時カラムの名前
            window_size : int
                移動平均の窓サイズ
            confidence_interval : float
                信頼区間(0-1)
            subplot_label_ch4 : str | None
                CH4プロットのラベル
            subplot_label_c2h6 : str | None
                C2H6プロットのラベル
            subplot_fontsize : int
                サブプロットのフォントサイズ
            show_ci : bool
                信頼区間を表示するか
            ch4_ylim : tuple[float, float] | None
                CH4のy軸範囲
            c2h6_ylim : tuple[float, float] | None
                C2H6のy軸範囲
            start_date : str | None
                開始日（YYYY-MM-DD形式）
            end_date : str | None
                終了日（YYYY-MM-DD形式）
            figsize : tuple[float, float]
                図のサイズ。デフォルトは(16, 6)。
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        df_copied = df.copy()
        if not isinstance(df_copied.index, pd.DatetimeIndex):
            df_copied[col_datetime] = pd.to_datetime(df_copied[col_datetime])
            df_copied.set_index(col_datetime, inplace=True)

        # 日付範囲の処理
        if start_date is not None:
            start_dt = pd.to_datetime(start_date).normalize()  # 時刻を00:00:00に設定
            # df_min_date = (
            #     df_copied.index.normalize().min().normalize()
            # )  # 日付のみの比較のため正規化
            df_min_date = pd.to_datetime(df_copied.index.min()).normalize()

            # データの最小日付が指定開始日より後の場合にのみ警告
            if df_min_date.date() > start_dt.date():
                self.logger.warning(
                    f"指定された開始日{start_date}がデータの開始日{df_min_date.strftime('%Y-%m-%d')}より前です。"
                    f"データの開始日を使用します。"
                )
                start_dt = df_min_date
        else:
            # start_dt = df_copied.index.normalize().min()
            start_dt = pd.to_datetime(df_copied.index.min()).normalize()

        if end_date is not None:
            end_dt = (
                pd.to_datetime(end_date).normalize()
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            # df_max_date = (
            #     df_copied.index.normalize().max().normalize()
            # )  # 日付のみの比較のため正規化
            df_max_date = pd.to_datetime(df_copied.index.max()).normalize()

            # データの最大日付が指定終了日より前の場合にのみ警告
            if df_max_date.date() < pd.to_datetime(end_date).date():
                self.logger.warning(
                    f"指定された終了日{end_date}がデータの終了日{df_max_date.strftime('%Y-%m-%d')}より後です。"
                    f"データの終了日を使用します。"
                )
                end_dt = df_copied.index.max()
        else:
            end_dt = df_copied.index.max()

        # 指定された期間のデータを抽出
        mask = (df_copied.index >= start_dt) & (df_copied.index <= end_dt)
        df_copied = df_copied[mask]

        # CH4とC2H6の移動平均と信頼区間を計算
        ch4_mean, ch4_lower, ch4_upper = calculate_rolling_stats(
            df_copied[col_ch4_flux], window_size, confidence_interval
        )
        c2h6_mean, c2h6_lower, c2h6_upper = calculate_rolling_stats(
            df_copied[col_c2h6_flux], window_size, confidence_interval
        )

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # CH4プロット
        ax1.plot(df_copied.index, ch4_mean, "red", label="CH$_4$")
        if show_ci:
            ax1.fill_between(
                df_copied.index, ch4_lower, ch4_upper, color="red", alpha=0.2
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
        ax2.plot(df_copied.index, c2h6_mean, "orange", label="C$_2$H$_6$")
        if show_ci:
            ax2.fill_between(
                df_copied.index, c2h6_lower, c2h6_upper, color="orange", alpha=0.2
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

            # カスタムフォーマッタの作成（数字を通常フォントで表示）
            def date_formatter(x, p):
                date = mdates.num2date(x)
                return f"{date.strftime('%m')}"

            ax.xaxis.set_major_formatter(FuncFormatter(date_formatter))

            # 補助目盛りの設定
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            # ティックラベルの回転と位置調整
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_fluxes_comparison(
        self,
        df: pd.DataFrame,
        output_dir: str,
        cols_flux: list[str],
        labels: list[str],
        colors: list[str],
        output_filename: str = "ch4_flux_comparison.png",
        col_datetime: str = "Date",
        window_size: int = 24 * 7,  # 1週間の移動平均のデフォルト値
        confidence_interval: float = 0.95,  # 95%信頼区間
        subplot_label: str | None = None,
        subplot_fontsize: int = 20,
        show_ci: bool = True,
        y_lim: tuple[float, float] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_end_date: bool = True,
        figsize: tuple[float, float] = (12, 6),
        legend_loc: str = "upper right",
        apply_ma: bool = True,  # 移動平均を適用するかどうか
        hourly_mean: bool = False,  # 1時間平均を適用するかどうか
        x_interval: Literal["month", "10days"] = "month",  # "month" または "10days"
        xlabel: str = "Month",
        ylabel: str = "CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
        save_fig: bool = True,
        show_fig: bool = False,
    ) -> None:
        """複数のCH4フラックスの時系列比較プロット

        Parameters
        ------
            df : pd.DataFrame
                データフレーム
            output_dir : str
                出力ディレクトリのパス
            cols_flux : list[str]
                比較するフラックスのカラム名リスト
            labels : list[str]
                凡例に表示する各フラックスのラベルリスト
            colors : list[str]
                各フラックスの色リスト
            output_filename : str
                出力ファイル名
            col_datetime : str
                日時カラムの名前
            window_size : int
                移動平均の窓サイズ
            confidence_interval : float
                信頼区間(0-1)
            subplot_label : str | None
                プロットのラベル
            subplot_fontsize : int
                サブプロットのフォントサイズ
            show_ci : bool
                信頼区間を表示するか
            y_lim : tuple[float, float] | None
                y軸の範囲
            start_date : str | None
                開始日（YYYY-MM-DD形式）
            end_date : str | None
                終了日（YYYY-MM-DD形式）
            include_end_date : bool
                終了日を含めるかどうか。Falseの場合、終了日の前日までを表示
            figsize : tuple[float, float]
                図のサイズ
            legend_loc : str
                凡例の位置
            apply_ma : bool
                移動平均を適用するかどうか
            hourly_mean : bool
                1時間平均を適用するかどうか
            x_interval : Literal['month', '10days']
                x軸の目盛り間隔。"month"（月初めのみ）または"10days"（10日刻み）
            xlabel : str
                x軸のラベル（通常は"Month"）
            ylabel : str
                y軸のラベル（通常は"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)"）
            save_fig : bool
                図を保存するかどうか
            show_fig : bool
                図を表示するかどうか
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df[col_datetime] = pd.to_datetime(df[col_datetime])
            df.set_index(col_datetime, inplace=True)

        # 1時間平均の適用
        if hourly_mean:
            # 時間情報のみを使用してグループ化
            df = df.groupby([df.index.date, df.index.hour]).mean()
            # マルチインデックスを日時インデックスに変換
            df.index = pd.to_datetime(
                [f"{date} {hour:02d}:00:00" for date, hour in df.index]
            )

        # 日付範囲の処理
        if start_date is not None:
            start_dt = pd.to_datetime(start_date).normalize()  # 時刻を00:00:00に設定
            df_min_date = (
                df.index.normalize().min().normalize()
            )  # 日付のみの比較のため正規化

            # データの最小日付が指定開始日より後の場合にのみ警告
            if df_min_date.date() > start_dt.date():
                self.logger.warning(
                    f"指定された開始日{start_date}がデータの開始日{df_min_date.strftime('%Y-%m-%d')}より前です。"
                    f"データの開始日を使用します。"
                )
                start_dt = df_min_date
        else:
            start_dt = df.index.normalize().min()

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
                df.index.normalize().max().normalize()
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
                end_dt = df.index.max()
        else:
            end_dt = df.index.max()

        # 指定された期間のデータを抽出
        mask = (df.index >= start_dt) & (df.index <= end_dt)
        df = df[mask]

        # プロットの作成
        fig, ax = plt.subplots(figsize=figsize)

        # 各フラックスのプロット
        for flux_col, label, color in zip(cols_flux, labels, colors):
            if apply_ma:
                # 移動平均の計算
                mean, lower, upper = calculate_rolling_stats(
                    df[flux_col], window_size, confidence_interval
                )
                ax.plot(df.index, mean, color, label=label, alpha=0.7)
                if show_ci:
                    ax.fill_between(df.index, lower, upper, color=color, alpha=0.2)
            else:
                # 生データのプロット
                ax.plot(df.index, df[flux_col], color, label=label, alpha=0.7)

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
        ax.set_xlim(start_dt, end_dt)

        if x_interval == "month":
            # 月初めにメジャー線のみ表示
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_minor_locator(plt.NullLocator())  # マイナー線を非表示
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
                                # 30日が存在しない月（2月など）の場合は
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
                        np.array([mdates.date2num(date) for date in dates])
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

        ax.xaxis.set_major_formatter(plt.FuncFormatter(date_formatter))
        plt.setp(
            ax.xaxis.get_majorticklabels(), ha="center", rotation=0
        )  # 中央揃えに変更

        plt.tight_layout()

        if save_fig:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig)

    def plot_c1c2_fluxes_diurnal_patterns(
        self,
        df: pd.DataFrame,
        y_cols_ch4: list[str],
        y_cols_c2h6: list[str],
        labels_ch4: list[str],
        labels_c2h6: list[str],
        colors_ch4: list[str],
        colors_c2h6: list[str],
        output_dir: str,
        output_filename: str = "diurnal.png",
        legend_only_ch4: bool = False,
        add_label: bool = True,
        add_legend: bool = True,
        show_std: bool = False,  # 標準偏差表示のオプションを追加
        std_alpha: float = 0.2,  # 標準偏差の透明度
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        ax1_ylim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
    ) -> None:
        """CH4とC2H6の日変化パターンを1つの図に並べてプロットする

        Parameters
        ------
            df : pd.DataFrame
                入力データフレーム。
            y_cols_ch4 : list[str]
                CH4のプロットに使用するカラム名のリスト。
            y_cols_c2h6 : list[str]
                C2H6のプロットに使用するカラム名のリスト。
            labels_ch4 : list[str]
                CH4の各ラインに対応するラベルのリスト。
            labels_c2h6 : list[str]
                C2H6の各ラインに対応するラベルのリスト。
            colors_ch4 : list[str]
                CH4の各ラインに使用する色のリスト。
            colors_c2h6 : list[str]
                C2H6の各ラインに使用する色のリスト。
            output_dir : str
                出力先ディレクトリのパス。
            output_filename : str, optional
                出力ファイル名。デフォルトは"diurnal.png"。
            legend_only_ch4 : bool, optional
                CH4の凡例のみを表示するかどうか。デフォルトはFalse。
            add_label : bool, optional
                サブプロットラベルを表示するかどうか。デフォルトはTrue。
            add_legend : bool, optional
                凡例を表示するかどうか。デフォルトはTrue。
            show_std : bool, optional
                標準偏差を表示するかどうか。デフォルトはFalse。
            std_alpha : float, optional
                標準偏差の透明度。デフォルトは0.2。
            subplot_fontsize : int, optional
                サブプロットのフォントサイズ。デフォルトは20。
            subplot_label_ch4 : str | None, optional
                CH4プロットのラベル。デフォルトは"(a)"。
            subplot_label_c2h6 : str | None, optional
                C2H6プロットのラベル。デフォルトは"(b)"。
            ax1_ylim : tuple[float, float] | None, optional
                CH4プロットのy軸の範囲。デフォルトはNone。
            ax2_ylim : tuple[float, float] | None, optional
                C2H6プロットのy軸の範囲。デフォルトはNone。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        target_columns = y_cols_ch4 + y_cols_c2h6
        hourly_means, time_points = self._prepare_diurnal_data(df, target_columns)

        # 標準偏差の計算を追加
        hourly_stds = {}
        if show_std:
            hourly_stds = df.groupby(df.index.hour)[target_columns].std()
            # 24時間目のデータ点を追加
            last_hour = hourly_stds.iloc[0:1].copy()
            last_hour.index = [24]
            hourly_stds = pd.concat([hourly_stds, last_hour])

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # CH4のプロット (左側)
        ch4_lines = []
        for y_col, label, color in zip(y_cols_ch4, labels_ch4, colors_ch4):
            mean_values = hourly_means["all"][y_col]
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
                std_values = hourly_stds[y_col]
                ax1.fill_between(
                    time_points,
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=std_alpha,
                )

        # C2H6のプロット (右側)
        c2h6_lines = []
        for y_col, label, color in zip(y_cols_c2h6, labels_c2h6, colors_c2h6):
            mean_values = hourly_means["all"][y_col]
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
                std_values = hourly_stds[y_col]
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

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_c1c2_fluxes_diurnal_patterns_by_date(
        self,
        df: pd.DataFrame,
        y_col_ch4: str,
        y_col_c2h6: str,
        output_dir: str,
        output_filename: str = "diurnal_by_date.png",
        plot_all: bool = True,
        plot_weekday: bool = True,
        plot_weekend: bool = True,
        plot_holiday: bool = True,
        add_label: bool = True,
        add_legend: bool = True,
        show_std: bool = False,  # 標準偏差表示のオプションを追加
        std_alpha: float = 0.2,  # 標準偏差の透明度
        legend_only_ch4: bool = False,
        subplot_fontsize: int = 20,
        subplot_label_ch4: str | None = "(a)",
        subplot_label_c2h6: str | None = "(b)",
        ax1_ylim: tuple[float, float] | None = None,
        ax2_ylim: tuple[float, float] | None = None,
        print_summary: bool = False,  # 追加: 統計情報を表示するかどうか
    ) -> None:
        """CH4とC2H6の日変化パターンを日付分類して1つの図に並べてプロットする

        Parameters
        ------
            df : pd.DataFrame
                入力データフレーム。
            y_col_ch4 : str
                CH4フラックスを含むカラム名。
            y_col_c2h6 : str
                C2H6フラックスを含むカラム名。
            output_dir : str
                出力先ディレクトリのパス。
            output_filename : str, optional
                出力ファイル名。デフォルトは"diurnal_by_date.png"。
            plot_all : bool, optional
                すべての日をプロットするかどうか。デフォルトはTrue。
            plot_weekday : bool, optional
                平日をプロットするかどうか。デフォルトはTrue。
            plot_weekend : bool, optional
                週末をプロットするかどうか。デフォルトはTrue。
            plot_holiday : bool, optional
                祝日をプロットするかどうか。デフォルトはTrue。
            add_label : bool, optional
                サブプロットラベルを表示するかどうか。デフォルトはTrue。
            add_legend : bool, optional
                凡例を表示するかどうか。デフォルトはTrue。
            show_std : bool, optional
                標準偏差を表示するかどうか。デフォルトはFalse。
            std_alpha : float, optional
                標準偏差の透明度。デフォルトは0.2。
            legend_only_ch4 : bool, optional
                CH4の凡例のみを表示するかどうか。デフォルトはFalse。
            subplot_fontsize : int, optional
                サブプロットのフォントサイズ。デフォルトは20。
            subplot_label_ch4 : str | None, optional
                CH4プロットのラベル。デフォルトは"(a)"。
            subplot_label_c2h6 : str | None, optional
                C2H6プロットのラベル。デフォルトは"(b)"。
            ax1_ylim : tuple[float, float] | None, optional
                CH4プロットのy軸の範囲。デフォルトはNone。
            ax2_ylim : tuple[float, float] | None, optional
                C2H6プロットのy軸の範囲。デフォルトはNone。
            print_summary : bool, optional
                統計情報を表示するかどうか。デフォルトはTrue。
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        target_columns = [y_col_ch4, y_col_c2h6]
        hourly_means, time_points = self._prepare_diurnal_data(
            df, target_columns, include_date_types=True
        )

        # 標準偏差の計算を追加
        hourly_stds = {}
        if show_std:
            for condition in ["all", "weekday", "weekend", "holiday"]:
                if condition == "all":
                    condition_data = df
                elif condition == "weekday":
                    condition_data = df[
                        ~(
                            df.index.dayofweek.isin([5, 6])
                            | df.index.map(lambda x: jpholiday.is_holiday(x.date()))
                        )
                    ]
                elif condition == "weekend":
                    condition_data = df[df.index.dayofweek.isin([5, 6])]
                else:  # holiday
                    condition_data = df[
                        df.index.map(lambda x: jpholiday.is_holiday(x.date()))
                    ]

                hourly_stds[condition] = condition_data.groupby(
                    condition_data.index.hour
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
            if col in plot_conditions and plot_conditions[col]
        }

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # 日変化パターンの統計分析を追加
        if print_summary:
            # 平日と休日のデータを準備
            dates = pd.to_datetime(df.index)
            is_weekend = dates.dayofweek.isin([5, 6])
            is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
            is_weekday = ~(is_weekend | is_holiday)

            weekday_data = df[is_weekday]
            holiday_data = df[is_weekend | is_holiday]

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
        output_dir: str,
        col_ch4_conc: str = "CH4_ultra_cal",
        col_c2h6_conc: str = "C2H6_ultra_cal",
        col_datetime: str = "Date",
        output_filename: str = "diurnal_concentrations.png",
        show_std: bool = True,
        alpha_std: float = 0.2,
        add_legend: bool = True,  # 凡例表示のオプションを追加
        print_summary: bool = False,
        subplot_label_ch4: str | None = None,
        subplot_label_c2h6: str | None = None,
        subplot_fontsize: int = 24,
        ch4_ylim: tuple[float, float] | None = None,
        c2h6_ylim: tuple[float, float] | None = None,
        interval: str = "1H",  # "30min" または "1H" を指定
    ) -> None:
        """CH4とC2H6の濃度の日内変動を描画する

        Parameters
        ------
            df : pd.DataFrame
                濃度データを含むDataFrame
            output_dir : str
                出力ディレクトリのパス
            col_ch4_conc : str
                CH4濃度のカラム名
            col_c2h6_conc : str
                C2H6濃度のカラム名
            col_datetime : str
                日時カラム名
            output_filename : str
                出力ファイル名
            show_std : bool
                標準偏差を表示するかどうか
            alpha_std : float
                標準偏差の透明度
            add_legend : bool
                凡例を追加するかどうか
            print_summary : bool
                統計情報を表示するかどうか
            subplot_label_ch4 : str | None
                CH4プロットのラベル
            subplot_label_c2h6 : str | None
                C2H6プロットのラベル
            subplot_fontsize : int
                サブプロットのフォントサイズ
            ch4_ylim : tuple[float, float] | None
                CH4のy軸範囲
            c2h6_ylim : tuple[float, float] | None
                C2H6のy軸範囲
            interval : str
                時間間隔。"30min"または"1H"を指定
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの準備
        df = df.copy()
        if interval == "30min":
            # 30分間隔の場合、時間と30分を別々に取得
            df["hour"] = pd.to_datetime(df[col_datetime]).dt.hour
            df["minute"] = pd.to_datetime(df[col_datetime]).dt.minute
            df["time_bin"] = df["hour"] + df["minute"].map({0: 0, 30: 0.5})
        else:
            # 1時間間隔の場合
            df["time_bin"] = pd.to_datetime(df[col_datetime]).dt.hour

        # 時間ごとの平均値と標準偏差を計算
        hourly_stats = df.groupby("time_bin")[[col_ch4_conc, col_c2h6_conc]].agg(
            ["mean", "std"]
        )

        # 最後のデータポイントを追加（最初のデータを使用）
        last_point = hourly_stats.iloc[0:1].copy()
        last_point.index = [
            hourly_stats.index[-1] + (0.5 if interval == "30min" else 1)
        ]
        hourly_stats = pd.concat([hourly_stats, last_point])

        # 時間軸の作成
        if interval == "30min":
            time_points = pd.date_range("2024-01-01", periods=49, freq="30min")
            x_ticks = [0, 6, 12, 18, 24]  # 主要な時間のティック
        else:
            time_points = pd.date_range("2024-01-01", periods=25, freq="1H")
            x_ticks = [0, 6, 12, 18, 24]

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
            # 補助目盛りは表示するが、グリッド線は表示しない
            # if interval == "30min":
            #     ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))
            #     ax.tick_params(which='minor', length=4)

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
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

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
        output_dir: str,
        col_ch4_flux: str = "Fch4",
        col_c2h6_flux: str = "Fc2h6",
        ch4_label: str = r"$\mathregular{CH_{4}}$フラックス",
        c2h6_label: str = r"$\mathregular{C_{2}H_{6}}$フラックス",
        col_datetime: str = "Date",
        output_filename: str = "diurnal_patterns.png",
        window_size: int = 6,  # 移動平均の窓サイズ
        show_std: bool = True,  # 標準偏差の表示有無
        alpha_std: float = 0.1,  # 標準偏差の透明度
    ) -> None:
        """CH4とC2H6フラックスの日変化パターンをプロットする

        Parameters
        ------
            df : pd.DataFrame
                データフレーム
            output_dir : str
                出力ディレクトリのパス
            col_ch4_flux : str
                CH4フラックスのカラム名
            col_c2h6_flux : str
                C2H6フラックスのカラム名
            ch4_label : str
                CH4フラックスのラベル
            c2h6_label : str
                C2H6フラックスのラベル
            col_datetime : str
                日時カラムの名前
            output_filename : str
                出力ファイル名
            window_size : int
                移動平均の窓サイズ（デフォルト6）
            show_std : bool
                標準偏差を表示するかどうか
            alpha_std : float
                標準偏差の透明度（0-1）
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # # プロットのスタイル設定
        # plt.rcParams.update({
        #     'font.size': 20,
        #     'axes.labelsize': 20,
        #     'axes.titlesize': 20,
        #     'xtick.labelsize': 20,
        #     'ytick.labelsize': 20,
        #     'legend.fontsize': 20,
        # })

        # 日時インデックスの処理
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df[col_datetime] = pd.to_datetime(df[col_datetime])
            df.set_index(col_datetime, inplace=True)

        # 時刻データの抽出とグループ化
        df["hour"] = df.index.hour
        hourly_means = df.groupby("hour")[[col_ch4_flux, col_c2h6_flux]].agg(
            ["mean", "std"]
        )

        # 24時間目のデータ点を追加（0時のデータを使用）
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = [24]
        hourly_means = pd.concat([hourly_means, last_hour])

        # 24時間分のデータポイントを作成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

        # プロットの作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

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
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        # 統計情報の表示（オプション）
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
        output_dir: str,
        col_ratio_1: str,
        col_ratio_2: str,
        label_1: str,
        label_2: str,
        color_1: str,
        color_2: str,
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
        save_fig: bool = True,
        show_fig: bool = False,
    ) -> None:
        """2つの比率の日変化を比較するプロット

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム
        output_dir : str
            出力ディレクトリ
        col_ratio_1 : str
            1つ目の比率のカラム名
        col_ratio_2 : str
            2つ目の比率のカラム名
        label_1 : str
            1つ目の比率のラベル
        label_2 : str
            2つ目の比率のラベル
        color_1 : str
            1つ目の比率の色
        color_2 : str
            2つ目の比率の色
        output_filename : str, optional
            出力ファイル名, by default "gas_ratio_diurnal.png"
        add_xlabel : bool, optional
            x軸ラベルを追加するかどうか, by default True
        add_ylabel : bool, optional
            y軸ラベルを追加するかどうか, by default True
        add_legend : bool, optional
            凡例を追加するかどうか, by default True
        xlabel : str, optional
            x軸のラベル, by default "Hour"
        ylabel : str, optional
            y軸のラベル, by default "都市ガスが占める排出比率 (%)"
        subplot_fontsize : int, optional
            サブプロットのフォントサイズ, by default 20
        subplot_label : str | None, optional
            サブプロットのラベル, by default None
        y_max : float | None, optional
            y軸の最大値, by default 100
        figsize : tuple[float, float], optional
            図のサイズ, by default (12, 5)
        save_fig : bool, optional
            図を保存するかどうか, by default True
        show_fig : bool, optional
            図を表示するかどうか, by default False
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        # 時刻でグループ化して平均を計算
        hourly_means = df.groupby(df.index.hour)[[col_ratio_1, col_ratio_2]].mean()
        hourly_stds = df.groupby(df.index.hour)[[col_ratio_1, col_ratio_2]].std()

        # 24時間目のデータ点を追加（0時のデータを使用）
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = [24]
        hourly_means = pd.concat([hourly_means, last_hour])

        last_hour_std = hourly_stds.iloc[0:1].copy()
        last_hour_std.index = [24]
        hourly_stds = pd.concat([hourly_stds, last_hour_std])

        # 24時間分の時刻を生成
        time_points = pd.date_range("2024-01-01", periods=25, freq="h")

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
        ax.set_xlim(time_points[0], time_points[-1])
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
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_scatter(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        output_dir: str,
        output_filename: str = "scatter.png",
        xlabel: str | None = None,
        ylabel: str | None = None,
        add_label: bool = True,
        x_axis_range: tuple | None = None,
        y_axis_range: tuple | None = None,
        fixed_slope: float = 0.076,
        show_fixed_slope: bool = False,
        x_scientific: bool = False,  # 追加：x軸を指数表記にするかどうか
        y_scientific: bool = False,  # 追加：y軸を指数表記にするかどうか
    ) -> None:
        """散布図を作成し、TLS回帰直線を描画します。

        Parameters
        ------
            df : pd.DataFrame
                プロットに使用するデータフレーム
            x_col : str
                x軸に使用する列名
            y_col : str
                y軸に使用する列名
            xlabel : str
                x軸のラベル
            ylabel : str
                y軸のラベル
            output_dir : str
                出力先ディレクトリ
            output_filename : str, optional
                出力ファイル名。デフォルトは"scatter.png"
            add_label : bool, optional
                軸ラベルを表示するかどうか。デフォルトはTrue
            x_axis_range : tuple, optional
                x軸の範囲。デフォルトはNone。
            y_axis_range : tuple, optional
                y軸の範囲。デフォルトはNone。
            fixed_slope : float, optional
                固定傾きを指定するための値。デフォルトは0.076
            show_fixed_slope : bool, optional
                固定傾きの線を表示するかどうか。デフォルトはFalse
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 有効なデータの抽出
        df = MonthlyFiguresGenerator.get_valid_data(df, x_col, y_col)

        # データの準備
        x = df[x_col].values
        y = df[y_col].values

        # データの中心化
        x_mean = np.mean(x)
        y_mean = np.mean(y)
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
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        # プロットの作成
        fig, ax = plt.subplots(figsize=(6, 6))

        # データ点のプロット
        ax.scatter(x, y, color="black")

        # データの範囲を取得
        if x_axis_range is None:
            x_axis_range = (df[x_col].min(), df[x_col].max())
        if y_axis_range is None:
            y_axis_range = (df[y_col].min(), df[y_col].max())

        # 回帰直線のプロット
        x_range = np.linspace(x_axis_range[0], x_axis_range[1], 150)
        y_range = slope * x_range + intercept
        ax.plot(x_range, y_range, "r", label="TLS regression")

        # 傾き固定の線を追加（フラグがTrueの場合）
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

        # 1:1の関係を示す点線（軸の範囲が同じ場合のみ表示）
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

        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_source_contributions_diurnal(
        self,
        df: pd.DataFrame,
        output_dir: str,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        col_datetime: str = "Date",
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "bio",
        label_gas: str = "gas",
        figsize: tuple[float, float] = (10, 6),
        flux_alpha: float = 0.6,
        output_filename: str = "source_contributions.png",
        window_size: int = 6,  # 移動平均の窓サイズ
        print_summary: bool = False,  # 統計情報を表示するかどうか
        add_xlabel: bool = True,
        add_ylabel: bool = True,
        add_legend: bool = True,
        smooth: bool = False,
        y_max: float = 100,  # y軸の上限値を追加
        subplot_label: str | None = None,
        subplot_fontsize: int = 20,
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の日変化を積み上げグラフとして表示

        Parameters
        ----------
            df : pd.DataFrame
                CH4フラックスデータを含むデータフレーム
            output_dir : str
                出力先のディレクトリ
            col_ch4_flux : str
                CH4フラックスの列名
            col_c2h6_flux : str
                C2H6フラックスの列名
            col_datetime : str, optional
                日時の列名（デフォルトは"Date"）
            color_bio : str, optional
                生物起源の色（デフォルトは青）
            color_gas : str, optional
                都市ガス起源の色（デフォルトは赤）
            label_bio : str, optional
                生物起源のラベル（デフォルトは"bio"）
            label_gas : str, optional
                都市ガスのラベル（デフォルトは"gas"）
            figsize : tuple[float, float], optional
                プロットのサイズ（デフォルトは(10, 6)）
            flux_alpha : float, optional
                フラックスの透明度（デフォルトは0.6）
            output_filename : str, optional
                出力ファイル名（デフォルトは"source_contributions.png"）
            window_size : int, optional
                移動平均の窓サイズ（デフォルトは6）
            print_summary : bool, optional
                統計情報を表示するかどうか（デフォルトはTrue）
            add_xlabel : bool, optional
                x軸のラベルを追加するかどうか（デフォルトはTrue）
            add_ylabel : bool, optional
                y軸のラベルを追加するかどうか（デフォルトはTrue）
            add_legend : bool, optional
                凡例を追加するかどうか（デフォルトはTrue）
            smooth : bool, optional
                移動平均を適用するかどうか（デフォルトはFalse）
            y_max : float, optional
                y軸の上限値（デフォルトは100）
            subplot_label : str | None, optional
                サブプロットのラベル（デフォルトはNone）
            subplot_fontsize : int, optional
                サブプロットラベルのフォントサイズ（デフォルトは20）
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            col_ch4_flux=col_ch4_flux,
            col_c2h6_flux=col_c2h6_flux,
            col_datetime=col_datetime,
        )

        # 時刻データの抽出とグループ化
        df_with_sources["hour"] = df_with_sources.index.hour
        hourly_means = df_with_sources.groupby("hour")[["ch4_gas", "ch4_bio"]].mean()

        # 24時間目のデータ点を追加（0時のデータを使用）
        last_hour = hourly_means.iloc[0:1].copy()
        last_hour.index = [24]
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

        # サブプロットラベルの追加（subplot_labelが指定されている場合）
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
        ax.set_xlim(time_points[0], time_points[-1])
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

        # グラフの保存
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
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
                gas_flux[daytime_hours].sum() / total_flux[daytime_hours].sum()
            ) * 100
            nighttime_gas_ratio = (
                gas_flux[nighttime_hours].sum() / total_flux[nighttime_hours].sum()
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
                f"  昼間（{daytime_range[0]}~{daytime_range[1] - 1}時）の都市ガス比率: {daytime_gas_ratio:.1f}%"
            )
            print(
                f"  夜間（{daytime_range[1] - 1}~{daytime_range[0]}時）の都市ガス比率: {nighttime_gas_ratio:.1f}%"
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
                daytime_data = data[daytime_hours]
                nighttime_data = data[nighttime_hours]

                daytime_mean = daytime_data.mean()
                nighttime_mean = nighttime_data.mean()

                self.logger.info(f"\n{source}の統計:")
                print(f"  平均値: {mean_val:.2f}")
                print(f"  最小値: {min_val:.2f} (Hour: {min_time})")
                print(f"  最大値: {max_val:.2f} (Hour: {max_time})")
                if min_val != 0:
                    print(f"  最大/最小比: {max_val / min_val:.2f}")
                print(
                    f"  昼間（{daytime_range[0]}~{daytime_range[1] - 1}時）の平均: {daytime_mean:.2f}"
                )
                print(
                    f"  夜間（{daytime_range[1] - 1}~{daytime_range[0]}時）の平均: {nighttime_mean:.2f}"
                )
                if nighttime_mean != 0:
                    print(f"  昼/夜比: {daytime_mean / nighttime_mean:.2f}")

    def plot_source_contributions_diurnal_by_date(
        self,
        df: pd.DataFrame,
        output_dir: str,
        col_ch4_flux: str,
        col_c2h6_flux: str,
        col_datetime: str = "Date",
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "bio",
        label_gas: str = "gas",
        figsize: tuple[float, float] = (12, 5),
        flux_alpha: float = 0.6,
        output_filename: str = "source_contributions_by_date.png",
        add_xlabel: bool = True,
        add_ylabel: bool = True,
        add_legend: bool = True,
        print_summary: bool = False,  # 統計情報を表示するかどうか,
        subplot_fontsize: int = 20,
        subplot_label_weekday: str | None = None,
        subplot_label_weekend: str | None = None,
        y_max: float | None = None,  # y軸の上限値
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の日変化を平日・休日別に表示

        Parameters
        ------
            df : pd.DataFrame
                データフレーム
            output_dir : str
                出力ディレクトリのパス
            col_ch4_flux : str
                CH4フラックスのカラム名
            col_c2h6_flux : str
                C2H6フラックスのカラム名
            col_datetime : str
                日時カラムの名前
            color_bio : str, optional
                生物起源の色（デフォルトは青）
            color_gas : str, optional
                都市ガス起源の色（デフォルトは赤）
            label_bio : str, optional
                生物起源のラベル（デフォルトは"bio"）
            label_gas : str, optional
                都市ガスのラベル（デフォルトは"gas"）
            figsize : tuple[float, float], optional
                プロットのサイズ（デフォルトは(10, 6)）
            output_filename : str
                出力ファイル名
            add_xlabel : bool, optional
                x軸のラベルを追加するかどうか（デフォルトはTrue）
            add_ylabel : bool, optional
                y軸のラベルを追加するかどうか（デフォルトはTrue）
            add_legend : bool
                凡例を表示するか
            subplot_fontsize : int
                サブプロットのフォントサイズ
            subplot_label_weekday : str | None
                平日グラフのラベル
            subplot_label_weekend : str | None
                休日グラフのラベル
            y_max : float | None
                y軸の上限値
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # 起源の計算
        df_with_sources = self._calculate_source_contributions(
            df=df,
            col_ch4_flux=col_ch4_flux,
            col_c2h6_flux=col_c2h6_flux,
            col_datetime=col_datetime,
        )

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
        for ax, data, label in [
            (ax1, data_weekday, "Weekdays"),
            (ax2, data_holiday, "Weekends & Holidays"),
        ]:
            # 時間ごとの平均値を計算
            hourly_means = data.groupby(data.index.hour)[["ch4_gas", "ch4_bio"]].mean()

            # 24時間目のデータ点を追加
            last_hour = hourly_means.iloc[0:1].copy()
            last_hour.index = [24]
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
            ax.set_xlim(time_points[0], time_points[-1])
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
                ncol=len(handles),  # ハンドルの数だけ列を作成（一行に表示）
            )
            # 凡例用のスペースを確保
            plt.subplots_adjust(bottom=0.2)  # 下部に30%のスペースを確保

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig=fig)

        # 統計情報の表示
        if print_summary:
            for data, label in [
                (data_weekday, "Weekdays"),
                (data_holiday, "Weekends & Holidays"),
            ]:
                hourly_means = data.groupby(data.index.hour)[
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
                daytime_gas = gas_flux[daytime_hours]
                daytime_bio = bio_flux[daytime_hours]
                daytime_total = daytime_gas + daytime_bio
                daytime_ratio = (daytime_gas.sum() / daytime_total.sum()) * 100

                # 夜間の統計
                nighttime_gas = gas_flux[nighttime_hours]
                nighttime_bio = bio_flux[nighttime_hours]
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
                    f"  昼間（{daytime_range[0]}~{daytime_range[1] - 1}時）の割合: {daytime_ratio:.1f}%"
                )
                print(
                    f"  夜間（{daytime_range[1] - 1}~{daytime_range[0]}時）の割合: {nighttime_ratio:.1f}%"
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

    def plot_spectra(
        self,
        fs: float,
        lag_second: float,
        input_dir: str | Path | None,
        output_dir: str | Path | None,
        output_basename: str = "spectrum",
        col_ch4: str = "Ultra_CH4_ppm_C",
        col_c2h6: str = "Ultra_C2H6_ppb",
        col_tv: str = "Tv",
        label_ch4: str | None = None,
        label_c2h6: str | None = None,
        label_tv: str | None = None,
        file_pattern: str = "*.csv",
        markersize: float = 14,
        are_inputs_resampled: bool = True,
        save_fig: bool = True,
        show_fig: bool = True,
        plot_power: bool = True,
        plot_co: bool = True,
        add_tv_in_co: bool = True,
    ) -> None:
        """
        月間の平均パワースペクトル密度を計算してプロットする。

        データファイルを指定されたディレクトリから読み込み、パワースペクトル密度を計算し、
        結果を指定された出力ディレクトリにプロットして保存します。

        Parameters
        ------
            fs : float
                サンプリング周波数。
            lag_second : float
                ラグ時間（秒）。
            input_dir : str | Path | None
                データファイルが格納されているディレクトリ。
            output_dir : str | Path | None
                出力先ディレクトリ。
            col_ch4 : str, optional
                CH4の濃度データが入ったカラムのキー。デフォルトは"Ultra_CH4_ppm_C"。
            col_c2h6 : str, optional
                C2H6の濃度データが入ったカラムのキー。デフォルトは"Ultra_C2H6_ppb"。
            col_tv : str, optional
                気温データが入ったカラムのキー。デフォルトは"Tv"。
            label_ch4 : str | None, optional
                CH4のラベル。デフォルトはNone。
            label_c2h6 : str | None, optional
                C2H6のラベル。デフォルトはNone。
            label_tv : str | None, optional
                気温のラベル。デフォルトはNone。
            file_pattern : str, optional
                処理対象のファイルパターン。デフォルトは"*.csv"。
            markersize : float, optional
                プロットマーカーのサイズ。デフォルトは14。
            are_inputs_resampled : bool, optional
                入力データが再サンプリングされているかどうか。デフォルトはTrue。
            save_fig : bool, optional
                図を保存するかどうか。デフォルトはTrue。
            show_fig : bool, optional
                図を表示するかどうか。デフォルトはTrue。
            plot_power : bool, optional
                パワースペクトルをプロットするかどうか。デフォルトはTrue。
            plot_co : bool, optional
                COのスペクトルをプロットするかどうか。デフォルトはTrue。
            add_tv_in_co : bool, optional
                顕熱フラックスのコスペクトルを表示するかどうか。デフォルトはTrue。
        """
        # 出力ディレクトリの作成
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirに有効なディレクトリパスを指定する必要があります。"
                )
            os.makedirs(output_dir, exist_ok=True)

        # データの読み込みと結合
        edp = EddyDataPreprocessor(fs=fs)
        col_wind_w: str = EddyDataPreprocessor.WIND_W

        # 各変数のパワースペクトルを格納する辞書
        power_spectra = {col_ch4: [], col_c2h6: []}
        co_spectra = {col_ch4: [], col_c2h6: [], col_tv: []}
        freqs = None

        # プログレスバーを表示しながらファイルを処理
        file_list = glob.glob(os.path.join(input_dir, file_pattern))
        for filepath in tqdm(file_list, desc="Processing files"):
            df, _ = edp.get_resampled_df(
                filepath=filepath, resample_in_processing=are_inputs_resampled
            )

            # 風速成分の計算を追加
            df = edp.add_uvw_columns(df)

            # NaNや無限大を含む行を削除
            df = df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=[col_ch4, col_c2h6, col_tv, col_wind_w]
            )

            # データが十分な行数を持っているか確認
            if len(df) < 100:
                continue

            # 各ファイルごとにスペクトル計算
            calculator = SpectrumCalculator(
                df=df,
                fs=fs,
            )

            for col in power_spectra.keys():
                # 各変数のパワースペクトルを計算して保存
                if plot_power:
                    f, ps = calculator.calculate_power_spectrum(
                        col=col,
                        dimensionless=True,
                        frequency_weighted=True,
                        interpolate_points=True,
                        scaling="density",
                    )
                    # 最初のファイル処理時にfreqsを初期化
                    if freqs is None:
                        freqs = f
                        power_spectra[col].append(ps)
                    # 以降は周波数配列の長さが一致する場合のみ追加
                    elif len(f) == len(freqs):
                        power_spectra[col].append(ps)

                # コスペクトル
                if plot_co:
                    _, cs, _ = calculator.calculate_co_spectrum(
                        col1=col_wind_w,
                        col2=col,
                        dimensionless=True,
                        frequency_weighted=True,
                        interpolate_points=True,
                        scaling="spectrum",
                        apply_lag_correction_to_col2=True,
                        lag_second=lag_second,
                    )
                    if freqs is not None and len(cs) == len(freqs):
                        co_spectra[col].append(cs)

            # 顕熱フラックスのコスペクトル計算を追加
            if plot_co and add_tv_in_co:
                _, cs_heat, _ = calculator.calculate_co_spectrum(
                    col1=col_wind_w,
                    col2=col_tv,
                    dimensionless=True,
                    frequency_weighted=True,
                    interpolate_points=True,
                    scaling="spectrum",
                )
                if freqs is not None and len(cs_heat) == len(freqs):
                    co_spectra[col_tv].append(cs_heat)

        # 各変数のスペクトルを平均化
        if plot_power:
            averaged_power_spectra = {
                col: np.mean(spectra, axis=0) for col, spectra in power_spectra.items()
            }
        if plot_co:
            averaged_co_spectra = {
                col: np.mean(spectra, axis=0) for col, spectra in co_spectra.items()
            }
        # 顕熱フラックスの平均コスペクトル計算
        if plot_co and add_tv_in_co and co_spectra[col_tv]:
            averaged_heat_co_spectra = np.mean(co_spectra[col_tv], axis=0)

        # プロット設定を修正
        plot_configs = [
            {
                "col": col_ch4,
                "psd_ylabel": r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
                "co_ylabel": r"$fC_{w\mathrm{CH_4}} / \overline{w'\mathrm{CH_4}'}$",
                "color": "red",
                "label": label_ch4,
            },
            {
                "col": col_c2h6,
                "psd_ylabel": r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
                "co_ylabel": r"$fC_{w\mathrm{C_2H_6}} / \overline{w'\mathrm{C_2H_6}'}$",
                "color": "orange",
                "label": label_c2h6,
            },
        ]
        plot_tv_config = {
            "col": col_tv,
            "psd_ylabel": r"$fS_{T_v} / s_{T_v}^2$",
            "co_ylabel": r"$fC_{wT_v} / \overline{w'T_v'}$",
            "color": "blue",
            "label": label_tv,
        }

        # パワースペクトルの図を作成
        if plot_power:
            fig_power, axes_psd = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            for ax, config in zip(axes_psd, plot_configs):
                ax.plot(
                    freqs,
                    averaged_power_spectra[config["col"]],
                    "o",  # マーカーを丸に設定
                    color=config["color"],
                    markersize=markersize,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.001, 10)
                ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
                ax.text(0.1, 0.06, "-2/3", fontsize=18)
                ax.set_ylabel(config["psd_ylabel"])
                if config["label"] is not None:
                    ax.text(
                        0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
                    )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("f (Hz)")

            plt.tight_layout()

            if save_fig:
                output_path_psd: str = os.path.join(
                    output_dir, f"power_{output_basename}.png"
                )
                plt.savefig(
                    output_path_psd,
                    dpi=300,
                    bbox_inches="tight",
                )
            if show_fig:
                plt.show()
            else:
                plt.close(fig=fig_power)

        # コスペクトルの図を作成
        if plot_co:
            fig_co, axes_cosp = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            for ax, config in zip(axes_cosp, plot_configs):
                # 顕熱フラックスのコスペクトルを先に描画（背景として）
                if add_tv_in_co and len(co_spectra[col_tv]) > 0:
                    ax.plot(
                        freqs,
                        averaged_heat_co_spectra,
                        "o",
                        color="gray",
                        alpha=0.3,
                        markersize=markersize,
                        label=plot_tv_config["label"]
                        if plot_tv_config["label"]
                        else None,
                    )

                # CH4またはC2H6のコスペクトルを描画
                ax.plot(
                    freqs,
                    averaged_co_spectra[config["col"]],
                    "o",
                    color=config["color"],
                    markersize=markersize,
                    label=config["label"] if config["label"] else None,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.001, 10)
                # ax.plot([0.01, 10], [1, 0.01], "-", color="black", alpha=0.5)
                # ax.text(0.1, 0.1, "-4/3", fontsize=18)
                ax.set_ylabel(config["co_ylabel"])
                if config["label"] is not None:
                    ax.text(
                        0.02, 0.98, config["label"], transform=ax.transAxes, va="top"
                    )
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("f (Hz)")
                # 凡例を追加（顕熱フラックスが含まれる場合）
                if add_tv_in_co and label_tv:
                    ax.legend(loc="lower left")

            plt.tight_layout()
            if save_fig:
                output_path_csd: str = os.path.join(
                    output_dir, f"co_{output_basename}.png"
                )
                plt.savefig(
                    output_path_csd,
                    dpi=300,
                    bbox_inches="tight",
                )
            if show_fig:
                plt.show()
            else:
                plt.close(fig=fig_co)

    def plot_turbulence(
        self,
        df: pd.DataFrame,
        output_dir: str,
        output_filename: str = "turbulence.png",
        col_uz: str = "Uz",
        col_ch4: str = "Ultra_CH4_ppm_C",
        col_c2h6: str = "Ultra_C2H6_ppb",
        col_timestamp: str = "TIMESTAMP",
        add_serial_labels: bool = True,
        figsize: tuple[float, float] = (12, 10),
    ) -> None:
        """時系列データのプロットを作成する

        Parameters
        ------
            df : pd.DataFrame
                プロットするデータを含むDataFrame
            output_dir : str
                出力ディレクトリのパス
            output_filename : str
                出力ファイル名
            col_uz : str
                鉛直風速データのカラム名
            col_ch4 : str
                メタンデータのカラム名
            col_c2h6 : str
                エタンデータのカラム名
            col_timestamp : str
                タイムスタンプのカラム名
            add_serial_labels : bool
                シリアルラベルを追加するかどうかのフラグ
            figsize : tuple[float, float]
                プロットのサイズを指定するタプル
        """
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, output_filename)

        # データの前処理
        df = df.copy()

        # タイムスタンプをインデックスに設定（まだ設定されていない場合）
        if not isinstance(df.index, pd.DatetimeIndex):
            df[col_timestamp] = pd.to_datetime(df[col_timestamp])
            df.set_index(col_timestamp, inplace=True)

        # 開始時刻と終了時刻を取得
        start_time = df.index[0]
        end_time = df.index[-1]

        # 開始時刻の分を取得
        start_minute = start_time.minute

        # 時間軸の作成（実際の開始時刻からの経過分数）
        minutes_elapsed = (df.index - start_time).total_seconds() / 60

        # プロットの作成
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # 鉛直風速
        ax1.plot(minutes_elapsed, df[col_uz], "k-", linewidth=0.5)
        ax1.set_ylabel(r"$w$ (m s$^{-1}$)")
        if add_serial_labels:
            ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top")
        ax1.grid(True, alpha=0.3)

        # CH4濃度
        ax2.plot(minutes_elapsed, df[col_ch4], "r-", linewidth=0.5)
        ax2.set_ylabel(r"$\mathrm{CH_4}$ (ppm)")
        if add_serial_labels:
            ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top")
        ax2.grid(True, alpha=0.3)

        # C2H6濃度
        ax3.plot(minutes_elapsed, df[col_c2h6], "orange", linewidth=0.5)
        ax3.set_ylabel(r"$\mathrm{C_2H_6}$ (ppb)")
        if add_serial_labels:
            ax3.text(0.02, 0.98, "(c)", transform=ax3.transAxes, va="top")
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel("Time (minutes)")

        # x軸の範囲を実際の開始時刻から30分後までに設定
        total_minutes = (end_time - start_time).total_seconds() / 60
        ax3.set_xlim(0, min(30, total_minutes))

        # x軸の目盛りを5分間隔で設定
        np.arange(start_minute, start_minute + 35, 5)
        ax3.xaxis.set_major_locator(MultipleLocator(5))

        # レイアウトの調整
        plt.tight_layout()

        # 図の保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_wind_rose_sources(
        self,
        df: pd.DataFrame,
        output_dir: str | Path | None = None,
        output_filename: str = "edp_wind_rose.png",
        col_datetime: str = "Date",
        col_ch4_flux: str = "Fch4",
        col_c2h6_flux: str = "Fc2h6",
        col_wind_dir: str = "Wind direction",
        flux_unit: str = r"(nmol m$^{-2}$ s$^{-1}$)",
        ymax: float | None = None,  # フラックスの上限値
        color_bio: str = "blue",
        color_gas: str = "red",
        label_bio: str = "生物起源",
        label_gas: str = "都市ガス起源",
        figsize: tuple[float, float] = (8, 8),
        flux_alpha: float = 0.4,
        num_directions: int = 8,  # 方位の数（8方位）
        gap_degrees: float = 0.0,  # セクター間の隙間（度数）
        center_on_angles: bool = True,  # 追加：45度刻みの線を境界にするかどうか
        subplot_label: str | None = None,
        add_legend: bool = True,
        stack_bars: bool = True,  # 追加：積み上げ方式を選択するパラメータ
        print_summary: bool = False,  # 統計情報を表示するかどうか
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """CH4フラックスの都市ガス起源と生物起源の風配図を作成する関数

        Parameters
        ------
            df : pd.DataFrame
                風配図を作成するためのデータフレーム
            output_dir : str | Path | None
                生成された図を保存するディレクトリのパス
            output_filename : str
                保存するファイル名（デフォルトは"edp_wind_rose.png"）
            col_ch4_flux : str
                CH4フラックスを示すカラム名
            col_c2h6_flux : str
                C2H6フラックスを示すカラム名
            col_wind_dir : str
                風向を示すカラム名
            color_bio : str
                生物起源のフラックスに対する色
            color_gas : str
                都市ガス起源のフラックスに対する色
                風向を示すカラム名
            label_bio : str
                生物起源のフラックスに対するラベル
            label_gas : str
                都市ガス起源のフラックスに対するラベル
            col_datetime : str
                日時を示すカラム名
            num_directions : int
                風向の数（デフォルトは8）
            gap_degrees : float
                セクター間の隙間の大きさ（度数）。0の場合は隙間なし。
            center_on_angles: bool
                Trueの場合、45度刻みの線を境界として扇形を描画します。
                Falseの場合、45度の中間（22.5度）を中心として扇形を描画します。
            subplot_label : str
                サブプロットに表示するラベル
            print_summary : bool
                統計情報を表示するかどうかのフラグ
            flux_unit : str
                フラックスの単位
            ymax : float | None
                y軸の上限値（指定しない場合はデータの最大値に基づいて自動設定）
            figsize : tuple[float, float]
                図のサイズ
            flux_alpha : float
                フラックスの透明度
            stack_bars : bool, optional
                Trueの場合、生物起源の上に都市ガス起源を積み上げます（デフォルト）。
                Falseの場合、両方を0から積み上げます。
            save_fig : bool
                図を保存するかどうかのフラグ
            show_fig : bool
                図を表示するかどうかのフラグ
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
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")

        # 方位の角度（ラジアン）を計算
        theta = np.array(
            [np.radians(angle) for angle in direction_data["center_angle"]]
        )

        # セクターの幅を計算（隙間を考慮）
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
        ax.set_theta_zero_location("N")  # 北を上に設定
        ax.set_theta_direction(-1)  # 時計回りに設定

        # 方位ラベルの表示
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        angles = np.radians(np.linspace(0, 360, len(labels), endpoint=False))
        ax.set_xticks(angles)
        ax.set_xticklabels(labels)

        # プロット領域の調整（上部と下部にスペースを確保）
        plt.subplots_adjust(
            top=0.8,  # 上部に20%のスペースを確保
            bottom=0.2,  # 下部に20%のスペースを確保（凡例用）
        )

        # サブプロットラベルの追加（デフォルトは左上）
        if subplot_label:
            ax.text(
                0.01,
                0.99,
                subplot_label,
                transform=ax.transAxes,
            )

        # 単位の追加（図の下部中央に配置）
        plt.figtext(
            0.5,  # x位置（中央）
            0.1,  # y位置（下部）
            flux_unit,
            ha="center",  # 水平方向の位置揃え
            va="bottom",  # 垂直方向の位置揃え
        )

        # 凡例の追加（単位の下に配置）
        if add_legend:
            # 最初のプロットから凡例のハンドルとラベルを取得
            handles, labels = ax.get_legend_handles_labels()
            # 図の下部に凡例を配置
            fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, 0.05),  # x=0.5で中央、y=0.05で下部に配置
                ncol=len(handles),  # ハンドルの数だけ列を作成（一行に表示）
            )

        # グラフの保存
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirに有効なパスを指定する必要があります。"
                )
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)
            output_path: str = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        # グラフの表示
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
        ------
            num_directions : int
                方位の数（デフォルトは8）
            center_on_angles : bool
                Trueの場合、45度刻みの線を境界として扇形を描画します。
                Falseの場合、45度の中間（22.5度）を中心として扇形を描画します。

        Returns
        ------
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
        ------
            df : pd.DataFrame
                ソース分離済みのデータフレーム
            col_wind_dir : str
                風向のカラム名
            direction_ranges : pd.DataFrame
                方位の定義

        Returns
        ------
            pd.DataFrame
                方位ごとの集計データ
        """
        result_data = direction_ranges.copy()
        result_data["gas_flux"] = 0.0
        result_data["bio_flux"] = 0.0

        for idx, row in direction_ranges.iterrows():
            if row["start_angle"] < row["end_angle"]:
                mask = (df[col_wind_dir] > row["start_angle"]) & (
                    df[col_wind_dir] <= row["end_angle"]
                )
            else:  # 北方向など、-180度と180度をまたぐ場合
                mask = (df[col_wind_dir] > row["start_angle"]) | (
                    df[col_wind_dir] <= row["end_angle"]
                )

            result_data.loc[idx, "gas_flux"] = df.loc[mask, "ch4_gas"].mean()
            result_data.loc[idx, "bio_flux"] = df.loc[mask, "ch4_bio"].mean()

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
        ------
            df : pd.DataFrame
                入力データフレーム
            col_ch4_flux : str
                CH4フラックスのカラム名
            col_c2h6_flux : str
                C2H6フラックスのカラム名
            gas_ratio_c1c2 : float
                ガスのC2H6/CH4比（ppb/ppb）
            col_datetime : str
                日時カラムの名前

        Returns
        ------
            pd.DataFrame
                起源別のフラックス値を含むデータフレーム
        """
        df_copied = df.copy()

        # 日時インデックスの処理
        if not isinstance(df_copied.index, pd.DatetimeIndex):
            df_copied[col_datetime] = pd.to_datetime(df_copied[col_datetime])
            df_copied.set_index(col_datetime, inplace=True)

        # C2H6/CH4比の計算
        df_copied["c2c1_ratio"] = df_copied[col_c2h6_flux] / df_copied[col_ch4_flux]

        # 都市ガスの標準組成に基づく都市ガス比率の計算
        df_copied["gas_ratio"] = df_copied["c2c1_ratio"] / gas_ratio_c1c2 * 100

        # gas_ratioに基づいて都市ガス起源と生物起源の寄与を比例配分
        df_copied["ch4_gas"] = df_copied[col_ch4_flux] * np.clip(
            df_copied["gas_ratio"] / 100, 0, 1
        )
        df_copied["ch4_bio"] = df_copied[col_ch4_flux] * (
            1 - np.clip(df_copied["gas_ratio"] / 100, 0, 1)
        )

        return df_copied

    def _prepare_diurnal_data(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        include_date_types: bool = False,
    ) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
        """
        日変化パターンの計算に必要なデータを準備する。

        Parameters
        ------
            df : pd.DataFrame
                入力データフレーム
            target_columns : list[str]
                計算対象の列名のリスト
            include_date_types : bool
                日付タイプ（平日/休日など）の分類を含めるかどうか

        Returns
        ------
            tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]
                - 時間帯ごとの平均値を含むDataFrameの辞書
                - 24時間分の時間点
        """
        df = df.copy()
        df["hour"] = pd.to_datetime(df["Date"]).dt.hour

        # 時間ごとの平均値を計算する関数
        def calculate_hourly_means(data_df, condition=None):
            if condition is not None:
                data_df = data_df[condition]
            return data_df.groupby("hour")[target_columns].mean().reset_index()

        # 基本の全日データを計算
        hourly_means = {"all": calculate_hourly_means(df)}

        # 日付タイプによる分類が必要な場合
        if include_date_types:
            dates = pd.to_datetime(df["Date"])
            is_weekend = dates.dt.dayofweek.isin([5, 6])
            is_holiday = dates.map(lambda x: jpholiday.is_holiday(x.date()))
            is_weekday = ~(is_weekend | is_holiday)

            hourly_means.update(
                {
                    "weekday": calculate_hourly_means(df, is_weekday),
                    "weekend": calculate_hourly_means(df, is_weekend),
                    "holiday": calculate_hourly_means(df, is_weekend | is_holiday),
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
        ax: plt.Axes,
        time_points: pd.DatetimeIndex,
        ylabel: str,
        subplot_label: str | None = None,
        add_label: bool = True,
        add_legend: bool = True,
        subplot_fontsize: int = 20,
    ) -> None:
        """日変化プロットの軸の設定を行う

        Parameters
        ------
            ax : plt.Axes
                設定対象の軸
            time_points : pd.DatetimeIndex
                時間軸のポイント
            ylabel : str
                y軸のラベル
            subplot_label : str | None
                サブプロットのラベル
            add_label : bool
                軸ラベルを表示するかどうか
            add_legend : bool
                凡例を表示するかどうか
            subplot_fontsize : int
                サブプロットのフォントサイズ
        """
        if add_label:
            ax.set_xlabel("Time (hour)")
            ax.set_ylabel(ylabel)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-H"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18, 24]))
        ax.set_xlim(time_points[0], time_points[-1])
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
    def get_valid_data(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        指定された列の有効なデータ（NaNを除いた）を取得します。

        Parameters
        ------
            df : pd.DataFrame
                データフレーム
            x_col : str
                X軸の列名
            y_col : str
                Y軸の列名

        Returns
        ------
            pd.DataFrame
                有効なデータのみを含むDataFrame
        """
        return df.copy().dropna(subset=[x_col, y_col])

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        Parameters
        ------
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level : int
                ロガーのログレベル。デフォルトはINFO。

        Returns
        ------
            Logger
                設定されたロガーオブジェクト。
        """
        if logger is not None and isinstance(logger, Logger):
            return logger
        # 渡されたロガーがNoneまたは正しいものでない場合は独自に設定
        new_logger: Logger = getLogger()
        # 既存のハンドラーをすべて削除
        for handler in new_logger.handlers[:]:
            new_logger.removeHandler(handler)
        new_logger.setLevel(log_level)  # ロガーのレベルを設定
        ch = StreamHandler()
        ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
        new_logger.addHandler(ch)  # StreamHandlerの追加
        return new_logger

    @staticmethod
    def plot_fluxes_distributions(
        flux_data: dict[str, pd.Series],
        month: int,
        output_dir: str | Path | None = None,
        output_filename: str = "flux_distribution.png",
        colors: dict[str, str] | None = None,
        xlim: tuple[float, float] = (-50, 200),
        bandwidth: float = 1.0,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """複数のフラックスデータの分布を可視化

        Parameters
        ------
            flux_data : dict[str, pd.Series]
                各測器のフラックスデータを格納した辞書
                キー: 測器名, 値: フラックスデータ
            month : int
                測定月
            output_dir : str | Path | None
                出力ディレクトリ。指定しない場合はデフォルトのディレクトリに保存されます。
            output_filename : str
                出力ファイル名。デフォルトは"flux_distribution.png"です。
            colors : dict[str, str] | None
                各測器の色を指定する辞書。指定がない場合は自動で色を割り当てます。
            xlim : tuple[float, float]
                x軸の範囲。デフォルトは(-50, 200)です。
            bandwidth : float
                カーネル密度推定のバンド幅調整係数。デフォルトは1.0です。
            save_fig : bool
                図を保存するかどうか。デフォルトはTrueです。
            show_fig : bool
                図を表示するかどうか。デフォルトはTrueです。
        """
        # デフォルトの色を設定
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if colors is None:
            colors = {
                name: default_colors[i % len(default_colors)]
                for i, name in enumerate(flux_data.keys())
            }

        fig = plt.figure(figsize=(10, 6))

        # 統計情報を格納する辞書
        stats_info = {}

        # 各測器のデータをプロット
        for i, (name, flux) in enumerate(flux_data.items()):
            # nanを除去
            flux = flux.dropna()
            color = colors.get(name, default_colors[i % len(default_colors)])

            # KDEプロット
            sns.kdeplot(
                data=flux,
                label=name,
                color=color,
                alpha=0.5,
                bw_adjust=bandwidth,
            )

            # 平均値と中央値のマーカー
            mean_val = flux.mean()
            median_val = np.median(flux)
            plt.axvline(
                mean_val,
                color=color,
                linestyle="--",
                alpha=0.5,
                label=f"{name} mean",
            )
            plt.axvline(
                median_val,
                color=color,
                linestyle=":",
                alpha=0.5,
                label=f"{name} median",
            )

            # 統計情報を保存
            stats_info[name] = {
                "mean": mean_val,
                "median": median_val,
                "std": flux.std(),
            }

        # 軸ラベルとタイトル
        plt.xlabel(r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)")
        plt.ylabel("Probability Density")
        plt.title(f"Distribution of CH$_4$ fluxes - Month {month}")

        # x軸の範囲設定
        plt.xlim(xlim)

        # グリッド表示
        plt.grid(True, alpha=0.3)

        # 統計情報のテキスト作成
        stats_text = ""
        for name, stats_item in stats_info.items():
            stats_text += (
                f"{name}:\n"
                f"  Mean: {stats_item['mean']:.2f}\n"
                f"  Median: {stats_item['median']:.2f}\n"
                f"  Std: {stats_item['std']:.2f}\n"
            )

        # 統計情報の表示
        plt.text(
            0.02,
            0.98,
            stats_text.rstrip(),  # 最後の改行を削除
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 凡例の表示
        plt.legend(loc="upper right")
        plt.tight_layout()

        # グラフの保存
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirに有効なディレクトリパスを指定する必要があります。"
                )
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(output_dir, f"{output_filename.format(month=month)}"),
                dpi=300,
                bbox_inches="tight",
            )
        if show_fig:
            plt.show()
        plt.close(fig=fig)
