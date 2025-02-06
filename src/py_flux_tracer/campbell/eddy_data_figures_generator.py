import os
from dataclasses import dataclass
from logging import DEBUG, INFO, Logger
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from ..commons.utilities import setup_logger
from .eddy_data_preprocessor import EddyDataPreprocessor
from .spectrum_calculator import SpectrumCalculator


@dataclass
class SlopeLine:
    """傾き線の設定用のデータクラス

    Parameters
    ----------
        coordinates: tuple[tuple[float, float], tuple[float, float]]
            傾き線の始点と終点の座標。((x1, y1), (x2, y2))の形式で指定。
            x座標は周波数(Hz)、y座標は無次元化されたスペクトル値を表す。
        text: str
            傾き線に付随するテキスト(例:"-2/3", "-4/3"など)。
        text_pos: tuple[float, float] | None
            テキストを表示する位置の座標(x, y)。
            x座標は周波数(Hz)、y座標は無次元化されたスペクトル値を表す。
            Noneの場合、テキストは表示されない。
        fontsize: float, optional
            テキストのフォントサイズ。デフォルトは20。

    Examples
    --------
        >>> power_slope = SlopeLine(
        ...     coordinates=((0.01, 10), (10, 0.01)),
        ...     text="-2/3",
        ...     text_pos=(0.1, 0.06),
        ...     fontsize=18
        ... )
    """

    coordinates: tuple[tuple[float, float], tuple[float, float]]
    text: str
    text_pos: tuple[float, float] | None
    fontsize: float = 20

    def plot(self, ax: Axes) -> None:
        """傾き線とテキストを描画する

        Parameters
        ----------
            ax: matplotlib.axes.Axes
                描画対象のAxesオブジェクト
        """
        (x1, y1), (x2, y2) = self.coordinates
        ax.plot([x1, x2], [y1, y2], "-", color="black", alpha=0.5)
        if self.text_pos:
            ax.text(
                self.text_pos[0], self.text_pos[1], self.text, fontsize=self.fontsize
            )


@dataclass
class SpectralPlotConfig:
    """スペクトルプロット設定用のデータクラス

    Parameters
    ----------
        psd_ylabel: str
            パワースペクトル密度のy軸ラベル。
            LaTeXの数式表記が使用可能(例r"$fS_{\\mathrm{CH_4}} / s_{\\mathrm{CH_4}}^2$")。
        co_ylabel: str
            コスペクトルのy軸ラベル。
            LaTeXの数式表記が使用可能(例:r"$fC_{w\\mathrm{CH_4}} / \\overline{w'\\mathrm{CH_4}'}$")。
        color: str
            プロットの色。matplotlib.colorsで定義されている色名または16進数カラーコードを指定。
        label: str | None, optional
            凡例に表示するラベル。Noneの場合、凡例は表示されない。デフォルトはNone。

    Examples
    --------
        >>> ch4_config = SpectralPlotConfig(
        ...     psd_ylabel=r"$fS_{\\mathrm{CH_4}} / s_{\\mathrm{CH_4}}^2$",
        ...     co_ylabel=r"$fC_{w\\mathrm{CH_4}} / \\overline{w'\\mathrm{CH_4}'}$",
        ...     color="red",
        ...     label="CH4"
        ... )
    """

    psd_ylabel: str
    co_ylabel: str
    color: str
    label: str | None = None


class EddyDataFiguresGenerator:
    """
    データロガーの30分間データファイルから図を作成するクラス
    """

    def __init__(
        self,
        fs: float,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        Parameters
        ----------
            fs: float
                サンプリング周波数
            logger: Logger | None, optional
                ロガーオブジェクト。デフォルトはNone。
            logging_debug: bool, optional
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self._fs: float = fs
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

    def plot_c1c2_spectra(
        self,
        input_dirpath: str | Path,
        output_dirpath: str | Path,
        output_filename_power: str = "power_spectrum.png",
        output_filename_co: str = "co_spectrum.png",
        col_ch4: str = "Ultra_CH4_ppm_C",
        col_c2h6: str = "Ultra_C2H6_ppb",
        col_tv: str = "Tv",
        ch4_config: SpectralPlotConfig | None = None,
        c2h6_config: SpectralPlotConfig | None = None,
        tv_config: SpectralPlotConfig | None = None,
        lag_second: float | None = None,
        file_pattern: str = r"Eddy_(\d+)",
        file_suffix: str = ".dat",
        figsize: tuple[float, float] = (20, 6),
        dpi: float | None = 350,
        markersize: float = 14,
        xlim_power: tuple[float, float] | None = None,
        ylim_power: tuple[float, float] | None = None,
        xlim_co: tuple[float, float] | None = (0.001, 10),
        ylim_co: tuple[float, float] | None = (0.0001, 10),
        scaling_power: str = "density",
        scaling_co: str = "spectrum",
        power_slope: SlopeLine | None = None,
        co_slope: SlopeLine | None = None,
        are_configs_resampled: bool = True,
        save_fig: bool = True,
        show_fig: bool = True,
        plot_power: bool = True,
        plot_co: bool = True,
        add_tv_in_co: bool = True,
        xlabel: str = "f (Hz)",
    ) -> None:
        """月間平均のスペクトル密度を計算してプロットする。

        データファイルを指定されたディレクトリから読み込み、スペクトル密度を計算し、
        結果を指定された出力ディレクトリにプロットして保存します。

        Parameters
        ----------
            input_dirpath: str | Path
                データファイルが格納されているディレクトリ。
            output_dirpath: str | Path
                出力先ディレクトリ。
            output_filename_power: str, optional
                出力するパワースペクトルのファイル名。デフォルトは`"power_spectrum.png"`。
            output_filename_co: str, optional
                出力するコスペクトルのファイル名。デフォルトは`"co_spectrum.png"`。
            col_ch4: str, optional
                CH4の濃度データが入ったカラムのキー。デフォルトは`"Ultra_CH4_ppm_C"`。
            col_c2h6: str, optional
                C2H6の濃度データが入ったカラムのキー。デフォルトは`"Ultra_C2H6_ppb"`。
            col_tv: str, optional
                気温データが入ったカラムのキー。デフォルトは`"Tv"`。
            ch4_config: SpectralPlotConfig | None, optional
                CH4のプロット設定。Noneの場合はデフォルト設定を使用。
            c2h6_config: SpectralPlotConfig | None, optional
                C2H6のプロット設定。Noneの場合はデフォルト設定を使用。
            tv_config: SpectralPlotConfig | None, optional
                気温のプロット設定。Noneの場合はデフォルト設定を使用。
            lag_second: float | None, optional
                ラグ時間(秒)。デフォルトはNone。
            file_pattern: str, optional
                入力ファイルのパターン。デフォルトは`r"Eddy_(\\d+)"`。
            file_suffix: str, optional
                入力ファイルの拡張子。デフォルトは`".dat"`。
            figsize: tuple[float, float], optional
                プロットのサイズ。デフォルトは`(20, 6)`。
            dpi: float | None, optional
                プロットのdpi。デフォルトは`350`。
            markersize: float, optional
                プロットマーカーのサイズ。デフォルトは`14`。
            xlim_power: tuple[float, float] | None, optional
                パワースペクトルのx軸の範囲。デフォルトはNone。
            ylim_power: tuple[float, float] | None, optional
                パワースペクトルのy軸の範囲。デフォルトはNone。
            xlim_co: tuple[float, float] | None, optional
                コスペクトルのx軸の範囲。デフォルトは`(0.001, 10)`。
            ylim_co: tuple[float, float] | None, optional
                コスペクトルのy軸の範囲。デフォルトは`(0.0001, 10)`。
            scaling_power: str, optional
                パワースペクトルのスケーリング方法。`'spectrum'`または`'density'`などが指定可能。
                signal.welchのパラメータに渡すものと同様の値を取ることが可能。デフォルトは`"density"`。
            scaling_co: str, optional
                コスペクトルのスケーリング方法。`'spectrum'`または`'density'`などが指定可能。
                signal.welchのパラメータに渡すものと同様の値を取ることが可能。デフォルトは`"spectrum"`。
            power_slope: SlopeLine | None, optional
                パワースペクトルの傾き線設定。Noneの場合はデフォルト設定を使用。
            co_slope: SlopeLine | None, optional
                コスペクトルの傾き線設定。Noneの場合はデフォルト設定を使用。
            are_configs_resampled: bool, optional
                入力データが再サンプリングされているかどうか。デフォルトは`True`。
            save_fig: bool, optional
                図を保存するかどうか。デフォルトは`True`。
            show_fig: bool, optional
                図を表示するかどうか。デフォルトは`True`。
            plot_power: bool, optional
                パワースペクトルをプロットするかどうか。デフォルトは`True`。
            plot_co: bool, optional
                コスペクトルをプロットするかどうか。デフォルトは`True`。
            add_tv_in_co: bool, optional
                顕熱フラックスのコスペクトルを表示するかどうか。デフォルトは`True`。
            xlabel: str, optional
                x軸のラベル。デフォルトは`"f (Hz)"`。

        Examples
        --------
        >>> edfg = EddyDataFiguresGenerator(fs=10)
        >>> edfg.plot_c1c2_spectra(
        ...     input_dirpath="data/eddy",
        ...     output_dirpath="outputs",
        ...     output_filename_power="power.png",
        ...     output_filename_co="co.png"
        ... )
        """
        # 出力ディレクトリの作成
        if save_fig:
            os.makedirs(output_dirpath, exist_ok=True)

        # デフォルトのconfig設定
        if ch4_config is None:
            ch4_config = SpectralPlotConfig(
                psd_ylabel=r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
                co_ylabel=r"$fC_{w\mathrm{CH_4}} / \overline{w'\mathrm{CH_4}'}$",
                color="red",
                label="CH4",
            )

        if c2h6_config is None:
            c2h6_config = SpectralPlotConfig(
                psd_ylabel=r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
                co_ylabel=r"$fC_{w\mathrm{C_2H_6}} / \overline{w'\mathrm{C_2H_6}'}$",
                color="orange",
                label="C2H6",
            )

        if tv_config is None:
            tv_config = SpectralPlotConfig(
                psd_ylabel=r"$fS_{T_v} / s_{T_v}^2$",
                co_ylabel=r"$fC_{wT_v} / \overline{w'T_v'}$",
                color="blue",
                label="Tv",
            )

        # データの読み込みと結合
        edp = EddyDataPreprocessor(fs=self._fs)
        col_wind_w: str = EddyDataPreprocessor.WIND_W

        # 各変数のパワースペクトルを格納する辞書
        power_spectra = {col_ch4: [], col_c2h6: []}
        co_spectra = {col_ch4: [], col_c2h6: [], col_tv: []}
        freqs = None

        # ファイルリストの取得
        csv_files = edp._get_sorted_files(input_dirpath, file_pattern, file_suffix)
        if not csv_files:
            raise FileNotFoundError(
                f"file_suffix:'{file_suffix}'に一致するファイルが見つかりませんでした。"
            )

        for filename in tqdm(csv_files, desc="Processing files"):
            df, _ = edp.get_resampled_df(
                filepath=os.path.join(input_dirpath, filename),
                resample=are_configs_resampled,
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
                fs=self._fs,
            )

            for col in power_spectra.keys():
                # 各変数のパワースペクトルを計算して保存
                if plot_power:
                    f, ps = calculator.calculate_power_spectrum(
                        col=col,
                        dimensionless=True,
                        frequency_weighted=True,
                        interpolate_points=True,
                        scaling=scaling_power,
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
                        scaling=scaling_co,
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
                    scaling=scaling_co,
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

        # パワースペクトルの図を作成
        if plot_power:
            fig_power, axes_psd = plt.subplots(1, 2, figsize=figsize, sharex=True)
            configs = [(col_ch4, ch4_config), (col_c2h6, c2h6_config)]

            for ax, (col, config) in zip(axes_psd, configs, strict=True):
                ax.plot(
                    freqs,
                    averaged_power_spectra[col],
                    "o",
                    color=config.color,
                    markersize=markersize,
                )
                ax.set_xscale("log")
                ax.set_yscale("log")
                if xlim_power:
                    ax.set_xlim(*xlim_power)
                if ylim_power:
                    ax.set_ylim(*ylim_power)

                # 傾き線とテキストの追加
                if power_slope:
                    power_slope.plot(ax)

                ax.set_ylabel(config.psd_ylabel)
                if config.label is not None:
                    ax.text(0.02, 0.98, config.label, transform=ax.transAxes, va="top")
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xlabel)

            plt.tight_layout()

            if save_fig:
                output_filepath_psd: str = os.path.join(
                    output_dirpath, output_filename_power
                )
                plt.savefig(
                    output_filepath_psd,
                    dpi=dpi,
                    bbox_inches="tight",
                )
            if show_fig:
                plt.show()
            plt.close(fig=fig_power)

        # コスペクトルの図を作成
        if plot_co:
            fig_co, axes_cosp = plt.subplots(1, 2, figsize=figsize, sharex=True)
            configs = [(col_ch4, ch4_config), (col_c2h6, c2h6_config)]

            for ax, (col, config) in zip(axes_cosp, configs, strict=True):
                if add_tv_in_co:
                    ax.plot(
                        freqs,
                        averaged_heat_co_spectra,
                        "o",
                        color=tv_config.color,
                        alpha=0.3,
                        markersize=markersize,
                        label=tv_config.label,
                    )

                ax.plot(
                    freqs,
                    averaged_co_spectra[col],
                    "o",
                    color=config.color,
                    markersize=markersize,
                    label=config.label,
                )

                ax.set_xscale("log")
                ax.set_yscale("log")
                if xlim_co:
                    ax.set_xlim(*xlim_co)
                if ylim_co:
                    ax.set_ylim(*ylim_co)

                # 傾き線とテキストの追加
                if co_slope:
                    co_slope.plot(ax)

                ax.set_ylabel(config.co_ylabel)
                if config.label is not None:
                    ax.text(0.02, 0.98, config.label, transform=ax.transAxes, va="top")
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(xlabel)

                if add_tv_in_co and tv_config.label:
                    ax.legend(loc="lower left")

            plt.tight_layout()
            if save_fig:
                output_filepath_csd: str = os.path.join(
                    output_dirpath, output_filename_co
                )
                plt.savefig(
                    output_filepath_csd,
                    dpi=dpi,
                    bbox_inches="tight",
                )
            if show_fig:
                plt.show()
            plt.close(fig=fig_co)

    def plot_turbulence(
        self,
        df: pd.DataFrame,
        output_dirpath: str | Path | None = None,
        output_filename: str = "turbulence.png",
        col_uz: str = "Uz",
        col_ch4: str = "Ultra_CH4_ppm_C",
        col_c2h6: str = "Ultra_C2H6_ppb",
        col_timestamp: str = "TIMESTAMP",
        add_serial_labels: bool = True,
        figsize: tuple[float, float] = (12, 10),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """時系列データのプロットを作成する

        Parameters
        ------
            df: pd.DataFrame
                プロットするデータを含むDataFrame
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパス。デフォルトはNone。
            output_filename: str, optional
                出力ファイル名。デフォルトは"turbulence.png"。
            col_uz: str, optional
                鉛直風速データのカラム名。デフォルトは"Uz"。
            col_ch4: str, optional
                メタンデータのカラム名。デフォルトは"Ultra_CH4_ppm_C"。
            col_c2h6: str, optional
                エタンデータのカラム名。デフォルトは"Ultra_C2H6_ppb"。
            col_timestamp: str, optional
                タイムスタンプのカラム名。デフォルトは"TIMESTAMP"。
            add_serial_labels: bool, optional
                シリアルラベルを追加するかどうかのフラグ。デフォルトはTrue。
            figsize: tuple[float, float], optional
                プロットのサイズ。デフォルトは(12, 10)。
            dpi: float | None, optional
                プロットのdpi。デフォルトは350。
            save_fig: bool, optional
                プロットを保存するかどうか。デフォルトはTrue。
            show_fig: bool, optional
                プロットを表示するかどうか。デフォルトはTrue。

        Examples
        --------
        >>> edfg = EddyDataFiguresGenerator(fs=10)
        >>> edfg.plot_turbulence(df=data_frame)
        """
        # データの前処理
        df_internal = df.copy()
        df_internal.index = pd.to_datetime(df_internal.index)

        # タイムスタンプをインデックスに設定(まだ設定されていない場合)
        if not isinstance(df_internal.index, pd.DatetimeIndex):
            df_internal[col_timestamp] = pd.to_datetime(df_internal[col_timestamp])
            df_internal.set_index(col_timestamp, inplace=True)

        # 開始時刻と終了時刻を取得
        start_time = df_internal.index[0]
        end_time = df_internal.index[-1]

        # 開始時刻の分を取得
        start_minute = start_time.minute

        # 時間軸の作成(実際の開始時刻からの経過分数)
        minutes_elapsed = (df_internal.index - start_time).total_seconds() / 60

        # プロットの作成
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # 鉛直風速
        ax1.plot(minutes_elapsed, df_internal[col_uz], "k-", linewidth=0.5)
        ax1.set_ylabel(r"$w$ (m s$^{-1}$)")
        if add_serial_labels:
            ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, va="top")
        ax1.grid(True, alpha=0.3)

        # CH4濃度
        ax2.plot(minutes_elapsed, df_internal[col_ch4], "r-", linewidth=0.5)
        ax2.set_ylabel(r"$\mathrm{CH_4}$ (ppm)")
        if add_serial_labels:
            ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top")
        ax2.grid(True, alpha=0.3)

        # C2H6濃度
        ax3.plot(minutes_elapsed, df_internal[col_c2h6], "orange", linewidth=0.5)
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

        # グラフの保存または表示
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)
