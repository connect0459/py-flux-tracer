import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import curve_fit


@dataclass
class TfCurvesFromCsvConfig:
    """伝達関数曲線のプロット設定を保持するデータクラス"""

    col_coef_a: str  # 係数のカラム名
    label_gas: str  # ガスの表示ラベル
    base_color: str  # 平均線の色
    gas_name: str  # 出力ファイル用のガス名

    @classmethod
    def create_default_configs(cls) -> list["TfCurvesFromCsvConfig"]:
        """デフォルトの設定リストを生成"""
        return [
            cls("a_ch4-used", "CH$_4$", "red", "ch4"),
            cls("a_c2h6-used", "C$_2$H$_6$", "orange", "c2h6"),
        ]

    @classmethod
    def from_tuple(
        cls, config_tuple: tuple[str, str, str, str]
    ) -> "TfCurvesFromCsvConfig":
        """タプルから設定オブジェクトを生成"""
        return cls(*config_tuple)


class TransferFunctionCalculator:
    """
    このクラスは、CSVファイルからデータを読み込み、処理し、
    伝達関数を計算してプロットするための機能を提供します。

    この実装は Moore (1986) の論文に基づいています。
    """

    def __init__(
        self,
        filepath: str | Path,
        col_freq: str,
        cutoff_freq_low: float = 0.01,
        cutoff_freq_high: float = 1,
    ):
        """
        TransferFunctionCalculatorクラスのコンストラクタ。

        Parameters
        ----------
            filepath : str | Path
                分析対象のCSVファイルのパス。
            col_freq : str
                周波数のキー。
            cutoff_freq_low : float
                カットオフ周波数の最低値。
            cutoff_freq_high : float
                カットオフ周波数の最高値。
        """
        self._col_freq: str = col_freq
        self._cutoff_freq_low: float = cutoff_freq_low
        self._cutoff_freq_high: float = cutoff_freq_high
        self._df: pd.DataFrame = TransferFunctionCalculator._load_data(filepath)

    def calculate_transfer_function(
        self, col_reference: str, col_target: str
    ) -> tuple[float, float, pd.DataFrame]:
        """
        伝達関数の係数を計算する。

        Parameters
        ----------
            col_reference : str
                参照データのカラム名。
            col_target : str
                ターゲットデータのカラム名。

        Returns
        ----------
            tuple[float, float, pandas.DataFrame]
                伝達関数の係数aとその標準誤差、および計算に用いたDataFrame。
        """
        df_processed: pd.DataFrame = self.process_data(
            col_reference=col_reference, col_target=col_target
        )
        df_cutoff: pd.DataFrame = self._cutoff_df(df_processed)

        array_x = np.array(df_cutoff.index)
        array_y = np.array(df_cutoff["target"] / df_cutoff["reference"])

        # フィッティングパラメータと共分散行列を取得
        popt, pcov = curve_fit(
            TransferFunctionCalculator.transfer_function, array_x, array_y
        )

        # 標準誤差を計算（共分散行列の対角成分の平方根）
        perr = np.sqrt(np.diag(pcov))

        # 係数aとその標準誤差、および計算に用いたDataFrameを返す
        return popt[0], perr[0], df_processed

    def create_plot_co_spectra(
        self,
        col1: str,
        col2: str,
        color1: str = "gray",
        color2: str = "red",
        figsize: tuple[int, int] = (10, 6),
        dpi: float | None = 350,
        label1: str | None = None,
        label2: str | None = None,
        output_dirpath: str | Path | None = None,
        output_filename: str = "co.png",
        add_legend: bool = True,
        add_xy_labels: bool = True,
        legend_font_size: float = 16,
        save_fig: bool = True,
        show_fig: bool = True,
        subplot_label: str | None = None,
        window_size: int = 5,  # 移動平均の窓サイズ
        markersize: float = 14,
        xlim: tuple[float, float] = (0.0001, 10),
        ylim: tuple[float, float] = (0.0001, 10),
        slope_line: tuple[tuple[float, float], tuple[float, float]] = (
            (0.01, 10),
            (10, 0.001),
        ),
        slope_text: tuple[str, tuple[float, float]] = ("-4/3", (0.25, 0.4)),
        subplot_label_pos: tuple[float, float] = (0.00015, 3),
    ) -> None:
        """
        2種類のコスペクトルをプロットする。

        Parameters
        ----------
            col1 : str
                1つ目のコスペクトルデータのカラム名。
            col2 : str
                2つ目のコスペクトルデータのカラム名。
            color1 : str, optional
                1つ目のデータの色。デフォルトは'gray'。
            color2 : str, optional
                2つ目のデータの色。デフォルトは'red'。
            figsize : tuple[int, int], optional
                プロットのサイズ。デフォルトは(10, 6)。
            dpi : float | None, optional
                プロットのdpi。デフォルトは350。
            label1 : str, optional
                1つ目のデータのラベル名。デフォルトはNone。
            label2 : str, optional
                2つ目のデータのラベル名。デフォルトはNone。
            output_dirpath : str | Path | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_filename : str, optional
                保存するファイル名。デフォルトは"co.png"。
            add_legend : bool, optional
                凡例を追加するかどうか。デフォルトはTrue。
            legend_font_size : bool, optional
                凡例のフォントサイズ。デフォルトは16。
            save_fig : bool, optional
                プロットを保存するかどうか。デフォルトはTrue.
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
            subplot_label : str | None, optional
                左上に表示するサブプロットラベル。デフォルトは"(a)"。
            window_size : int, optional
                移動平均の窓サイズ。デフォルトは5。
            xlim : tuple[float, float], optional
                x軸の表示範囲。デフォルトは(0.0001, 10)。
            ylim : tuple[float, float], optional
                y軸の表示範囲。デフォルトは(0.0001, 10)。
            slope_line : tuple[tuple[float, float], tuple[float, float]], optional
                傾きを示す直線の始点と終点の座標。デフォルトは((0.01, 10), (10, 0.001))。
            slope_text : tuple[str, tuple[float, float]], optional
                傾きを示すテキストとその位置。デフォルトは("-4/3", (0.25, 0.4))。
            subplot_label_pos : tuple[float, float], optional
                サブプロットラベルの位置。デフォルトは(0.00015, 3)。
        """
        df_copied: pd.DataFrame = self._df.copy()
        # データの取得と移動平均の適用
        data1 = df_copied[df_copied[col1] > 0].groupby(self._col_freq)[col1].median()
        data2 = df_copied[df_copied[col2] > 0].groupby(self._col_freq)[col2].median()

        data1 = data1.rolling(window=window_size, center=True, min_periods=1).mean()
        data2 = data2.rolling(window=window_size, center=True, min_periods=1).mean()

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        # マーカーサイズを設定して見やすくする
        ax.plot(
            data1.index, data1, "o", color=color1, label=label1, markersize=markersize
        )
        ax.plot(
            data2.index, data2, "o", color=color2, label=label2, markersize=markersize
        )

        # 傾きを示す直線とテキストを追加
        (x1, y1), (x2, y2) = slope_line
        ax.plot([x1, x2], [y1, y2], "-", color="black")
        text, (text_x, text_y) = slope_text
        ax.text(text_x, text_y, text)

        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if add_xy_labels:
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("無次元コスペクトル")

        if add_legend:
            ax.legend(
                bbox_to_anchor=(0.05, 1),
                loc="lower left",
                fontsize=legend_font_size,
                ncol=3,
                frameon=False,
            )
        if subplot_label is not None:
            ax.text(*subplot_label_pos, subplot_label)
        fig.tight_layout()

        if save_fig and output_dirpath is not None:
            os.makedirs(output_dirpath, exist_ok=True)
            # プロットをPNG形式で保存
            fig.savefig(os.path.join(output_dirpath, output_filename), dpi=dpi)
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def create_plot_ratio(
        self,
        df_processed: pd.DataFrame,
        reference_name: str,
        target_name: str,
        figsize: tuple[int, int] = (10, 6),
        dpi: float | None = 350,
        output_dirpath: str | Path | None = None,
        output_filename: str = "ratio.png",
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        ターゲットと参照の比率をプロットする。

        Parameters
        ----------
            df_processed : pd.DataFrame
                処理されたデータフレーム。
            reference_name : str
                参照の名前。
            target_name : str
                ターゲットの名前。
            figsize : tuple[int, int], optional
                プロットのサイズ。デフォルトは(10, 6)。
            dpi : float | None, optional
                プロットのdpi。デフォルトは350。
            output_dirpath : str | Path | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_filename : str, optional
                保存するファイル名。デフォルトは"ratio.png"。
            save_fig : bool, optional
                プロットを保存するかどうか。デフォルトはTrue。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
        """
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        ax.plot(
            df_processed.index, df_processed["target"] / df_processed["reference"], "o"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f (Hz)")
        ax.set_ylabel(f"{target_name} / {reference_name}")
        ax.set_title(f"{target_name}と{reference_name}の比")

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirpathに有効なディレクトリパスを指定する必要があります。"
                )
            fig.savefig(os.path.join(output_dirpath, output_filename), dpi=dpi)
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    @classmethod
    def create_plot_tf_curves_from_csv(
        cls,
        filepath: str,
        config: TfCurvesFromCsvConfig,
        csv_encoding: str | None = "utf-8-sig",
        output_dirpath: str | Path | None = None,
        output_filename: str = "all_tf_curves.png",
        col_datetime: str = "Date",
        figsize: tuple[float, float] = (10, 6),
        dpi: float | None = 350,
        add_legend: bool = True,
        add_xlabel: bool = True,
        label_x: str = "f (Hz)",
        label_y: str = "無次元コスペクトル比",
        label_avg: str = "Avg.",
        label_co_ref: str = "Tv",
        legend_font_size: float = 16,
        line_colors: list[str] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        伝達関数の係数をプロットし、平均値を表示します。
        各ガスのデータをCSVファイルから読み込み、指定された設定に基づいてプロットを生成します。
        プロットはオプションで保存することも可能です。

        Parameters
        ----------
            filepath : str
                伝達関数の係数が格納されたCSVファイルのパス。
            config : TfCurvesFromCsvConfig
                プロット設定。
            output_dirpath : str | Path | None, optional
                出力ディレクトリ。Noneの場合は保存しない。
            output_filename : str, optional
                出力ファイル名。デフォルトは"all_tf_curves.png"。
            col_datetime : str, optional
                日付情報が格納されているカラム名。デフォルトは"Date"。
            figsize : tuple[float, float], optional
                プロットのサイズ。デフォルトは(10, 6)。
            dpi : float | None, optional
                プロットのdpi。デフォルトは350。
            add_legend : bool, optional
                凡例を追加するかどうか。デフォルトはTrue。
            add_xlabel : bool, optional
                x軸ラベルを追加するかどうか。デフォルトはTrue。
            label_x : str, optional
                x軸のラベル。デフォルトは"f (Hz)"。
            label_y : str, optional
                y軸のラベル。デフォルトは"無次元コスペクトル比"。
            label_avg : str, optional
                平均値のラベル。デフォルトは"Avg."。
            legend_font_size: float, optional
                凡例のフォントサイズ。デフォルトは16。
            line_colors : list[str] | None, optional
                各日付のデータに使用する色のリスト。
            save_fig : bool, optional
                プロットを保存するかどうか。デフォルトはTrue。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
        """
        # CSVファイルを読み込む
        df = pd.read_csv(filepath, encoding=csv_encoding)

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # データ数に応じたデフォルトの色リストを作成
        if line_colors is None:
            default_colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            n_dates = len(df)
            plot_colors = (default_colors * (n_dates // len(default_colors) + 1))[
                :n_dates
            ]
        else:
            plot_colors = line_colors

        # 全てのa値を用いて伝達関数をプロット
        for i, row in enumerate(df.iterrows()):
            a = row[1][config.col_coef_a]
            date = row[1][col_datetime]
            x_fit = np.logspace(-3, 1, 1000)
            y_fit = cls.transfer_function(x_fit, a)
            plt.plot(
                x_fit,
                y_fit,
                "-",
                color=plot_colors[i],
                alpha=0.7,
                label=f"{date} (a = {a:.3f})",
            )

        # 平均のa値を用いた伝達関数をプロット
        a_mean = df[config.col_coef_a].mean()
        x_fit = np.logspace(-3, 1, 1000)
        y_fit = cls.transfer_function(x_fit, a_mean)
        plt.plot(
            x_fit,
            y_fit,
            "-",
            color=config.base_color,
            linewidth=3,
            label=f"{label_avg} (a = {a_mean:.3f})",
        )

        # グラフの設定
        label_y_formatted: str = f"{label_y}\n({config.label_gas} / {label_co_ref})"
        plt.xscale("log")
        if add_xlabel:
            plt.xlabel(label_x)
        plt.ylabel(label_y_formatted)
        if add_legend:
            plt.legend(loc="lower left", fontsize=legend_font_size)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirpathに有効なディレクトリパスを指定する必要があります。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            # 出力ファイル名が指定されていない場合、gas_nameを使用
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, dpi=dpi, bbox_inches="tight")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def create_plot_transfer_function(
        self,
        a: float,
        df_processed: pd.DataFrame,
        reference_name: str,
        target_name: str,
        figsize: tuple[int, int] = (10, 6),
        dpi: float | None = 350,
        output_dirpath: str | Path | None = None,
        output_filename: str = "tf.png",
        save_fig: bool = True,
        show_fig: bool = True,
        add_xlabel: bool = True,
        label_x: str = "f (Hz)",
        label_y: str = "コスペクトル比",
        label_target: str | None = None,
        label_ref: str = "Tv",
    ) -> None:
        """
        伝達関数とそのフィットをプロットする。

        Parameters
        ----------
            a : float
                伝達関数の係数。
            df_processed : pd.DataFrame
                処理されたデータフレーム。
            reference_name : str
                参照の名前。
            target_name : str
                ターゲットの名前。
            figsize : tuple[int, int], optional
                プロットのサイズ。デフォルトは(10, 6)。
            dpi : float | None, optional
                プロットのdpi。デフォルトは350。
            output_dirpath : str | Path | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_filename : str, optional
                保存するファイル名。デフォルトは"tf.png"。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
            add_xlabel : bool, optional
                x軸のラベルを追加するかどうか。デフォルトはTrue。
            label_x : str, optional
                x軸のラベル名。デフォルトは"f (Hz)"。
            label_y : str, optional
                y軸のラベル名。デフォルトは"コスペクトル比"。
            label_target : str | None, optional
                比較先のラベル名。デフォルトはNone。
            label_ref : str, optional
                比較元のラベル名。デフォルトは"Tv"。
        """
        df_cutoff: pd.DataFrame = self._cutoff_df(df_processed)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        ax.plot(
            df_cutoff.index,
            df_cutoff["target"] / df_cutoff["reference"],
            "o",
            label=f"{target_name} / {reference_name}",
        )

        x_fit = np.logspace(
            np.log10(self._cutoff_freq_low), np.log10(self._cutoff_freq_high), 1000
        )
        y_fit = self.transfer_function(x_fit, a)
        ax.plot(x_fit, y_fit, "-", label=f"フィット (a = {a:.4f})")

        ax.set_xscale("log")
        # グラフの設定
        label_y_formatted: str = f"{label_y}\n({label_target} / {label_ref})"
        plt.xscale("log")
        if add_xlabel:
            plt.xlabel(label_x)
        plt.ylabel(label_y_formatted)
        ax.legend()

        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig=Trueのとき、output_dirpathに有効なディレクトリパスを指定する必要があります。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            # プロットをPNG形式で保存
            fig.savefig(os.path.join(output_dirpath, output_filename), dpi=dpi)
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def process_data(self, col_reference: str, col_target: str) -> pd.DataFrame:
        """
        指定されたキーに基づいてデータを処理する。

        Parameters
        ----------
            col_reference : str
                参照データのカラム名。
            col_target : str
                ターゲットデータのカラム名。

        Returns
        ----------
            pd.DataFrame
                処理されたデータフレーム。
        """
        df_copied: pd.DataFrame = self._df.copy()
        col_freq: str = self._col_freq

        # データ型の確認と変換
        df_copied[col_freq] = pd.to_numeric(df_copied[col_freq], errors="coerce")
        df_copied[col_reference] = pd.to_numeric(
            df_copied[col_reference], errors="coerce"
        )
        df_copied[col_target] = pd.to_numeric(df_copied[col_target], errors="coerce")

        # NaNを含む行を削除
        df_copied = df_copied.dropna(subset=[col_freq, col_reference, col_target])

        # グループ化と中央値の計算
        grouped = df_copied.groupby(col_freq)
        reference_data = grouped[col_reference].median()
        target_data = grouped[col_target].median()

        df_processed = pd.DataFrame(
            {"reference": reference_data, "target": target_data}
        )

        # 異常な比率を除去
        df_processed.loc[
            (
                (df_processed["target"] / df_processed["reference"] > 1)
                | (df_processed["target"] / df_processed["reference"] < 0)
            )
        ] = np.nan
        df_processed = df_processed.dropna()

        return df_processed

    def _cutoff_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        カットオフ周波数に基づいてDataFrameを加工するメソッド

        Parameters
        ----------
            df : pd.DataFrame
                加工対象のデータフレーム。

        Returns
        ----------
            pd.DataFrame
                カットオフ周波数に基づいて加工されたデータフレーム。
        """
        df_cutoff: pd.DataFrame = df.loc[
            (self._cutoff_freq_low <= df.index) & (df.index <= self._cutoff_freq_high)
        ]
        return df_cutoff

    @classmethod
    def transfer_function(cls, x: np.ndarray, a: float) -> np.ndarray:
        """
        伝達関数を計算する。

        Parameters
        ----------
            x : np.ndarray
                周波数の配列。
            a : float
                伝達関数の係数。

        Returns
        ----------
            np.ndarray
                伝達関数の値。
        """
        return np.exp(-np.log(np.sqrt(2)) * np.power(x / a, 2))

    @staticmethod
    def _load_data(filepath: str | Path) -> pd.DataFrame:
        """
        CSVファイルからデータを読み込む。

        Parameters
        ----------
            filepath : str | Path
                csvファイルのパス。

        Returns
        ----------
            pd.DataFrame
                読み込まれたデータフレーム。
        """
        tmp = pd.read_csv(filepath, header=None, nrows=1, skiprows=0)
        header = tmp.loc[tmp.index[0]]
        df = pd.read_csv(filepath, header=None, skiprows=1)
        df.columns = header
        return df
