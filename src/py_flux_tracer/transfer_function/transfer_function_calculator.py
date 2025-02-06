import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        伝達関数計算のためのクラスを初期化します。

        Parameters
        ----------
            filepath: str | Path
                分析対象のCSVファイルのパス
            col_freq: str
                周波数データが格納されている列名
            cutoff_freq_low: float, optional
                カットオフ周波数の最低値。デフォルトは0.01Hz
            cutoff_freq_high: float, optional
                カットオフ周波数の最高値。デフォルトは1Hz

        Examples
        -------
        >>> calculator = TransferFunctionCalculator(
        ...     filepath="data.csv",
        ...     col_freq="frequency",
        ...     cutoff_freq_low=0.02,
        ...     cutoff_freq_high=0.5
        ... )
        """
        self._col_freq: str = col_freq
        self._cutoff_freq_low: float = cutoff_freq_low
        self._cutoff_freq_high: float = cutoff_freq_high
        self._df: pd.DataFrame = TransferFunctionCalculator._load_data(filepath)

    def calculate_transfer_function(
        self, col_reference: str, col_target: str
    ) -> tuple[float, float, pd.DataFrame]:
        """
        伝達関数の係数を計算します。

        Parameters
        ----------
            col_reference: str
                参照データが格納されている列名を指定します。
            col_target: str
                ターゲットデータが格納されている列名を指定します。

        Returns
        ----------
            tuple[float, float, pandas.DataFrame]
                伝達関数の係数a、係数aの標準誤差、および計算に使用したDataFrameを返します。

        Examples
        -------
        >>> calculator = TransferFunctionCalculator("data.csv", "frequency")
        >>> a, a_err, df = calculator.calculate_transfer_function(
        ...     col_reference="ref_data",
        ...     col_target="target_data"
        ... )
        >>> print(f"伝達関数の係数: {a:.3f} ± {a_err:.3f}")
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

        # 標準誤差を計算(共分散行列の対角成分の平方根)
        perr = np.sqrt(np.diag(pcov))

        # 係数aとその標準誤差、および計算に用いたDataFrameを返す
        return popt[0], perr[0], df_processed

    def create_plot_co_spectra(
        self,
        col1: str,
        col2: str,
        color1: str = "gray",
        color2: str = "red",
        figsize: tuple[float, float] = (10, 6),
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
        window_size: int = 5,
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
        2種類のコスペクトルをプロットします。

        Parameters
        ----------
            col1: str
                1つ目のコスペクトルデータのカラム名を指定します。
            col2: str
                2つ目のコスペクトルデータのカラム名を指定します。
            color1: str, optional
                1つ目のデータの色を指定します。デフォルトは'gray'です。
            color2: str, optional
                2つ目のデータの色を指定します。デフォルトは'red'です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルトは(10, 6)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルトは350です。
            label1: str | None, optional
                1つ目のデータの凡例ラベルを指定します。デフォルトはNoneです。
            label2: str | None, optional
                2つ目のデータの凡例ラベルを指定します。デフォルトはNoneです。
            output_dirpath: str | Path | None, optional
                プロットを保存するディレクトリを指定します。save_fig=Trueの場合は必須です。デフォルトはNoneです。
            output_filename: str, optional
                保存するファイル名を指定します。デフォルトは"co.png"です。
            add_legend: bool, optional
                凡例を追加するかどうかを指定します。デフォルトはTrueです。
            add_xy_labels: bool, optional
                x軸とy軸のラベルを追加するかどうかを指定します。デフォルトはTrueです。
            legend_font_size: float, optional
                凡例のフォントサイズを指定します。デフォルトは16です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルトはTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルトはTrueです。
            subplot_label: str | None, optional
                左上に表示するサブプロットラベルを指定します。デフォルトはNoneです。
            window_size: int, optional
                移動平均の窓サイズを指定します。デフォルトは5です。
            markersize: float, optional
                プロットのマーカーサイズを指定します。デフォルトは14です。
            xlim: tuple[float, float], optional
                x軸の表示範囲を指定します。デフォルトは(0.0001, 10)です。
            ylim: tuple[float, float], optional
                y軸の表示範囲を指定します。デフォルトは(0.0001, 10)です。
            slope_line: tuple[tuple[float, float], tuple[float, float]], optional
                傾きを示す直線の始点と終点の座標を指定します。デフォルトは((0.01, 10), (10, 0.001))です。
            slope_text: tuple[str, tuple[float, float]], optional
                傾きを示すテキストとその位置を指定します。デフォルトは("-4/3", (0.25, 0.4))です。
            subplot_label_pos: tuple[float, float], optional
                サブプロットラベルの位置を指定します。デフォルトは(0.00015, 3)です。

        Returns
        ----------
            None
                戻り値はありません。

        Examples
        -------
        >>> calculator = TransferFunctionCalculator("data.csv", "frequency")
        >>> calculator.create_plot_co_spectra(
        ...     col1="co_spectra1",
        ...     col2="co_spectra2",
        ...     label1="データ1",
        ...     label2="データ2",
        ...     output_dirpath="output"
        ... )
        """
        df_internal: pd.DataFrame = self._df.copy()
        # データの取得と移動平均の適用
        data1 = df_internal[df_internal[col1] > 0].groupby(self._col_freq)[col1].median()
        data2 = df_internal[df_internal[col2] > 0].groupby(self._col_freq)[col2].median()

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
        output_dirpath: str | Path | None = None,
        output_filename: str = "ratio.png",
        figsize: tuple[float, float] = (10, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        ターゲットと参照の比率をプロットします。

        Parameters
        ----------
            df_processed: pd.DataFrame
                処理されたデータフレーム
            reference_name: str
                参照の名前
            target_name: str
                ターゲットの名前
            output_dirpath: str | Path | None, optional
                プロットを保存するディレクトリパス。デフォルト値はNoneで、save_fig=Trueの場合は必須
            output_filename: str, optional
                保存するファイル名。デフォルト値は"ratio.png"
            figsize: tuple[float, float], optional
                プロットのサイズ。デフォルト値は(10, 6)
            dpi: float | None, optional
                プロットの解像度。デフォルト値は350
            save_fig: bool, optional
                プロットを保存するかどうか。デフォルト値はTrue
            show_fig: bool, optional
                プロットを表示するかどうか。デフォルト値はTrue

        Returns
        -------
            None
                戻り値はありません

        Examples
        -------
        >>> df = pd.DataFrame({"target": [1, 2, 3], "reference": [1, 1, 1]})
        >>> calculator = TransferFunctionCalculator()
        >>> calculator.create_plot_ratio(
        ...     df_processed=df,
        ...     reference_name="参照データ",
        ...     target_name="ターゲットデータ",
        ...     output_dirpath="output",
        ...     show_fig=False
        ... )
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
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
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
            filepath: str
                伝達関数の係数が格納されたCSVファイルのパスを指定します。
            config: TfCurvesFromCsvConfig
                プロット設定を指定します。
            csv_encoding: str | None, optional
                CSVファイルのエンコーディングを指定します。デフォルト値は"utf-8-sig"です。
            output_dirpath: str | Path | None, optional
                出力ディレクトリを指定します。save_fig=Trueの場合は必須です。デフォルト値はNoneです。
            output_filename: str, optional
                出力ファイル名を指定します。デフォルト値は"all_tf_curves.png"です。
            col_datetime: str, optional
                日付情報が格納されているカラム名を指定します。デフォルト値は"Date"です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルト値は(10, 6)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルト値は350です。
            add_legend: bool, optional
                凡例を追加するかどうかを指定します。デフォルト値はTrueです。
            add_xlabel: bool, optional
                x軸ラベルを追加するかどうかを指定します。デフォルト値はTrueです。
            label_x: str, optional
                x軸のラベルを指定します。デフォルト値は"f (Hz)"です。
            label_y: str, optional
                y軸のラベルを指定します。デフォルト値は"無次元コスペクトル比"です。
            label_avg: str, optional
                平均値のラベルを指定します。デフォルト値は"Avg."です。
            label_co_ref: str, optional
                参照ガスのラベルを指定します。デフォルト値は"Tv"です。
            legend_font_size: float, optional
                凡例のフォントサイズを指定します。デフォルト値は16です。
            line_colors: list[str] | None, optional
                各日付のデータに使用する色のリストを指定します。デフォルト値はNoneです。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルト値はTrueです。

        Returns
        -------
            None
                戻り値はありません。

        Examples
        -------
        >>> config = TfCurvesFromCsvConfig(col_coef_a="a", base_color="red")
        >>> calculator = TransferFunctionCalculator()
        >>> calculator.create_plot_tf_curves_from_csv(
        ...     filepath="transfer_functions.csv",
        ...     config=config,
        ...     output_dirpath="output",
        ...     show_fig=False
        ... )
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
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
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
        figsize: tuple[float, float] = (10, 6),
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
        伝達関数とそのフィットをプロットします。

        Parameters
        ----------
            a: float
                伝達関数の係数です。
            df_processed: pd.DataFrame
                処理されたデータフレームです。
            reference_name: str
                参照データの名前です。
            target_name: str
                ターゲットデータの名前です。
            figsize: tuple[float, float], optional
                プロットのサイズを指定します。デフォルトは(10, 6)です。
            dpi: float | None, optional
                プロットの解像度を指定します。デフォルトは350です。
            output_dirpath: str | Path | None, optional
                プロットを保存するディレクトリを指定します。save_fig=Trueの場合は必須です。
            output_filename: str, optional
                保存するファイル名を指定します。デフォルトは"tf.png"です。
            save_fig: bool, optional
                プロットを保存するかどうかを指定します。デフォルトはTrueです。
            show_fig: bool, optional
                プロットを表示するかどうかを指定します。デフォルトはTrueです。
            add_xlabel: bool, optional
                x軸のラベルを追加するかどうかを指定します。デフォルトはTrueです。
            label_x: str, optional
                x軸のラベル名を指定します。デフォルトは"f (Hz)"です。
            label_y: str, optional
                y軸のラベル名を指定します。デフォルトは"コスペクトル比"です。
            label_target: str | None, optional
                比較先のラベル名を指定します。デフォルトはNoneです。
            label_ref: str, optional
                比較元のラベル名を指定します。デフォルトは"Tv"です。

        Returns
        -------
            None
                戻り値はありません。

        Examples
        -------
        >>> # 伝達関数のプロットを作成し、保存する
        >>> calculator = TransferFunctionCalculator()
        >>> calculator.create_plot_transfer_function(
        ...     a=0.5,
        ...     df_processed=processed_df,
        ...     reference_name="温度",
        ...     target_name="CO2",
        ...     output_dirpath="./output",
        ...     output_filename="transfer_function.png"
        ... )
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
                    "save_fig = True のとき、 output_dirpath に有効なディレクトリパスを指定する必要があります。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            # プロットをPNG形式で保存
            fig.savefig(os.path.join(output_dirpath, output_filename), dpi=dpi)
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def process_data(self, col_reference: str, col_target: str) -> pd.DataFrame:
        """
        指定されたキーに基づいてデータを処理します。
        周波数ごとにデータをグループ化し、中央値を計算します。
        また、異常な比率のデータを除去します。

        Parameters
        ----------
            col_reference: str
                参照データのカラム名を指定します。
            col_target: str
                ターゲットデータのカラム名を指定します。

        Returns
        ----------
            pd.DataFrame
                処理されたデータフレーム。
                indexは周波数、columnsは'reference'と'target'です。

        Examples
        -------
        >>> calculator = TransferFunctionCalculator()
        >>> processed_df = calculator.process_data(
        ...     col_reference="温度",
        ...     col_target="CO2"
        ... )
        """
        df_internal: pd.DataFrame = self._df.copy()
        col_freq: str = self._col_freq

        # データ型の確認と変換
        df_internal[col_freq] = pd.to_numeric(df_internal[col_freq], errors="coerce")
        df_internal[col_reference] = pd.to_numeric(
            df_internal[col_reference], errors="coerce"
        )
        df_internal[col_target] = pd.to_numeric(df_internal[col_target], errors="coerce")

        # NaNを含む行を削除
        df_internal = df_internal.dropna(subset=[col_freq, col_reference, col_target])

        # グループ化と中央値の計算
        grouped = df_internal.groupby(col_freq)
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
            df: pd.DataFrame
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
            x: np.ndarray
                周波数の配列。
            a: float
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
            filepath: str | Path
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
