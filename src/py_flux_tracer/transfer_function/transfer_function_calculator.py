import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class TransferFunctionCalculator:
    """
    このクラスは、CSVファイルからデータを読み込み、処理し、
    伝達関数を計算してプロットするための機能を提供します。

    この実装は Moore (1986) の論文に基づいています。
    """

    def __init__(
        self,
        file_path: str,
        col_freq: str,
        cutoff_freq_low: float = 0.01,
        cutoff_freq_high: float = 1,
    ):
        """
        TransferFunctionCalculatorクラスのコンストラクタ。

        Parameters:
        ------
            file_path : str
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
        self._df: pd.DataFrame = TransferFunctionCalculator._load_data(file_path)

    def calculate_transfer_function(
        self, col_reference: str, col_target: str
    ) -> tuple[float, float, pd.DataFrame]:
        """
        伝達関数の係数を計算する。

        Parameters:
        ------
            col_reference : str
                参照データのカラム名。
            col_target : str
                ターゲットデータのカラム名。

        Returns:
        ------
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
        figsize: tuple[int, int] = (10, 8),
        label1: str | None = None,
        label2: str | None = None,
        output_dir: str | None = None,
        output_basename: str = "co",
        add_legend: bool = True,
        add_xy_labels: bool = True,
        show_fig: bool = True,
        subplot_label: str | None = "(a)",
        window_size: int = 5,  # 移動平均の窓サイズ
        markersize: float = 14,
    ) -> None:
        """
        2種類のコスペクトルをプロットする。

        Parameters:
        ------
            col1 : str
                1つ目のコスペクトルデータのカラム名。
            col2 : str
                2つ目のコスペクトルデータのカラム名。
            color1 : str, optional
                1つ目のデータの色。デフォルトは'gray'。
            color2 : str, optional
                2つ目のデータの色。デフォルトは'red'。
            figsize : tuple[int, int], optional
                プロットのサイズ。デフォルトは(10, 8)。
            label1 : str, optional
                1つ目のデータのラベル名。デフォルトはNone。
            label2 : str, optional
                2つ目のデータのラベル名。デフォルトはNone。
            output_dir : str | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_basename : str, optional
                保存するファイル名のベース。デフォルトは"co"。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
            subplot_label : str | None, optional
                左上に表示するサブプロットラベル。デフォルトは"(a)"。
            window_size : int, optional
                移動平均の窓サイズ。デフォルトは5。
        """
        df: pd.DataFrame = self._df.copy()
        # データの取得と移動平均の適用
        data1 = df[df[col1] > 0].groupby(self._col_freq)[col1].median()
        data2 = df[df[col2] > 0].groupby(self._col_freq)[col2].median()

        data1 = data1.rolling(window=window_size, center=True, min_periods=1).mean()
        data2 = data2.rolling(window=window_size, center=True, min_periods=1).mean()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        # マーカーサイズを設定して見やすくする
        ax.plot(
            data1.index, data1, "o", color=color1, label=label1, markersize=markersize
        )
        ax.plot(
            data2.index, data2, "o", color=color2, label=label2, markersize=markersize
        )
        ax.plot([0.01, 10], [10, 0.001], "-", color="black")
        ax.text(0.25, 0.4, "-4/3")

        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.0001, 10)
        ax.set_ylim(0.0001, 10)
        if add_xy_labels:
            ax.set_xlabel("f (Hz)")
            ax.set_ylabel("無次元コスペクトル")

        if add_legend:
            ax.legend(
                bbox_to_anchor=(0.05, 1),
                loc="lower left",
                fontsize=16,
                ncol=3,
                frameon=False,
            )
        if subplot_label is not None:
            ax.text(0.00015, 3, subplot_label)
        fig.tight_layout()

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            # プロットをPNG形式で保存
            filename: str = f"{output_basename}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def create_plot_ratio(
        self,
        df_processed: pd.DataFrame,
        reference_name: str,
        target_name: str,
        figsize: tuple[int, int] = (10, 6),
        output_dir: str | None = None,
        output_basename: str = "ratio",
        show_fig: bool = True,
    ) -> None:
        """
        ターゲットと参照の比率をプロットする。

        Parameters:
        ------
            df_processed : pd.DataFrame
                処理されたデータフレーム。
            reference_name : str
                参照の名前。
            target_name : str
                ターゲットの名前。
            figsize : tuple[int, int], optional
                プロットのサイズ。デフォルトは(10, 6)。
            output_dir : str | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_basename : str, optional
                保存するファイル名のベース。デフォルトは"ratio"。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.plot(
            df_processed.index, df_processed["target"] / df_processed["reference"], "o"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("f (Hz)")
        ax.set_ylabel(f"{target_name} / {reference_name}")
        ax.set_title(f"{target_name}と{reference_name}の比")

        if output_dir is not None:
            # プロットをPNG形式で保存
            filename: str = f"{output_basename}-{reference_name}_{target_name}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    @classmethod
    def create_plot_tf_curves_from_csv(
        cls,
        file_path: str,
        gas_configs: list[tuple[str, str, str, str]],
        output_dir: str | None = None,
        output_basename: str = "all_tf_curves",
        col_datetime: str = "Date",
        add_xlabel: bool = True,
        label_x: str = "f (Hz)",
        label_y: str = "無次元コスペクトル比",
        label_avg: str = "Avg.",
        label_co_ref: str = "Tv",
        line_colors: list[str] | None = None,
        font_family: list[str] = ["Arial", "MS Gothic"],
        font_size: float = 20,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        複数の伝達関数の係数をプロットし、各ガスの平均値を表示します。
        各ガスのデータをCSVファイルから読み込み、指定された設定に基づいてプロットを生成します。
        プロットはオプションで保存することも可能です。

        Parameters:
        ------
            file_path : str
                伝達関数の係数が格納されたCSVファイルのパス。
            gas_configs : list[tuple[str, str, str, str]]
                ガスごとの設定のリスト。各タプルは以下の要素を含む:
                (係数のカラム名, ガスの表示ラベル, 平均線の色, 出力ファイル用のガス名)
                例: [("a_ch4-used", "CH$_4$", "red", "ch4")]
            output_dir : str | None, optional
                出力ディレクトリ。Noneの場合は保存しない。
            output_basename : str, optional
                出力ファイル名のベース。デフォルトは"all_tf_curves"。
            col_datetime : str, optional
                日付情報が格納されているカラム名。デフォルトは"Date"。
            add_xlabel : bool, optional
                x軸ラベルを追加するかどうか。デフォルトはTrue。
            label_x : str, optional
                x軸のラベル。デフォルトは"f (Hz)"。
            label_y : str, optional
                y軸のラベル。デフォルトは"無次元コスペクトル比"。
            label_avg : str, optional
                平均値のラベル。デフォルトは"Avg."。
            line_colors : list[str] | None, optional
                各日付のデータに使用する色のリスト。
            font_family : list[str], optional
                使用するフォントファミリーのリスト。
            font_size : float, optional
                フォントサイズ。
            save_fig : bool, optional
                プロットを保存するかどうか。デフォルトはTrue。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
        """
        # プロットパラメータの設定
        plt.rcParams.update(
            {
                "font.family": font_family,
                "font.size": font_size,
                "axes.labelsize": font_size,
                "axes.titlesize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
                "legend.fontsize": font_size,
            }
        )

        # CSVファイルを読み込む
        df = pd.read_csv(file_path)

        # 各ガスについてプロット
        for col_coef_a, label_gas, base_color, gas_name in gas_configs:
            fig = plt.figure(figsize=(10, 6))

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
                a = row[1][col_coef_a]
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
            a_mean = df[col_coef_a].mean()
            x_fit = np.logspace(-3, 1, 1000)
            y_fit = cls.transfer_function(x_fit, a_mean)
            plt.plot(
                x_fit,
                y_fit,
                "-",
                color=base_color,
                linewidth=3,
                label=f"{label_avg} (a = {a_mean:.3f})",
            )

            # グラフの設定
            label_y_formatted: str = f"{label_y}\n({label_gas} / {label_co_ref})"
            plt.xscale("log")
            if add_xlabel:
                plt.xlabel(label_x)
            plt.ylabel(label_y_formatted)
            plt.legend(loc="lower left", fontsize=font_size - 6)
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()

            if save_fig:
                if output_dir is None:
                    raise ValueError(
                        "save_fig=Trueのとき、output_dirに有効なディレクトリパスを指定する必要があります。"
                    )
                os.makedirs(output_dir, exist_ok=True)
                output_path: str = os.path.join(
                    output_dir, f"{output_basename}-{gas_name}.png"
                )
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
            if show_fig:
                plt.show()
            else:
                plt.close(fig=fig)

    def create_plot_transfer_function(
        self,
        a: float,
        df_processed: pd.DataFrame,
        reference_name: str,
        target_name: str,
        figsize: tuple[int, int] = (10, 6),
        output_dir: str | None = None,
        output_basename: str = "tf",
        show_fig: bool = True,
        add_xlabel: bool = True,
        label_x: str = "f (Hz)",
        label_y: str = "コスペクトル比",
        label_gas: str | None = None,
    ) -> None:
        """
        伝達関数とそのフィットをプロットする。

        Parameters:
        ------
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
            output_dir : str | None, optional
                プロットを保存するディレクトリ。デフォルトはNoneで、保存しない。
            output_basename : str, optional
                保存するファイル名のベース。デフォルトは"tf"。
            show_fig : bool, optional
                プロットを表示するかどうか。デフォルトはTrue。
        """
        df_cutoff: pd.DataFrame = self._cutoff_df(df_processed)

        fig = plt.figure(figsize=figsize)
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
        label_y_formatted: str = f"{label_y}\n({label_gas} / 顕熱)"
        plt.xscale("log")
        if add_xlabel:
            plt.xlabel(label_x)
        plt.ylabel(label_y_formatted)
        ax.legend()

        if output_dir is not None:
            # プロットをPNG形式で保存
            filename: str = f"{output_basename}-{reference_name}_{target_name}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=300)
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def process_data(self, col_reference: str, col_target: str) -> pd.DataFrame:
        """
        指定されたキーに基づいてデータを処理する。

        Parameters:
        ------
            col_reference : str
                参照データのカラム名。
            col_target : str
                ターゲットデータのカラム名。

        Returns:
        ------
            pd.DataFrame
                処理されたデータフレーム。
        """
        df: pd.DataFrame = self._df.copy()
        col_freq: str = self._col_freq

        # データ型の確認と変換
        df[col_freq] = pd.to_numeric(df[col_freq], errors="coerce")
        df[col_reference] = pd.to_numeric(df[col_reference], errors="coerce")
        df[col_target] = pd.to_numeric(df[col_target], errors="coerce")

        # NaNを含む行を削除
        df = df.dropna(subset=[col_freq, col_reference, col_target])

        # グループ化と中央値の計算
        grouped = df.groupby(col_freq)
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

        Parameters:
        ------
            df : pd.DataFrame
                加工対象のデータフレーム。

        Returns:
        ------
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

        Parameters:
        ------
            x : np.ndarray
                周波数の配列。
            a : float
                伝達関数の係数。

        Returns:
        ------
            np.ndarray
                伝達関数の値。
        """
        return np.exp(-np.log(np.sqrt(2)) * np.power(x / a, 2))

    @staticmethod
    def _load_data(file_path: str) -> pd.DataFrame:
        """
        CSVファイルからデータを読み込む。

        Parameters:
        ------
            file_path : str
                csvファイルのパス。

        Returns:
        ------
            pd.DataFrame
                読み込まれたデータフレーム。
        """
        tmp = pd.read_csv(file_path, header=None, nrows=1, skiprows=0)
        header = tmp.loc[tmp.index[0]]
        df = pd.read_csv(file_path, header=None, skiprows=1)
        df.columns = header
        return df

    @staticmethod
    def setup_plot_params(
        font_family: list[str] = ["Arial", "Dejavu Sans"],
        font_size: float = 20,
        legend_size: float = 20,
        tick_size: float = 20,
        title_size: float = 20,
        plot_params=None,
    ) -> None:
        """
        matplotlibのプロットパラメータを設定します。

        Parameters:
        ------
            font_family : list[str]
                使用するフォントファミリーのリスト。
            font_size : float
                軸ラベルのフォントサイズ。
            legend_size : float
                凡例のフォントサイズ。
            tick_size : float
                軸目盛りのフォントサイズ。
            title_size : float
                タイトルのフォントサイズ。
            plot_params : Optional[Dict[str, any]]
                matplotlibのプロットパラメータの辞書。
        """
        # デフォルトのプロットパラメータ
        default_params = {
            "axes.linewidth": 1.0,
            "axes.titlesize": title_size,  # タイトル
            "grid.color": "gray",
            "grid.linewidth": 1.0,
            "font.family": font_family,
            "font.size": font_size,  # 軸ラベル
            "legend.fontsize": legend_size,  # 凡例
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.labelsize": tick_size,  # 軸目盛
            "ytick.labelsize": tick_size,  # 軸目盛
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "ytick.direction": "out",
            "ytick.major.width": 1.0,
        }

        # plot_paramsが定義されている場合、デフォルトに追記
        if plot_params:
            default_params.update(plot_params)

        plt.rcParams.update(default_params)  # プロットパラメータを更新
