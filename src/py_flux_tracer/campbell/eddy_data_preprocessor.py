import os
import re
from datetime import datetime
from logging import DEBUG, INFO, Logger
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from ..commons.utilities import setup_logger

MeasuredWindKeyType = Literal["u_m", "v_m", "w_m"]


class EddyDataPreprocessor:
    # カラム名を定数として定義
    WIND_U = "edp_wind_u"
    WIND_V = "edp_wind_v"
    WIND_W = "edp_wind_w"
    RAD_WIND_DIR = "edp_rad_wind_dir"
    RAD_WIND_INC = "edp_rad_wind_inc"
    DEGREE_WIND_DIR = "edp_degree_wind_dir"
    DEGREE_WIND_INC = "edp_degree_wind_inc"

    def __init__(
        self,
        fs: float = 10,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        渦相関法によって記録されたデータファイルを処理するクラス。

        Parameters
        ----------
            fs: float
                サンプリング周波数。
            logger: Logger | None, optional
                使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug: bool, optional
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.fs: float = fs

        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

    def add_uvw_columns(
        self,
        df: pd.DataFrame,
        column_mapping: dict[MeasuredWindKeyType, str] | None = None,
    ) -> pd.DataFrame:
        """
        DataFrameに水平風速u、v、鉛直風速wの列を追加する関数。
        各成分のキーは`edp_wind_u`、`edp_wind_v`、`edp_wind_w`である。

        Parameters
        ----------
            df: pd.DataFrame
                風速データを含むDataFrame
            column_mapping: dict[MeasuredWindKeyType, str] | None, optional
                入力データのカラム名マッピング。
                キーは"u_m", "v_m", "w_m"で、値は対応する入力データのカラム名。
                Noneの場合は{"u_m": "Ux", "v_m": "Uy", "w_m": "Uz"}を使用する。

        Returns
        ----------
            pd.DataFrame
                水平風速u、v、鉛直風速wの列を追加したDataFrame

        Raises
        ----------
            ValueError
                必要なカラムが存在しない場合、またはマッピングに必要なキーが不足している場合

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'Ux': [1.0, 2.0, 3.0],
        ...     'Uy': [4.0, 5.0, 6.0],
        ...     'Uz': [7.0, 8.0, 9.0]
        ... })
        >>> edp = EddyDataPreprocessor()
        >>> df_with_uvw = edp.add_uvw_columns(df)
        """
        if column_mapping is None:
            column_mapping = {
                "u_m": "Ux",
                "v_m": "Uy",
                "w_m": "Uz",
            }
        required_keys: list[MeasuredWindKeyType] = ["u_m", "v_m", "w_m"]
        # マッピングに必要なキーが存在するか確認
        for key in required_keys:
            if key not in column_mapping:
                raise ValueError(f"column_mapping に必要なキー '{key}' が存在しません。")

        # 必要な列がDataFrameに存在するか確認
        for key, column in column_mapping.items():
            if column not in df.columns:
                raise ValueError(
                    f"必要な列 '{column}' (mapped from '{key}') がDataFrameに存在しません。"
                )

        df_internal: pd.DataFrame = df.copy()
        # pandasの.valuesを使用してnumpy配列を取得し、その型をnp.ndarrayに明示的にキャストする
        wind_x_array: np.ndarray = np.array(df_internal[column_mapping["u_m"]].values)
        wind_y_array: np.ndarray = np.array(df_internal[column_mapping["v_m"]].values)
        wind_z_array: np.ndarray = np.array(df_internal[column_mapping["w_m"]].values)

        # 平均風向を計算
        wind_direction: float = EddyDataPreprocessor._wind_direction(
            wind_x_array, wind_y_array
        )

        # 水平方向に座標回転を行u, v成分を求める
        wind_u_array, wind_v_array = EddyDataPreprocessor._horizontal_wind_speed(
            wind_x_array, wind_y_array, wind_direction
        )
        wind_w_array: np.ndarray = wind_z_array  # wはz成分そのまま

        # u, wから風の迎角を計算
        wind_inclination: float = EddyDataPreprocessor._wind_inclination(
            wind_u_array, wind_w_array
        )

        # 2回座標回転を行い、u, wを求める
        wind_u_array_rotated, wind_w_array_rotated = (
            EddyDataPreprocessor._vertical_rotation(
                wind_u_array, wind_w_array, wind_inclination
            )
        )

        df_internal[self.WIND_U] = wind_u_array_rotated
        df_internal[self.WIND_V] = wind_v_array
        df_internal[self.WIND_W] = wind_w_array_rotated
        df_internal[self.RAD_WIND_DIR] = wind_direction
        df_internal[self.RAD_WIND_INC] = wind_inclination
        df_internal[self.DEGREE_WIND_DIR] = np.degrees(wind_direction)
        df_internal[self.DEGREE_WIND_INC] = np.degrees(wind_inclination)

        return df_internal

    def analyze_lag_times(
        self,
        input_dirpath: str | Path,
        input_files_pattern: str = r"Eddy_(\d+)",
        input_files_suffix: str = ".dat",
        col1: str = "edp_wind_w",
        col2_list: list[str] | None = None,
        median_range: float = 20,
        output_dirpath: str | Path | None = None,
        output_tag: str = "",
        add_title: bool = True,
        figsize: tuple[float, float] = (10, 8),
        dpi: float | None = 350,
        plot_range_tuple: tuple = (-50, 200),
        print_results: bool = True,
        xlabel: str | None = "Seconds",
        ylabel: str | None = "Frequency",
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        resample_in_processing: bool = False,
        interpolate: bool = True,
        numeric_columns: list[str] | None = None,
        metadata_rows: int = 4,
        skiprows: list[int] | None = None,
        add_uvw_columns: bool = True,
        uvw_column_mapping: dict[MeasuredWindKeyType, str] | None = None,
    ) -> dict[str, float]:
        """遅れ時間(ラグ)の統計分析を行い、指定されたディレクトリ内のデータファイルを処理します。
        解析結果とメタデータはCSVファイルとして出力されます。

        Parameters
        ----------
            input_dirpath: str | Path
                入力データファイルが格納されているディレクトリのパス
            input_files_pattern: str, optional
                入力ファイル名のパターン(正規表現)。デフォルトは"Eddy_(\\d+)"
            input_files_suffix: str, optional
                入力ファイルの拡張子。デフォルトは".dat"
            col1: str, optional
                基準変数の列名。デフォルトは"edp_wind_w"
            col2_list: list[str] | None, optional
                比較変数の列名のリスト。デフォルトはNoneで、その場合は["Tv"]を使用
            median_range: float, optional
                中央値を中心とした範囲。デフォルトは20
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパス。デフォルトはNoneで、その場合は結果を保存しない
            output_tag: str, optional
                出力ファイルに付与するタグ。デフォルトは空文字
            add_title: bool, optional
                プロットにタイトルを追加するかどうか。デフォルトはTrue
            figsize: tuple[float, float], optional
                プロットのサイズ。デフォルトは(10, 8)
            dpi: float | None, optional
                プロットのdpi。デフォルトは350
            plot_range_tuple: tuple, optional
                ヒストグラムの表示範囲。デフォルトは(-50, 200)
            print_results: bool, optional
                結果をコンソールに表示するかどうか。デフォルトはTrue
            xlabel: str | None, optional
                x軸のラベル。デフォルトは"Seconds"
            ylabel: str | None, optional
                y軸のラベル。デフォルトは"Frequency"
            index_column: str, optional
                タイムスタンプが格納された列名。デフォルトは"TIMESTAMP"
            index_format: str, optional
                タイムスタンプのフォーマット。デフォルトは"%Y-%m-%d %H:%M:%S.%f"
            resample_in_processing: bool, optional
                データを遅れ時間の計算中にリサンプリングするかどうか。デフォルトはFalse
            interpolate: bool, optional
                欠損値の補完を適用するかどうか。デフォルトはTrue
            numeric_columns: list[str] | None, optional
                数値型に変換する列名のリスト。デフォルトはNoneで、その場合は標準の列名リストを使用
            metadata_rows: int, optional
                メタデータの行数。デフォルトは4
            skiprows: list[int] | None, optional
                スキップする行番号のリスト。デフォルトはNoneで、その場合は[0, 2, 3]を使用
            add_uvw_columns: bool, optional
                u, v, wの列を追加するかどうか。デフォルトはTrue
            uvw_column_mapping: dict[MeasuredWindKeyType, str] | None, optional
                u, v, wの列名をマッピングする辞書。デフォルトはNoneで、その場合は標準のマッピングを使用

        Returns
        -------
            dict[str, float]
                各変数の遅れ時間(平均値を採用)を含む辞書

        Examples
        --------
        >>> edp = EddyDataPreprocessor(fs=10)
        >>> result = edp.analyze_lag_times(
        ...     input_dirpath="data/eddy",
        ...     col2_list=["Ultra_CH4_ppm", "Ultra_C2H6_ppb"],
        ...     output_dirpath="outputs"
        ... )
        >>> print(result)
        {'Ultra_CH4_ppm': 0.2, 'Ultra_C2H6_ppb': 0.3}
        """
        if output_dirpath is None:
            self.logger.warn(
                "output_dirpathが指定されていません。解析結果を保存する場合は、有効なディレクトリを指定してください。"
            )
        all_lags_indices: list[list[int]] = []
        results: dict[str, float] = {}

        # デフォルト値の設定
        if col2_list is None:
            col2_list = ["Tv"]
        if numeric_columns is None:
            numeric_columns = [
                "Ux",
                "Uy",
                "Uz",
                "Tv",
                "diag_sonic",
                "CO2_new",
                "H2O",
                "diag_irga",
                "cell_tmpr",
                "cell_press",
                "Ultra_CH4_ppm",
                "Ultra_C2H6_ppb",
                "Ultra_H2O_ppm",
                "Ultra_CH4_ppm_C",
                "Ultra_C2H6_ppb_C",
            ]
        if skiprows is None:
            skiprows = [0, 2, 3]
        if uvw_column_mapping is None:
            uvw_column_mapping = {
                "u_m": "Ux",
                "v_m": "Uy",
                "w_m": "Uz",
            }

        # メイン処理
        # ファイル名に含まれる数字に基づいてソート
        csv_files = EddyDataPreprocessor._get_sorted_files(
            directory=input_dirpath,
            pattern=input_files_pattern,
            suffix=input_files_suffix,
        )
        if not csv_files:
            raise FileNotFoundError(
                f"There is no '{input_files_suffix}' file to process; input_dirpath: '{input_dirpath}', input_files_suffix: '{input_files_suffix}'"
            )

        for file in tqdm(csv_files, desc="Calculating"):
            path: str = os.path.join(input_dirpath, file)
            df: pd.DataFrame = pd.DataFrame()  # 空のDataFrameで初期化
            if resample_in_processing:
                df, _ = self.get_resampled_df(
                    filepath=path,
                    metadata_rows=metadata_rows,
                    skiprows=skiprows,
                    index_column=index_column,
                    index_format=index_format,
                    numeric_columns=numeric_columns,
                    interpolate=interpolate,
                    resample=resample_in_processing,
                )
            else:
                df = pd.read_csv(path, skiprows=skiprows)
            # 仰角などを補正した風速成分を追加
            if add_uvw_columns:
                df = self.add_uvw_columns(df=df, column_mapping=uvw_column_mapping)
            lags_list = EddyDataPreprocessor.calculate_lag_time(
                df=df,
                col1=col1,
                col2_list=col2_list,
            )
            all_lags_indices.append(lags_list)
        self.logger.info("すべてのCSVファイルにおける遅れ時間が計算されました。")

        # Convert all_lags_indices to a DataFrame
        lags_indices_df: pd.DataFrame = pd.DataFrame(
            all_lags_indices, columns=col2_list
        )

        # フォーマット用のキーの最大の長さ
        max_col_name_length: int = max(
            len(column) for column in lags_indices_df.columns
        )

        if print_results:
            self.logger.info(f"カラム`{col1}`に対する遅れ時間を表示します。")

        # 結果を格納するためのリスト
        output_data = []

        for column in lags_indices_df.columns:
            data: pd.Series = lags_indices_df[column]

            # ヒストグラムの作成
            fig = plt.figure(figsize=figsize, dpi=dpi)
            plt.hist(data, bins=20, range=plot_range_tuple)
            if add_title:
                plt.title(f"Delays of {column}")
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.xlim(plot_range_tuple)

            # ファイルとして保存するか
            if output_dirpath is not None:
                os.makedirs(output_dirpath, exist_ok=True)
                filename: str = f"lags_histogram-{column}{output_tag}.png"
                filepath: str = os.path.join(output_dirpath, filename)
                plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
                plt.close(fig=fig)

            # 中央値を計算し、その周辺のデータのみを使用
            median_value = np.median(data)
            filtered_data: pd.Series = data[
                (data >= median_value - median_range)
                & (data <= median_value + median_range)
            ]

            # 平均値を計算
            mean_value = np.mean(filtered_data)
            mean_seconds: float = float(mean_value / self.fs)  # 統計値を秒に変換
            results[column] = mean_seconds

            # 結果とメタデータを出力データに追加
            output_data.append(
                {
                    "col1": col1,
                    "col2": column,
                    "col2_lag": round(mean_seconds, 2),  # 数値として小数点2桁を保持
                    "lag_unit": "s",
                    "median_range": median_range,
                }
            )

            if print_results:
                print(f"{column:<{max_col_name_length}}: {mean_seconds:.2f} s")

        # 結果をCSVファイルとして出力
        if output_dirpath is not None:
            output_df: pd.DataFrame = pd.DataFrame(output_data)
            csv_filepath: str = os.path.join(
                output_dirpath, f"lags_results{output_tag}.csv"
            )
            output_df.to_csv(csv_filepath, index=False, encoding="utf-8")
            self.logger.info(f"解析結果をCSVファイルに保存しました: {csv_filepath}")

        return results

    def get_generated_columns_names(self, print_summary: bool = True) -> list[str]:
        """
        クラス内部で生成されるカラム名を取得する。

        Parameters
        ----------
            print_summary: bool, optional
                結果をprint()で表示するかどうかを指定するフラグ。デフォルトはTrueで表示する。

        Returns
        ----------
            list[str]
                生成されるカラム名のリスト。以下のカラムが含まれる:
                - edp_wind_u: 水平風速u成分
                - edp_wind_v: 水平風速v成分 
                - edp_wind_w: 鉛直風速w成分
                - edp_rad_wind_dir: 風向(ラジアン)
                - edp_rad_wind_inc: 迎角(ラジアン)
                - edp_degree_wind_dir: 風向(度)
                - edp_degree_wind_inc: 迎角(度)

        Examples
        --------
        >>> edp = EddyDataPreprocessor()
        >>> cols = edp.get_generated_columns_names(print_summary=False)
        >>> print(cols)
        ['edp_wind_u', 'edp_wind_v', 'edp_wind_w', 'edp_rad_wind_dir', 
         'edp_rad_wind_inc', 'edp_degree_wind_dir', 'edp_degree_wind_inc']
        """
        list_cols: list[str] = [
            self.WIND_U,
            self.WIND_V,
            self.WIND_W,
            self.RAD_WIND_DIR,
            self.RAD_WIND_INC,
            self.DEGREE_WIND_DIR,
            self.DEGREE_WIND_INC,
        ]
        if print_summary:
            print(list_cols)
        return list_cols

    def get_resampled_df(
        self,
        filepath: str,
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        numeric_columns: list[str] | None = None,
        metadata_rows: int = 4,
        skiprows: list[int] | None = None,
        resample: bool = True,
        interpolate: bool = True,
    ) -> tuple[pd.DataFrame, list[str]]:
        """CSVファイルを読み込み、前処理を行う

        前処理の手順は以下の通りです:
        1. 不要な行を削除する。デフォルトの場合は、2行目をヘッダーとして残し、1、3、4行目が削除される。
        2. 数値データを float 型に変換する
        3. TIMESTAMP列をDateTimeインデックスに設定する
        4. エラー値をNaNに置き換える
        5. 指定されたサンプリングレートでリサンプリングする
        6. 欠損値(NaN)を前後の値から線形補間する
        7. DateTimeインデックスを削除する

        Parameters
        ----------
            filepath: str
                読み込むCSVファイルのパス
            index_column: str, optional
                インデックスに使用する列名。デフォルトは"TIMESTAMP"。
            index_format: str, optional
                インデックスの日付形式。デフォルトは"%Y-%m-%d %H:%M:%S.%f"。
            numeric_columns: list[str] | None, optional
                数値型に変換する列名のリスト。デフォルトはNoneで、その場合は標準の列名リストを使用。
            metadata_rows: int, optional
                メタデータとして読み込む行数。デフォルトは4。
            skiprows: list[int] | None, optional
                スキップする行インデックスのリスト。デフォルトはNoneで、その場合は[0, 2, 3]を使用。
            resample: bool, optional
                メソッド内でリサンプリング&欠損補間をするかのフラグ。デフォルトはTrue。
            interpolate: bool, optional
                欠損値の補完を適用するかのフラグ。デフォルトはTrue。

        Returns
        ----------
            tuple[pd.DataFrame, list[str]]
                前処理済みのデータフレームとメタデータのリストを含むタプル。

        Examples
        --------
        >>> edp = EddyDataPreprocessor()
        >>> df, metadata = edp.get_resampled_df(
        ...     filepath="data/eddy_data.csv",
        ...     numeric_columns=["Ux", "Uy", "Uz", "Tv"]
        ... )
        """
        # デフォルト値の設定
        if numeric_columns is None:
            numeric_columns = [
                "Ux",
                "Uy",
                "Uz",
                "Tv",
                "diag_sonic",
                "CO2_new",
                "H2O",
                "diag_irga",
                "cell_tmpr",
                "cell_press",
                "Ultra_CH4_ppm",
                "Ultra_C2H6_ppb",
                "Ultra_H2O_ppm",
                "Ultra_CH4_ppm_C",
                "Ultra_C2H6_ppb_C",
            ]
        if skiprows is None:
            skiprows = [0, 2, 3]

        # メタデータを読み込む
        metadata: list[str] = []
        with open(filepath) as f:
            for _ in range(metadata_rows):
                line = f.readline().strip()
                metadata.append(line.replace('"', ""))

        # CSVファイルを読み込む
        df: pd.DataFrame = pd.read_csv(filepath, skiprows=skiprows)

        # 数値データをfloat型に変換する
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if not resample:
            # μ秒がない場合は".0"を追加する
            df[index_column] = df[index_column].apply(
                lambda x: f"{x}.0" if "." not in x else x
            )
            # TIMESTAMPをDateTimeインデックスに設定する
            df[index_column] = pd.to_datetime(df[index_column], format=index_format)
            df = df.set_index(index_column)

            # リサンプリング前の有効数字を取得
            decimal_places = {}
            for col in numeric_columns:
                if col in df.columns:
                    max_decimals = (
                        df[col].astype(str).str.extract(r"\.(\d+)")[0].str.len().max()
                    )
                    decimal_places[col] = (
                        int(max_decimals) if pd.notna(max_decimals) else 0
                    )

            # リサンプリングを実行
            resampling_period: int = int(1000 / self.fs)
            df_resampled: pd.DataFrame = df.resample(f"{resampling_period}ms").mean(
                numeric_only=True
            )

            if interpolate:
                # 補間を実行
                df_resampled = df_resampled.interpolate()
                # 有効数字を調整
                for col, decimals in decimal_places.items():
                    if col in df_resampled.columns:
                        df_resampled[col] = df_resampled[col].round(decimals)

            # DateTimeインデックスを削除する
            df = df_resampled.reset_index()
            # ミリ秒を1桁にフォーマット
            df[index_column] = (
                df[index_column].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-5]
            )

        return df, metadata

    def output_resampled_data(
        self,
        input_dirpath: str,
        resampled_dirpath: str,
        c2c1_ratio_dirpath: str,
        input_file_pattern: str = r"Eddy_(\d+)",
        input_files_suffix: str = ".dat",
        col_c1: str = "Ultra_CH4_ppm_C",
        col_c2: str = "Ultra_C2H6_ppb",
        output_c2c1_ratio: bool = True,
        output_resampled: bool = True,
        c2c1_ratio_csv_prefix: str = "SAC.Ultra",
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        resample: bool = True,
        interpolate: bool = True,
        numeric_columns: list[str] | None = None,
        metadata_rows: int = 4,
        skiprows: list[int] | None = None,
    ) -> None:
        """
        指定されたディレクトリ内のCSVファイルを処理し、リサンプリングと欠損値補間を行います。

        このメソッドは、指定されたディレクトリ内のCSVファイルを読み込み、リサンプリングを行い、
        欠損値を補完します。処理結果として、リサンプリングされたCSVファイルを出力し、
        相関係数やC2H6/CH4比を計算してDataFrameに保存します。
        リサンプリングと欠損値補完は`get_resampled_df`と同様のロジックを使用します。

        Parameters
        ----------
            input_dirpath: str
                入力CSVファイルが格納されているディレクトリのパス
            resampled_dirpath: str
                リサンプリングされたCSVファイルを出力するディレクトリのパス
            c2c1_ratio_dirpath: str
                計算結果を保存するディレクトリのパス
            input_file_pattern: str, optional
                ファイル名からソートキーを抽出する正規表現パターン。デフォルトは"Eddy_(\\d+)"で、最初の数字グループでソートします
            input_files_suffix: str, optional
                入力ファイルの拡張子。デフォルトは".dat"
            col_c1: str, optional
                CH4濃度を含む列名。デフォルトは"Ultra_CH4_ppm_C"
            col_c2: str, optional
                C2H6濃度を含む列名。デフォルトは"Ultra_C2H6_ppb"
            output_c2c1_ratio: bool, optional
                線形回帰を行うかどうか。デフォルトはTrue
            output_resampled: bool, optional
                リサンプリングされたCSVファイルを出力するかどうか。デフォルトはTrue
            c2c1_ratio_csv_prefix: str, optional
                出力ファイルの接頭辞。デフォルトは"SAC.Ultra"で、出力時は"SAC.Ultra.2024.09.21.ratio.csv"のような形式となる
            index_column: str, optional
                日時情報を含む列名。デフォルトは"TIMESTAMP"
            index_format: str, optional
                インデックスの日付形式。デフォルトは"%Y-%m-%d %H:%M:%S.%f"
            resample: bool, optional
                リサンプリングを行うかどうか。デフォルトはTrue
            interpolate: bool, optional
                欠損値補間を行うかどうか。デフォルトはTrue
            numeric_columns: list[str] | None, optional
                数値型に変換する列名のリスト。デフォルトはNoneで、その場合は標準の列名リストを使用
            metadata_rows: int, optional
                メタデータとして読み込む行数。デフォルトは4
            skiprows: list[int] | None, optional
                スキップする行インデックスのリスト。デフォルトはNoneで、その場合は[0, 2, 3]を使用

        Raises
        ----------
            OSError
                ディレクトリの作成に失敗した場合
            FileNotFoundError
                入力ファイルが見つからない場合
            ValueError
                出力ディレクトリが指定されていない、またはデータの処理中にエラーが発生した場合

        Examples
        --------
        >>> edp = EddyDataPreprocessor(fs=10)
        >>> edp.output_resampled_data(
        ...     input_dirpath="data/raw",
        ...     resampled_dirpath="data/resampled",
        ...     c2c1_ratio_dirpath="data/ratio",
        ...     col_c1="Ultra_CH4_ppm_C",
        ...     col_c2="Ultra_C2H6_ppb"
        ... )
        """
        # 出力オプションとディレクトリの検証
        if output_resampled and resampled_dirpath is None:
            raise ValueError(
                "output_resampled が True の場合、resampled_dirpath を指定する必要があります"
            )
        if output_c2c1_ratio and c2c1_ratio_dirpath is None:
            raise ValueError(
                "output_c2c1_ratio が True の場合、c2c1_ratio_dirpath を指定する必要があります"
            )

        # ディレクトリの作成(必要な場合のみ)
        if output_resampled:
            os.makedirs(resampled_dirpath, exist_ok=True)
        if output_c2c1_ratio:
            os.makedirs(c2c1_ratio_dirpath, exist_ok=True)

        # デフォルト値の設定
        if numeric_columns is None:
            numeric_columns = [
                "Ux",
                "Uy",
                "Uz",
                "Tv",
                "diag_sonic",
                "CO2_new",
                "H2O",
                "diag_irga",
                "cell_tmpr",
                "cell_press",
                "Ultra_CH4_ppm",
                "Ultra_C2H6_ppb",
                "Ultra_H2O_ppm",
                "Ultra_CH4_ppm_C",
                "Ultra_C2H6_ppb_C",
            ]
        if skiprows is None:
            skiprows = [0, 2, 3]

        ratio_data: list[dict[str, str | float]] = []
        latest_date: datetime = datetime.min

        # csvファイル名のリスト
        csv_files: list[str] = EddyDataPreprocessor._get_sorted_files(
            input_dirpath, input_file_pattern, input_files_suffix
        )

        for filename in tqdm(csv_files, desc="Processing files"):
            input_filepath: str = os.path.join(input_dirpath, filename)
            # リサンプリング&欠損値補間
            df, metadata = self.get_resampled_df(
                filepath=input_filepath,
                index_column=index_column,
                index_format=index_format,
                interpolate=interpolate,
                resample=resample,
                numeric_columns=numeric_columns,
                metadata_rows=metadata_rows,
                skiprows=skiprows,
            )

            # 開始時間を取得
            start_time: datetime = pd.to_datetime(df[index_column].iloc[0])
            # 処理したファイルの中で最も最新の日付
            latest_date = max(latest_date, start_time)

            # リサンプリング&欠損値補間したCSVを出力
            if output_resampled:
                base_filename: str = re.sub(rf"\{input_files_suffix}$", "", filename)
                output_csv_path: str = os.path.join(
                    resampled_dirpath, f"{base_filename}-resampled.csv"
                )
                # メタデータを先に書き込む
                with open(output_csv_path, "w") as f:
                    for line in metadata:
                        f.write(f"{line}\n")
                # データフレームを追記モードで書き込む
                df.to_csv(
                    output_csv_path, index=False, mode="a", quoting=3, header=False
                )

            # 相関係数とC2H6/CH4比を計算
            if output_c2c1_ratio:
                ch4_data: pd.Series = df[col_c1]
                c2h6_data: pd.Series = df[col_c2]

                ratio_row: dict[str, str | float] = {
                    "Date": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "slope": f"{np.nan}",
                    "intercept": f"{np.nan}",
                    "r_value": f"{np.nan}",
                    "p_value": f"{np.nan}",
                    "stderr": f"{np.nan}",
                }
                # 近似直線の傾き、切片、相関係数を計算
                try:
                    slope, intercept, r_value, p_value, stderr = stats.linregress(
                        ch4_data, c2h6_data
                    )
                    ratio_row: dict[str, str | float] = {
                        "Date": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "slope": f"{slope:.6f}",
                        "intercept": f"{intercept:.6f}",
                        "r_value": f"{r_value:.6f}",
                        "p_value": f"{p_value:.6f}",
                        "stderr": f"{stderr:.6f}",
                    }
                except Exception:
                    # 何もせず、デフォルトの ratio_row を使用する
                    pass

                # 結果をリストに追加
                ratio_data.append(ratio_row)

        if output_c2c1_ratio:
            # DataFrameを作成し、Dateカラムで昇順ソート
            ratio_df: pd.DataFrame = pd.DataFrame(ratio_data)
            ratio_df["Date"] = pd.to_datetime(
                ratio_df["Date"]
            )  # Dateカラムをdatetime型に変換
            ratio_df = ratio_df.sort_values("Date")  # Dateカラムで昇順ソート

            # CSVとして保存
            ratio_filename: str = (
                f"{c2c1_ratio_csv_prefix}.{latest_date.strftime('%Y.%m.%d')}.ratio.csv"
            )
            ratio_path: str = os.path.join(c2c1_ratio_dirpath, ratio_filename)
            ratio_df.to_csv(ratio_path, index=False)

    @staticmethod
    def calculate_lag_time(
        df: pd.DataFrame,
        col1: str,
        col2_list: list[str],
    ) -> list[int]:
        """指定された基準変数と比較変数の間の遅れ時間(ディレイ)を計算します。
        周波数が10Hzでcol1がcol2より10.0秒遅れている場合は、+100がインデックスとして取得されます。

        Parameters
        ----------
            df: pd.DataFrame
                遅れ時間の計算に使用するデータフレーム
            col1: str
                基準変数の列名
            col2_list: list[str]
                比較変数の列名のリスト

        Returns
        -------
            list[int]
                各比較変数に対する遅れ時間(ディレイ)のリスト。正の値は基準変数が比較変数より遅れていることを示し、
                負の値は基準変数が比較変数より進んでいることを示します。

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'var1': [1, 2, 3, 4, 5],
        ...     'var2': [2, 3, 4, 5, 6],
        ...     'var3': [3, 4, 5, 6, 7]
        ... })
        >>> EddyDataPreprocessor.calculate_lag_time(df, 'var1', ['var2', 'var3'])
        [1, 2]
        """
        lags_list: list[int] = []
        for col2 in col2_list:
            data1: np.ndarray = np.array(df[col1].values)
            data2: np.ndarray = np.array(df[col2].values)

            # 平均を0に調整
            data1 = data1 - data1.mean()
            data2 = data2 - data2.mean()

            data_length: int = len(data1)

            # 相互相関の計算
            correlation: np.ndarray = np.correlate(
                data1, data2, mode="full"
            )  # data2とdata1の順序を入れ替え

            # 相互相関のピークのインデックスを取得
            lag: int = int((data_length - 1) - correlation.argmax())  # 符号を反転

            lags_list.append(lag)
        return lags_list

    @staticmethod
    def _get_sorted_files(
        directory: str | Path, pattern: str, suffix: str
    ) -> list[str]:
        """
        指定されたディレクトリ内のファイルを、ファイル名に含まれる数字に基づいてソートして返す。

        Parameters
        ----------
            directory: str | Path
                ファイルが格納されているディレクトリのパス
            pattern: str
                ファイル名からソートキーを抽出する正規表現パターン
            suffix: str
                ファイルの拡張子

        Returns
        ----------
            list[str]
                ソートされたファイル名のリスト
        """
        files: list[str] = [f for f in os.listdir(directory) if f.endswith(suffix)]
        files = [f for f in files if re.search(pattern, f)]
        files.sort(
            key=lambda x: int(re.search(pattern, x).group(1))  # type:ignore
            if re.search(pattern, x)
            else float("inf")
        )
        return files

    @staticmethod
    def _horizontal_wind_speed(
        x_array: np.ndarray, y_array: np.ndarray, wind_dir: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        風速のu成分とv成分を計算する関数

        Parameters
        ----------
            x_array: numpy.ndarray
                x方向の風速成分の配列
            y_array: numpy.ndarray
                y方向の風速成分の配列
            wind_dir: float
                水平成分の風向(ラジアン)

        Returns
        ----------
            tuple[numpy.ndarray, numpy.ndarray]
                u成分とv成分のタプル
        """
        # スカラー風速の計算
        scalar_hypotenuse: np.ndarray = np.sqrt(x_array**2 + y_array**2)
        # CSAT3では以下の補正が必要
        instantaneous_wind_directions = EddyDataPreprocessor._wind_direction(
            x_array=x_array, y_array=y_array
        )
        # ベクトル風速の計算
        vector_u: np.ndarray = scalar_hypotenuse * np.cos(
            instantaneous_wind_directions - wind_dir
        )
        vector_v: np.ndarray = scalar_hypotenuse * np.sin(
            instantaneous_wind_directions - wind_dir
        )
        return vector_u, vector_v

    @staticmethod
    def _vertical_rotation(
        u_array: np.ndarray,
        w_array: np.ndarray,
        wind_inc: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        鉛直方向の座標回転を行い、u, wを求める関数

        Parameters
        ----------
            u_array: numpy.ndarray
                u方向の風速。
            w_array: numpy.ndarray
                w方向の風速。
            wind_inc: float
                平均風向に対する迎角(ラジアン)。

        Returns
        -------
            tuple[numpy.ndarray, numpy.ndarray]
                回転後のuおよびwの配列。
        """
        # 迎角を用いて鉛直方向に座標回転
        u_rotated = u_array * np.cos(wind_inc) + w_array * np.sin(wind_inc)
        w_rotated = w_array * np.cos(wind_inc) - u_array * np.sin(wind_inc)
        return u_rotated, w_rotated

    @staticmethod
    def _wind_direction(
        x_array: np.ndarray, y_array: np.ndarray, correction_angle: float = 0.0
    ) -> float:
        """
        水平方向の平均風向を計算する関数。

        Parameters
        ----------
            x_array: numpy.ndarray
                西方向の風速成分。
            y_array: numpy.ndarray
                南北方向の風速成分。
            correction_angle: float, optional
                風向補正角度(ラジアン)。デフォルトは0.0。CSAT3の場合は0.0を指定。

        Returns
        -------
            float
                風向(ラジアン)。
        """
        wind_direction: float = np.arctan2(np.mean(y_array), np.mean(x_array))
        # 補正角度を適用
        wind_direction = correction_angle - wind_direction
        return wind_direction

    @staticmethod
    def _wind_inclination(u_array: np.ndarray, w_array: np.ndarray) -> float:
        """
        平均風向に対する迎角を計算する関数。

        Parameters
        ----------
            u_array: numpy.ndarray
                u方向の瞬間風速。
            w_array: numpy.ndarray
                w方向の瞬間風速。

        Returns
        -------
            float
                平均風向に対する迎角(ラジアン)。
        """
        wind_inc: float = np.arctan2(np.mean(w_array), np.mean(u_array))
        return wind_inc
