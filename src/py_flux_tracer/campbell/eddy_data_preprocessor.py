import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class EddyDataPreprocessor:
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
            fs (float): サンプリング周波数。
            logger (Logger | None): 使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug (bool): ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        self.fs: float = fs

        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = EddyDataPreprocessor.setup_logger(logger, log_level)

    def add_uvw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameに水平風速u、v、鉛直風速wの列を追加する関数。
        各成分のキーは`wind_u`、`wind_v`、`wind_w`である。
        
        Parameters
        -----
            df : pd.DataFrame
                風速データを含むDataFrame

        Returns
        -----
            pd.DataFrame
                水平風速u、v、鉛直風速wの列を追加したDataFrame
        """
        required_columns: list[str] = ["Ux", "Uy", "Uz"]
        # 必要な列がDataFrameに存在するか確認
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"必要な列 '{column}' がDataFrameに存在しません。")

        processed_df: pd.DataFrame = df.copy()
        # pandasの.valuesを使用してnumpy配列を取得し、その型をnp.ndarrayに明示的にキャストする
        wind_x_array: np.ndarray = np.array(processed_df["Ux"].values)
        wind_y_array: np.ndarray = np.array(processed_df["Uy"].values)
        wind_z_array: np.ndarray = np.array(processed_df["Uz"].values)

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

        processed_df["wind_u"] = wind_u_array_rotated
        processed_df["wind_v"] = wind_v_array
        processed_df["wind_w"] = wind_w_array_rotated
        processed_df["rad_wind_dir"] = wind_direction
        processed_df["rad_wind_inc"] = wind_inclination
        processed_df["degree_wind_dir"] = np.degrees(wind_direction)
        processed_df["degree_wind_inc"] = np.degrees(wind_inclination)

        return processed_df

    def analyze_lag_times(
        self,
        input_dir: str,
        figsize: tuple[float, float] = (10, 8),
        input_files_pattern: str = r"Eddy_(\d+)",
        input_files_suffix: str = ".dat",
        col1: str = "wind_w",
        col2_list: list[str] = ["Tv"],
        median_range: float = 20,
        metadata_rows: int = 4,
        output_dir: str | None = None,
        output_tag: str = "",
        plot_range_tuple: tuple = (-50, 200),
        print_results: bool = True,
        skiprows: list[int] = [0, 2, 3],
        use_resampling: bool = True,
    ) -> dict[str, float]:
        """
        遅れ時間（ラグ）の統計分析を行い、指定されたディレクトリ内のデータファイルを処理します。
        解析結果とメタデータはCSVファイルとして出力されます。

        Parameters:
        -----
            input_dir : str
                入力データファイルが格納されているディレクトリのパス。
            figsize : tuple[float, float]
                プロットのサイズ（幅、高さ）。
            input_files_pattern : str
                入力ファイル名のパターン（正規表現）。
            input_files_suffix : str
                入力ファイルの拡張子。
            col1 : str
                基準変数の列名。
            col2_list : list[str]
                比較変数の列名のリスト。
            median_range : float
                中央値を中心とした範囲。
            metadata_rows : int
                メタデータの行数。
            output_dir : str | None
                出力ディレクトリのパス。Noneの場合は保存しない。
            output_tag : str
                出力ファイルに付与するタグ。デフォルトは空文字で、何も付与されない。
            plot_range_tuple : tuple
                ヒストグラムの表示範囲。
            print_results : bool
                結果をコンソールに表示するかどうか。
            skiprows : list[int]
                スキップする行番号のリスト。
            use_resampling : bool
                データをリサンプリングするかどうか。
                inputするファイルが既にリサンプリング済みの場合はFalseでよい。
                デフォルトはTrue。

        Returns:
        -----
            dict[str, float]
                各変数の遅れ時間（平均値を採用）を含む辞書。
        """
        if output_dir is None:
            self.logger.warn(
                "output_dirが指定されていません。解析結果を保存する場合は、有効なディレクトリを指定してください。"
            )
        all_lags_indices: list[list[int]] = []
        results: dict[str, float] = {}

        # メイン処理
        # ファイル名に含まれる数字に基づいてソート
        csv_files = EddyDataPreprocessor._get_sorted_files(
            input_dir, input_files_pattern, input_files_suffix
        )
        if not csv_files:
            raise FileNotFoundError(
                f"There is no '{input_files_suffix}' file to process; input_dir: '{input_dir}', input_files_suffix: '{input_files_suffix}'"
            )

        for file in tqdm(csv_files, desc="Calculating"):
            path: str = os.path.join(input_dir, file)
            if use_resampling:
                df, _ = self.get_resampled_df(
                    filepath=path, metadata_rows=metadata_rows, skiprows=skiprows
                )
            else:
                df = pd.read_csv(path, skiprows=skiprows)
            df = self.add_uvw_columns(df)
            lags_list = EddyDataPreprocessor._calculate_lag_time(
                df,
                col1,
                col2_list,
            )
            all_lags_indices.append(lags_list)
        self.logger.info("すべてのCSVファイルにおける遅れ時間が計算されました。")

        # Convert all_lags_indices to a DataFrame
        lags_indices_df: pd.DataFrame = pd.DataFrame(
            all_lags_indices, columns=col2_list
        )

        # フォーマット用のキーの最大の長さ
        max_col_name_length: int = max(len(column) for column in lags_indices_df.columns)

        if print_results:
            self.logger.info(f"カラム`{col1}`に対する遅れ時間を表示します。")

        # 結果を格納するためのリスト
        output_data = []

        for column in lags_indices_df.columns:
            data: pd.Series = lags_indices_df[column]

            # ヒストグラムの作成
            plt.figure(figsize=figsize)
            plt.hist(data, bins=20, range=plot_range_tuple)
            plt.title(f"Delays of {column}")
            plt.xlabel("Seconds")
            plt.ylabel("Frequency")
            plt.xlim(plot_range_tuple)

            # ファイルとして保存するか
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                filename: str = f"lags_histogram-{column}{output_tag}.png"
                filepath: str = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()

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
                print(f"{column:<{max_col_name_length}} : {mean_seconds:.2f} s")

        # 結果をCSVファイルとして出力
        if output_dir is not None:
            output_df: pd.DataFrame = pd.DataFrame(output_data)
            csv_filepath: str = os.path.join(
                output_dir, f"lags_results{output_tag}.csv"
            )
            output_df.to_csv(csv_filepath, index=False, encoding="utf-8")
            self.logger.info(f"解析結果をCSVファイルに保存しました: {csv_filepath}")

        return results

    def get_resampled_df(
        self,
        filepath: str,
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        interpolate: bool = True,
        numeric_columns: list[str] = [
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
        ],
        metadata_rows: int = 4,
        skiprows: list[int] = [0, 2, 3],
        is_already_resampled: bool = False,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        CSVファイルを読み込み、前処理を行う

        前処理の手順は以下の通りです：
        1. 不要な行を削除する。デフォルト（`skiprows=[0, 2, 3]`）の場合は、2行目をヘッダーとして残し、1、3、4行目が削除される。
        2. 数値データを float 型に変換する
        3. TIMESTAMP列をDateTimeインデックスに設定する
        4. エラー値をNaNに置き換える
        5. 指定されたサンプリングレートでリサンプリングする
        6. 欠損値(NaN)を前後の値から線形補間する
        7. DateTimeインデックスを削除する
        
        Parameters:
        -----
            filepath : str
                読み込むCSVファイルのパス
            index_column : str, optional
                インデックスに使用する列名。デフォルトは'TIMESTAMP'。
            index_format : str, optional
                インデックスの日付形式。デフォルトは'%Y-%m-%d %H:%M:%S.%f'。
            interpolate : bool, optional
                欠損値の補完を適用するフラグ。デフォルトはTrue。
            numeric_columns : list[str], optional
                数値型に変換する列名のリスト。
                デフォルトは["Ux", "Uy", "Uz", "Tv", "diag_sonic", "CO2_new", "H2O", "diag_irga", "cell_tmpr", "cell_press", "Ultra_CH4_ppm", "Ultra_C2H6_ppb", "Ultra_H2O_ppm", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb_C"]。
            metadata_rows : int, optional
                メタデータとして読み込む行数。デフォルトは4。
            skiprows : list[int], optional
                スキップする行インデックスのリスト。デフォルトは[0, 2, 3]のため、1, 3, 4行目がスキップされる。
            is_already_resampled : bool
                既にリサンプリング&欠損補間されているか。Trueの場合はfloat変換などの処理のみ適用する。

        Returns:
        -----
            tuple[pd.DataFrame, list[str]]
                前処理済みのデータフレームとメタデータのリスト。
        """
        # メタデータを読み込む
        metadata: list[str] = []
        with open(filepath, "r") as f:
            for _ in range(metadata_rows):
                line = f.readline().strip()
                metadata.append(line.replace('"', ""))

        # CSVファイルを読み込む
        df: pd.DataFrame = pd.read_csv(filepath, skiprows=skiprows)

        # 数値データをfloat型に変換する
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if not is_already_resampled:
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
        input_dir: str,
        resampled_dir: str,
        ratio_dir: str,
        input_file_pattern: str = r"Eddy_(\d+)",
        input_files_suffix: str = ".dat",
        col_ch4_conc: str = "Ultra_CH4_ppm_C",
        col_c2h6_conc: str = "Ultra_C2H6_ppb",
        output_ratio: bool = True,
        output_resampled: bool = True,
        ratio_csv_prefix: str = "SAC.Ultra",
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        interpolate: bool = True,
        numeric_columns: list[str] = [
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
        ],
        metadata_rows: int = 4,
        skiprows: list[int] = [0, 2, 3],
    ) -> None:
        """
        指定されたディレクトリ内のCSVファイルを処理し、リサンプリングと欠損値補間を行います。

        このメソッドは、指定されたディレクトリ内のCSVファイルを読み込み、リサンプリングを行い、
        欠損値を補完します。処理結果として、リサンプリングされたCSVファイルを出力し、
        相関係数やC2H6/CH4比を計算してDataFrameに保存します。
        リサンプリングと欠損値補完は`get_resampled_df`と同様のロジックを使用します。

        Parameters:
        -----
            input_dir : str
                入力CSVファイルが格納されているディレクトリのパス。
            resampled_dir : str
                リサンプリングされたCSVファイルを出力するディレクトリのパス。
            ratio_dir : str
                計算結果を保存するディレクトリのパス。
            input_file_pattern : str
                ファイル名からソートキーを抽出する正規表現パターン。デフォルトでは、最初の数字グループでソートします。
            input_files_suffix : str
                入力ファイルの拡張子（.datや.csvなど）。デフォルトは".dat"。
            col_ch4_conc : str
                CH4濃度を含む列名。デフォルトは'Ultra_CH4_ppm_C'。
            col_c2h6_conc : str
                C2H6濃度を含む列名。デフォルトは'Ultra_C2H6_ppb'。
            output_ratio : bool, optional
                線形回帰を行うかどうか。デフォルトはTrue。
            output_resampled : bool, optional
                リサンプリングされたCSVファイルを出力するかどうか。デフォルトはTrue。
            ratio_csv_prefix : str
                出力ファイルの接頭辞。デフォルトは'SAC.Ultra'で、出力時は'SAC.Ultra.2024.09.21.ratio.csv'のような形式となる。
            index_column : str
                日時情報を含む列名。デフォルトは'TIMESTAMP'。
            index_format : str, optional
                インデックスの日付形式。デフォルトは'%Y-%m-%d %H:%M:%S.%f'。
            interpolate : bool
                欠損値補間を行うかどうか。デフォルトはTrue。
            numeric_columns : list[str]
                数値データを含む列名のリスト。デフォルトは指定された列名のリスト。
            metadata_rows : int
                メタデータとして読み込む行数。デフォルトは4。
            skiprows : list[int]
                読み飛ばす行のインデックスリスト。デフォルトは[0, 2, 3]。

        Raises:
        -----
            OSError
                ディレクトリの作成に失敗した場合。
            FileNotFoundError
                入力ファイルが見つからない場合。
            ValueError
                出力ディレクトリが指定されていない、またはデータの処理中にエラーが発生した場合。
        """
        # 出力オプションとディレクトリの検証
        if output_resampled and resampled_dir is None:
            raise ValueError("output_resampled が True の場合、resampled_dir を指定する必要があります")
        if output_ratio and ratio_dir is None:
            raise ValueError("output_ratio が True の場合、ratio_dir を指定する必要があります")

        # ディレクトリの作成（必要な場合のみ）
        if output_resampled:
            os.makedirs(resampled_dir, exist_ok=True)
        if output_ratio:
            os.makedirs(ratio_dir, exist_ok=True)

        ratio_data: list[dict[str, str | float]] = []
        latest_date: datetime = datetime.min

        # csvファイル名のリスト
        csv_files: list[str] = EddyDataPreprocessor._get_sorted_files(
            input_dir, input_file_pattern, input_files_suffix
        )

        for filename in tqdm(csv_files, desc="Processing files"):
            input_filepath: str = os.path.join(input_dir, filename)
            # リサンプリング＆欠損値補間
            df, metadata = self.get_resampled_df(
                filepath=input_filepath,
                index_column=index_column,
                index_format=index_format,
                interpolate=interpolate,
                numeric_columns=numeric_columns,
                metadata_rows=metadata_rows,
                skiprows=skiprows,
            )

            # 開始時間を取得
            start_time: datetime = pd.to_datetime(df[index_column].iloc[0])
            # 処理したファイルの中で最も最新の日付
            latest_date = max(latest_date, start_time)

            # リサンプリング＆欠損値補間したCSVを出力
            if output_resampled:
                base_filename: str = re.sub(rf"\{input_files_suffix}$", "", filename)
                output_csv_path: str = os.path.join(
                    resampled_dir, f"{base_filename}-resampled.csv"
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
            if output_ratio:
                ch4_data: pd.Series = df[col_ch4_conc]
                c2h6_data: pd.Series = df[col_c2h6_conc]

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

        if output_ratio:
            # DataFrameを作成し、Dateカラムで昇順ソート
            ratio_df: pd.DataFrame = pd.DataFrame(ratio_data)
            ratio_df["Date"] = pd.to_datetime(
                ratio_df["Date"]
            )  # Dateカラムをdatetime型に変換
            ratio_df = ratio_df.sort_values("Date")  # Dateカラムで昇順ソート

            # CSVとして保存
            ratio_filename: str = (
                f"{ratio_csv_prefix}.{latest_date.strftime('%Y.%m.%d')}.ratio.csv"
            )
            ratio_path: str = os.path.join(ratio_dir, ratio_filename)
            ratio_df.to_csv(ratio_path, index=False)
    
    def resample_and_analyze_lag_times(
        self,
        input_dir: str,
        input_file_pattern: str = r"Eddy_(\d+)",
        input_files_suffix: str = ".dat",
        col_ch4_conc: str = "Ultra_CH4_ppm_C",
        col_c2h6_conc: str = "Ultra_C2H6_ppb",
        output_ratio: bool = True,
        ratio_dir: str | None = None,
        output_resampled: bool = True,
        resampled_dir: str | None = None,
        output_lag_times: bool = True,  # lag times解析の有効化フラグ
        lag_times_dir: str | None = None,  # lag timesの結果出力ディレクトリ
        lag_times_col1: str = "wind_w",  # 基準変数
        lag_times_col2_list: list[str] = ["Tv"],  # 比較変数のリスト
        lag_times_median_range: float = 20,  # 中央値を中心とした範囲
        lag_times_plot_range: tuple[float, float] = (
            -50,
            200,
        ),  # ヒストグラムの表示範囲
        lag_times_figsize: tuple[float, float] = (10, 8),  # プロットサイズ
        ratio_csv_prefix: str = "SAC.Ultra",
        index_column: str = "TIMESTAMP",
        index_format: str = "%Y-%m-%d %H:%M:%S.%f",
        interpolate: bool = True,
        numeric_columns: list[str] = [
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
        ],
        metadata_rows: int = 4,
        skiprows: list[int] = [0, 2, 3],
    ) -> None:
        """
        指定されたディレクトリ内のCSVファイルを処理し、リサンプリングと欠損値補間を行います。

        このメソッドは、指定されたディレクトリ内のCSVファイルを読み込み、リサンプリングを行い、
        欠損値を補完します。処理結果として以下の出力が可能です：
        1. リサンプリングされたCSVファイル (output_resampled=True)
        2. 相関係数やC2H6/CH4比を計算したDataFrame (output_ratio=True)
        3. lag times解析結果 (output_lag_times=True)

        Parameters:
        -----
            input_dir : str
                入力CSVファイルが格納されているディレクトリのパス。
            resampled_dir : str | None
                リサンプリングされたCSVファイルを出力するディレクトリのパス。
            ratio_dir : str | None
                C2H6/CH4比の計算結果を保存するディレクトリのパス。
            input_file_pattern : str
                ファイル名からソートキーを抽出する正規表現パターン。
            input_files_suffix : str
                入力ファイルの拡張子（.datや.csvなど）。デフォルトは".dat"。
            col_ch4_conc : str
                CH4濃度を含む列名。デフォルトは'Ultra_CH4_ppm_C'。
            col_c2h6_conc : str
                C2H6濃度を含む列名。デフォルトは'Ultra_C2H6_ppb'。
            output_ratio : bool
                線形回帰を行うかどうか。デフォルトはTrue。
            output_resampled : bool
                リサンプリングされたCSVファイルを出力するかどうか。デフォルトはTrue。
            output_lag_times : bool
                lag times解析を行うかどうか。デフォルトはFalse。
            lag_times_dir : str | None
                lag times解析結果の出力ディレクトリ。
            lag_times_col1 : str
                lag times解析の基準変数。デフォルトは"wind_w"。
            lag_times_col2_list : list[str]
                lag times解析の比較変数のリスト。デフォルトは["Tv"]。
            lag_times_median_range : float
                lag times解析の中央値を中心とした範囲。デフォルトは20。
            lag_times_plot_range : tuple[float, float]
                lag times解析のヒストグラム表示範囲。デフォルトは(-50, 200)。
            lag_times_figsize : tuple[float, float]
                lag times解析のプロットサイズ。デフォルトは(10, 8)。
            ratio_csv_prefix : str
                出力ファイルの接頭辞。
            index_column : str
                日時情報を含む列名。デフォルトは'TIMESTAMP'。
            index_format : str
                インデックスの日付形式。デフォルトは'%Y-%m-%d %H:%M:%S.%f'。
            interpolate : bool
                欠損値補間を行うかどうか。デフォルトはTrue。
            numeric_columns : list[str]
                数値データを含む列名のリスト。
            metadata_rows : int
                メタデータとして読み込む行数。デフォルトは4。
            skiprows : list[int]
                読み飛ばす行のインデックスリスト。デフォルトは[0, 2, 3]。

        Raises:
        -----
            ValueError
                出力オプションが指定されているのにディレクトリが指定されていない場合。
            FileNotFoundError
                入力ファイルが見つからない場合。
            OSError
                ディレクトリの作成に失敗した場合。
        """
        # 出力オプションとディレクトリの検証
        if output_resampled and resampled_dir is None:
            raise ValueError(
                "output_resampled が True の場合、resampled_dir を指定する必要があります"
            )
        if output_ratio and ratio_dir is None:
            raise ValueError(
                "output_ratio が True の場合、ratio_dir を指定する必要があります"
            )
        if output_lag_times and lag_times_dir is None:
            raise ValueError(
                "output_lag_times が True の場合、lag_times_dir を指定する必要があります"
            )

        # ディレクトリの作成（必要な場合のみ）
        if output_resampled and resampled_dir is not None:
            os.makedirs(resampled_dir, exist_ok=True)
        if output_ratio and ratio_dir is not None:
            os.makedirs(ratio_dir, exist_ok=True)
        if output_lag_times and lag_times_dir is not None:
            os.makedirs(lag_times_dir, exist_ok=True)

        ratio_data: list[dict[str, str | float]] = []
        all_lags_indices: list[list[int]] = []
        latest_date: datetime = datetime.min

        # csvファイル名のリスト
        csv_files: list[str] = EddyDataPreprocessor._get_sorted_files(
            input_dir, input_file_pattern, input_files_suffix
        )

        if not csv_files:
            raise FileNotFoundError(
                f"There is no '{input_files_suffix}' file to process; input_dir: '{input_dir}'"
            )

        for filename in tqdm(csv_files, desc="Processing files"):
            input_filepath: str = os.path.join(input_dir, filename)
            # リサンプリング＆欠損値補間
            df, metadata = self.get_resampled_df(
                filepath=input_filepath,
                index_column=index_column,
                index_format=index_format,
                interpolate=interpolate,
                numeric_columns=numeric_columns,
                metadata_rows=metadata_rows,
                skiprows=skiprows,
            )

            # 開始時間を取得
            start_time: datetime = pd.to_datetime(df[index_column].iloc[0])
            # 処理したファイルの中で最も最新の日付を更新
            latest_date = max(latest_date, start_time)

            # リサンプリング＆欠損値補間したCSVを出力
            if output_resampled and resampled_dir is not None:
                base_filename: str = re.sub(rf"\{input_files_suffix}$", "", filename)
                output_csv_path: str = os.path.join(
                    resampled_dir, f"{base_filename}-resampled.csv"
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
            if output_ratio:
                ch4_data: pd.Series = df[col_ch4_conc]
                c2h6_data: pd.Series = df[col_c2h6_conc]

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
                    ratio_row = {
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

                ratio_data.append(ratio_row)

            # Lag times解析用のデータを収集
            if output_lag_times:
                df = self.add_uvw_columns(df)
                lags_list = EddyDataPreprocessor._calculate_lag_time(
                    df,
                    lag_times_col1,
                    lag_times_col2_list,
                )
                all_lags_indices.append(lags_list)

        # Ratio解析結果の保存
        if output_ratio and ratio_dir is not None:
            # DataFrameを作成し、Dateカラムで昇順ソート
            ratio_df: pd.DataFrame = pd.DataFrame(ratio_data)
            ratio_df["Date"] = pd.to_datetime(ratio_df["Date"])
            ratio_df = ratio_df.sort_values("Date")

            # CSVとして保存
            ratio_filename: str = (
                f"{ratio_csv_prefix}.{latest_date.strftime('%Y.%m.%d')}.ratio.csv"
            )
            ratio_path: str = os.path.join(ratio_dir, ratio_filename)
            ratio_df.to_csv(ratio_path, index=False)
            self.logger.info(f"Ratio解析結果を保存しました: {ratio_path}")

        # Lag times解析結果の処理と保存
        if output_lag_times and lag_times_dir is not None:
            # lag timesの解析結果をDataFrameに変換
            lags_indices_df = pd.DataFrame(
                all_lags_indices, columns=lag_times_col2_list
            )
            lag_times_output_data = []

            # 各変数に対する解析
            for column in lags_indices_df.columns:
                data = lags_indices_df[column]

                # ヒストグラムの作成
                plt.figure(figsize=lag_times_figsize)
                plt.hist(data, bins=20, range=lag_times_plot_range)
                plt.title(f"Delays of {column}")
                plt.xlabel("Seconds")
                plt.ylabel("Frequency")
                plt.xlim(lag_times_plot_range)

                # ヒストグラムの保存
                filename = f"lags_histogram-{column}.png"
                filepath = os.path.join(lag_times_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()

                # 中央値を計算し、その周辺のデータのみを使用
                median_value = np.median(data)
                filtered_data = data[
                    (data >= median_value - lag_times_median_range)
                    & (data <= median_value + lag_times_median_range)
                ]

                # 平均値を計算
                mean_value = np.mean(filtered_data)
                mean_seconds = float(mean_value / self.fs)

                # 結果を格納
                lag_times_output_data.append(
                    {
                        "col1": lag_times_col1,
                        "col2": column,
                        "col2_lag": round(mean_seconds, 2),
                        "lag_unit": "s",
                        "median_range": lag_times_median_range,
                    }
                )

            # 結果をCSVとして保存
            if lag_times_output_data:
                lag_times_df = pd.DataFrame(lag_times_output_data)
                lag_times_csv_path = os.path.join(lag_times_dir, "lags_results.csv")
                lag_times_df.to_csv(lag_times_csv_path, index=False, encoding="utf-8")
                self.logger.info(
                    f"Lag times解析結果を保存しました: {lag_times_csv_path}"
                )

                # 遅れ時間を表示
                self.logger.info(f"カラム`{lag_times_col1}`に対する遅れ時間:")
                max_col_name_length = max(len(column) for column in lag_times_df["col2"])
                for _, row in lag_times_df.iterrows():
                    print(f"{row['col2']:<{max_col_name_length}} : {row['col2_lag']:.2f} s")

    @staticmethod
    def _calculate_lag_time(
        df: pd.DataFrame,
        col1: str,
        col2_list: list[str],
    ) -> list[int]:
        """
        指定された基準変数（col1）と比較変数のリスト（col2_list）の間の遅れ時間（ディレイ）を計算する。
        周波数が10Hzでcol1がcol2より10.0秒遅れている場合は、+100がインデックスとして取得される

        Parameters:
        -----
            df : pd.DataFrame
                遅れ時間の計算に使用するデータフレーム
            col1 : str
                基準変数の列名
            col2_list : list[str]
                比較変数の列名のリスト

        Returns:
        -----
            list[int]
                各比較変数に対する遅れ時間（ディレイ）のリスト
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
    def _get_sorted_files(directory: str, pattern: str, suffix: str) -> list[str]:
        """
        指定されたディレクトリ内のファイルを、ファイル名に含まれる数字に基づいてソートして返す。

        Parameters:
        -----
            directory : str
                ファイルが格納されているディレクトリのパス
            pattern : str
                ファイル名からソートキーを抽出する正規表現パターン
            suffix : str
                ファイルの拡張子

        Returns:
        -----
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

        Parameters:
        -----
            x_array : numpy.ndarray
                x方向の風速成分の配列
            y_array : numpy.ndarray
                y方向の風速成分の配列
            wind_dir : float
                水平成分の風向（ラジアン）

        Returns:
        -----
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
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        Parameters:
        -----
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level : int
                ロガーのログレベル。デフォルトはINFO。

        Returns:
        -----
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
    def _vertical_rotation(
        u_array: np.ndarray,
        w_array: np.ndarray,
        wind_inc: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        鉛直方向の座標回転を行い、u, wを求める関数

        Parameters:
        -----
            u_array (numpy.ndarray): u方向の風速
            w_array (numpy.ndarray): w方向の風速
            wind_inc (float): 平均風向に対する迎角（ラジアン）

        Returns:
        -----
            tuple[numpy.ndarray, numpy.ndarray]: 回転後のu, w
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
        水平方向の平均風向を計算する関数

        Parameters:
        -----
            x_array (numpy.ndarray): 西方向の風速成分
            y_array (numpy.ndarray): 南北方向の風速成分
            correction_angle (float): 風向補正角度（ラジアン）。デフォルトは0.0。CSAT3の場合は0.0を指定。

        Returns:
        -----
            wind_direction (float): 風向 (radians)
        """
        wind_direction: float = np.arctan2(np.mean(y_array), np.mean(x_array))
        # 補正角度を適用
        wind_direction = correction_angle - wind_direction
        return wind_direction

    @staticmethod
    def _wind_inclination(u_array: np.ndarray, w_array: np.ndarray) -> float:
        """
        平均風向に対する迎角を計算する関数

        Parameters:
        -----
            u_array (numpy.ndarray): u方向の瞬間風速
            w_array (numpy.ndarray): w方向の瞬間風速

        Returns:
        -----
            wind_inc (float): 平均風向に対する迎角（ラジアン）
        """
        wind_inc: float = np.arctan2(np.mean(w_array), np.mean(u_array))
        return wind_inc
