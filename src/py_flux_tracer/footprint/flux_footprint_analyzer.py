import io
import os
import math
import requests
import jpholiday
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from PIL.ImageFile import ImageFile
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from ..commons.hotspot_data import HotspotData, HotspotType


class FluxFootprintAnalyzer:
    """
    フラックスフットプリントを解析および可視化するクラス。

    このクラスは、フラックスデータの処理、フットプリントの計算、
    および結果を衛星画像上に可視化するメソッドを提供します。
    座標系と単位に関する重要な注意:
    - すべての距離はメートル単位で計算されます
    - 座標系の原点(0,0)は測定タワーの位置に対応します
    - x軸は東西方向（正が東）
    - y軸は南北方向（正が北）
    - 風向は気象学的風向（北から時計回りに測定）を使用

    この実装は、Kormann and Meixner (2001) および Takano et al. (2021)に基づいています。
    """

    EARTH_RADIUS_METER: int = 6371000  # 地球の半径（メートル）

    def __init__(
        self,
        z_m: float,
        labelsize: float = 20,
        ticksize: float = 16,
        plot_params=None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        衛星画像を用いて FluxFootprintAnalyzer を初期化します。

        Parameters:
        ------
            z_m : float
                測定の高さ（メートル単位）。
            labelsize : float
                軸ラベルのフォントサイズ。デフォルトは20。
            ticksize : float
                軸目盛りのフォントサイズ。デフォルトは16。
            plot_params : Optional[Dict[str, any]]
                matplotlibのプロットパラメータを指定する辞書。
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを生成します。
            logging_debug : bool
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # 定数や共通の変数
        self._required_columns: list[str] = [
            "Date",
            "WS vector",
            "u*",
            "z/L",
            "Wind direction",
            "sigmaV",
        ]  # 必要なカラムの名前
        self._col_weekday: str = "ffa_is_weekday"  # クラスで生成するカラムのキー名
        self._z_m: float = z_m  # 測定高度
        # 状態を管理するフラグ
        self._got_satellite_image: bool = False

        # 図表の初期設定
        FluxFootprintAnalyzer.setup_plot_params(labelsize, ticksize, plot_params)
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = FluxFootprintAnalyzer.setup_logger(logger, log_level)

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
        ------
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level : int
                ロガーのログレベル。デフォルトはINFO。

        Returns:
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
    def setup_plot_params(labelsize: float, ticksize: float, plot_params=None) -> None:
        """
        matplotlibのプロットパラメータを設定します。

        Parameters:
        ------
            labelsize : float
                軸ラベルのフォントサイズ。
            ticksize : float
                軸目盛りのフォントサイズ。
            plot_params : Optional[Dict[str, any]]
                matplotlibのプロットパラメータの辞書。

        Returns:
        ------
            None
                このメソッドは戻り値を持ちませんが、プロットパラメータを更新します。
        """
        # デフォルトのプロットパラメータ
        default_params = {
            "font.family": ["Arial", "Dejavu Sans"],
            "axes.edgecolor": "None",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "gray",
            "axes.grid": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "ytick.direction": "out",
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
            "grid.linewidth": 1.0,
            "font.size": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
        }

        # plot_paramsが定義されている場合、デフォルトに追記
        if plot_params:
            default_params.update(plot_params)

        plt.rcParams.update(default_params)  # プロットパラメータを更新

    def calculate_flux_footprint(
        self,
        df: pd.DataFrame,
        col_flux: str,
        plot_count: int = 10000,
        start_time: str = "10:00",
        end_time: str = "16:00",
    ) -> tuple[list[float], list[float], list[float]]:
        """
        フラックスフットプリントを計算し、指定された時間帯のデータを基に可視化します。

        Parameters:
        ------
            df : pd.DataFrame
                分析対象のデータフレーム。フラックスデータを含む。
            col_flux : str
                フラックスデータの列名。計算に使用される。
            plot_count : int, optional
                生成するプロットの数。デフォルトは10000。
            start_time : str, optional
                フットプリント計算に使用する開始時間。デフォルトは"10:00"。
            end_time : str, optional
                フットプリント計算に使用する終了時間。デフォルトは"16:00"。

        Returns:
        ------
            tuple[list[float], list[float], list[float]]:
                x座標 (メートル): タワーを原点とした東西方向の距離
                y座標 (メートル): タワーを原点とした南北方向の距離
                対象スカラー量の値: 各地点でのフラックス値

        Notes:
        ------
            - 返却される座標は測定タワーを原点(0,0)とした相対位置です
            - すべての距離はメートル単位で表されます
            - 正のx値は東方向、正のy値は北方向を示します
        """
        df: pd.DataFrame = df.copy()

        # インデックスがdatetimeであることを確認し、必要に応じて変換
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # DatetimeIndexから直接dateプロパティにアクセス
        datelist: np.ndarray = np.array(df.index.date)

        # 各日付が平日かどうかを判定し、リストに格納
        numbers: list[int] = [
            FluxFootprintAnalyzer.is_weekday(date) for date in datelist
        ]

        # col_weekdayに基づいてデータフレームに平日情報を追加
        df.loc[:, self._col_weekday] = numbers  # .locを使用して値を設定

        # 値が1のもの(平日)をコピーする
        data_weekday: pd.DataFrame = df[df[self._col_weekday] == 1].copy()
        # 特定の時間帯を抽出
        data_weekday = data_weekday.between_time(
            start_time, end_time
        )  # 引数を使用して時間帯を抽出
        data_weekday = data_weekday.dropna(subset=[col_flux])

        directions: list[float] = [
            wind_direction if wind_direction >= 0 else wind_direction + 360
            for wind_direction in data_weekday["Wind direction"]
        ]

        data_weekday.loc[:, "Wind direction_360"] = directions
        data_weekday.loc[:, "radian"] = data_weekday["Wind direction_360"] / 180 * np.pi

        # 風向が欠測なら除去
        data_weekday = data_weekday.dropna(subset=["Wind direction", col_flux])

        # 数値型への変換を確実に行う
        numeric_columns: list[str] = ["u*", "WS vector", "sigmaV", "z/L"]
        for col in numeric_columns:
            data_weekday[col] = pd.to_numeric(data_weekday[col], errors="coerce")

        # 地面修正量dの計算
        z_m: float = self._z_m
        Z_d: float = FluxFootprintAnalyzer._calculate_ground_correction(
            z_m=z_m,
            wind_speed=data_weekday["WS vector"].values,
            friction_velocity=data_weekday["u*"].values,
            stability_parameter=data_weekday["z/L"].values,
        )

        x_list: list[float] = []
        y_list: list[float] = []
        c_list: list[float] | None = []

        # tqdmを使用してプログレスバーを表示
        for i in tqdm(range(len(data_weekday)), desc="Calculating footprint"):
            dUstar: float = data_weekday["u*"].iloc[i]
            dU: float = data_weekday["WS vector"].iloc[i]
            sigmaV: float = data_weekday["sigmaV"].iloc[i]
            dzL: float = data_weekday["z/L"].iloc[i]

            if pd.isna(dUstar) or pd.isna(dU) or pd.isna(sigmaV) or pd.isna(dzL):
                self.logger.warning(f"#N/A fields are exist.: i = {i}")
                continue
            elif dUstar < 5.0 and dUstar != 0.0 and dU > 0.1:
                phi_m, phi_c, n = FluxFootprintAnalyzer._calculate_stability_parameters(
                    dzL=dzL
                )
                m, U, r, mu, ksi = (
                    FluxFootprintAnalyzer._calculate_footprint_parameters(
                        dUstar=dUstar, dU=dU, Z_d=Z_d, phi_m=phi_m, phi_c=phi_c, n=n
                    )
                )

                # 80%ソースエリアの計算
                x80: float = FluxFootprintAnalyzer._source_area_KM2001(
                    ksi=ksi, mu=mu, dU=dU, sigmaV=sigmaV, Z_d=Z_d, max_ratio=0.8
                )

                if not np.isnan(x80):
                    x1, y1, flux1 = FluxFootprintAnalyzer._prepare_plot_data(
                        x80,
                        ksi,
                        mu,
                        r,
                        U,
                        m,
                        sigmaV,
                        data_weekday[col_flux].iloc[i],
                        plot_count=plot_count,
                    )
                    x1_, y1_ = FluxFootprintAnalyzer._rotate_coordinates(
                        x=x1, y=y1, radian=data_weekday["radian"].iloc[i]
                    )

                    x_list.extend(x1_)
                    y_list.extend(y1_)
                    c_list.extend(flux1)

        return (
            x_list,
            y_list,
            c_list,
        )

    def combine_all_data(
        self, data_source: str | pd.DataFrame, source_type: str = "csv", **kwargs
    ) -> pd.DataFrame:
        """
        CSVファイルまたはMonthlyConverterからのデータを統合します

        Parameters:
        ------
            data_source : str | pd.DataFrame
                CSVディレクトリパスまたはDataFrame
            source_type : str
                "csv" または "monthly"
            **kwargs :
                追加パラメータ
                - sheet_names : list[str]
                    Monthlyの場合のシート名
                - start_date : str
                    開始日
                - end_date : str
                    終了日

        Returns:
        ------
            pd.DataFrame
                処理済みのデータフレーム
        """
        if source_type == "csv":
            # 既存のCSV処理ロジック
            return self._combine_all_csv(data_source)
        elif source_type == "monthly":
            # MonthlyConverterからのデータを処理
            if not isinstance(data_source, pd.DataFrame):
                raise ValueError("monthly形式の場合、DataFrameを直接渡す必要があります")

            df = data_source.copy()

            # required_columnsからDateを除外して欠損値チェックを行う
            check_columns = [col for col in self._required_columns if col != "Date"]

            # インデックスがdatetimeであることを確認
            if not isinstance(df.index, pd.DatetimeIndex) and "Date" not in df.columns:
                raise ValueError("DatetimeIndexまたはDateカラムが必要です")

            if "Date" in df.columns:
                df.set_index("Date", inplace=True)

            # 必要なカラムの存在確認
            missing_columns = [
                col for col in check_columns if col not in df.columns.tolist()
            ]
            if missing_columns:
                missing_cols = "','".join(missing_columns)
                current_cols = "','".join(df.columns.tolist())
                raise ValueError(
                    f"必要なカラムが不足しています: '{missing_cols}'\n"
                    f"現在のカラム: '{current_cols}'"
                )

            # 平日/休日の判定用カラムを追加
            df[self._col_weekday] = df.index.map(FluxFootprintAnalyzer.is_weekday)

            # Dateを除外したカラムで欠損値の処理
            df = df.dropna(subset=check_columns)

            # インデックスの重複を除去
            df = df.loc[~df.index.duplicated(), :]

            return df
        else:
            raise ValueError("source_typeは'csv'または'monthly'である必要があります")

    def get_satellite_image_from_api(
        self,
        api_key: str,
        center_lat: float,
        center_lon: float,
        output_path: str,
        scale: int = 1,
        size: tuple[int, int] = (2160, 2160),
        zoom: int = 13,
    ) -> ImageFile:
        """
        Google Maps Static APIを使用して衛星画像を取得します。

        Parameters:
        ------
            api_key : str
                Google Maps Static APIのキー。
            center_lat : float
                中心の緯度。
            center_lon : float
                中心の経度。
            output_path : str
                画像の保存先パス。拡張子は'.png'のみ許可される。
            scale : int, optional
                画像の解像度スケール（1か2）。デフォルトは1。
            size : tuple[int, int], optional
                画像サイズ (幅, 高さ)。デフォルトは(2160, 2160)。
            zoom : int, optional
                ズームレベル（0-21）。デフォルトは13。

        Returns:
        ------
            ImageFile
                取得した衛星画像

        Raises:
        ------
            requests.RequestException
                API呼び出しに失敗した場合
        """
        # バリデーション
        if not output_path.endswith(".png"):
            raise ValueError("出力ファイル名は'.png'で終わる必要があります。")

        # HTTPリクエストの定義
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{center_lat},{center_lon}",
            "zoom": zoom,
            "size": f"{size[0]}x{size[1]}",
            "maptype": "satellite",
            "scale": scale,
            "key": api_key,
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            # 画像ファイルに変換
            image = Image.open(io.BytesIO(response.content))
            image.save(output_path)
            self._got_satellite_image = True
            self.logger.info(f"リモート画像を取得し、保存しました: {output_path}")
            return image
        except requests.RequestException as e:
            self.logger.error(f"衛星画像の取得に失敗しました: {str(e)}")
            raise

    def get_satellite_image_from_local(
        self,
        local_image_path: str,
    ) -> ImageFile:
        """
        ローカルファイルから衛星画像を読み込みます。

        Parameters:
        ------
            local_image_path : str
                ローカル画像のパス

        Returns:
        ------
            ImageFile
                読み込んだ衛星画像

        Raises:
        ------
            FileNotFoundError
                指定されたパスにファイルが存在しない場合
        """
        if not os.path.exists(local_image_path):
            raise FileNotFoundError(
                f"指定されたローカル画像が存在しません: {local_image_path}"
            )
        image = Image.open(local_image_path)
        self._got_satellite_image = True
        self.logger.info(f"ローカル画像を使用しました: {local_image_path}")
        return image

    def plot_flux_footprint(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float] | None,
        center_lat: float,
        center_lon: float,
        vmin: float,
        vmax: float,
        add_cbar: bool = True,
        add_legend: bool = True,
        cbar_label: str | None = None,
        cbar_labelpad: int = 20,
        cmap: str = "jet",
        reduce_c_function: callable = np.mean,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dir: str | None = None,
        output_filename: str = "footprint.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: ImageFile | None = None,
        xy_max: float = 5000,
    ) -> None:
        """
        フットプリントデータをプロットします。

        このメソッドは、指定されたフットプリントデータのみを可視化します。

        Parameters:
        ------
            x_list : list[float]
                フットプリントのx座標リスト（メートル単位）。
            y_list : list[float]
                フットプリントのy座標リスト（メートル単位）。
            c_list : list[float] | None
                フットプリントの強度を示す値のリスト。
            center_lat : float
                プロットの中心となる緯度。
            center_lon : float
                プロットの中心となる経度。
            cmap : str
                使用するカラーマップの名前。
            vmin : float
                カラーバーの最小値。
            vmax : float
                カラーバーの最大値。
            reduce_c_function : callable, optional
                フットプリントの集約関数（デフォルトはnp.mean）。
            cbar_label : str | None, optional
                カラーバーのラベル。
            cbar_labelpad : int, optional
                カラーバーラベルのパディング。
            lon_correction : float, optional
                経度方向の補正係数（デフォルトは1）。
            lat_correction : float, optional
                緯度方向の補正係数（デフォルトは1）。
            output_dir : str | None, optional
                プロット画像の保存先パス。
            output_filename : str
                プロット画像の保存ファイル名（拡張子を含む）。デフォルトは'footprint.png'。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
            show_fig : bool
                図の表示を許可するフラグ。デフォルトはTrue。
            satellite_image : ImageFile | None, optional
                使用する衛星画像。指定がない場合はデフォルトの画像が生成されます。
            xy_max : float, optional
                表示範囲の最大値（デフォルトは4000）。
        """
        self.plot_flux_footprint_with_hotspots(
            x_list=x_list,
            y_list=y_list,
            c_list=c_list,
            center_lat=center_lat,
            center_lon=center_lon,
            vmin=vmin,
            vmax=vmax,
            add_cbar=add_cbar,
            add_legend=add_legend,
            cbar_label=cbar_label,
            cbar_labelpad=cbar_labelpad,
            cmap=cmap,
            reduce_c_function=reduce_c_function,
            hotspots=None,  # hotspotsをNoneに設定
            hotspot_colors=None,
            lat_correction=lat_correction,
            lon_correction=lon_correction,
            output_dir=output_dir,
            output_filename=output_filename,
            save_fig=save_fig,
            show_fig=show_fig,
            satellite_image=satellite_image,
            xy_max=xy_max,
        )

    def plot_flux_footprint_with_hotspots(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float] | None,
        center_lat: float,
        center_lon: float,
        vmin: float,
        vmax: float,
        add_cbar: bool = True,
        add_legend: bool = True,
        cbar_label: str | None = None,
        cbar_labelpad: int = 20,
        cmap: str = "jet",
        reduce_c_function: callable = np.mean,
        hotspots: list[HotspotData] | None = None,
        hotspot_colors: dict[HotspotType, str] | None = None,
        hotspot_markers: dict[HotspotType, str] | None = None,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dir: str | None = None,
        output_filename: str = "footprint.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: ImageFile | None = None,
        xy_max: float = 5000,
    ) -> None:
        """
        Staticな衛星画像上にフットプリントデータとホットスポットをプロットします。

        このメソッドは、指定されたフットプリントデータとホットスポットを可視化します。
        ホットスポットが指定されない場合は、フットプリントのみ作図します。

        Parameters:
        ------
            x_list : list[float]
                フットプリントのx座標リスト（メートル単位）。
            y_list : list[float]
                フットプリントのy座標リスト（メートル単位）。
            c_list : list[float] | None
                フットプリントの強度を示す値のリスト。
            center_lat : float
                プロットの中心となる緯度。
            center_lon : float
                プロットの中心となる経度。
            vmin : float
                カラーバーの最小値。
            vmax : float
                カラーバーの最大値。
            add_cbar : bool, optional
                カラーバーを追加するかどうか（デフォルトはTrue）。
            add_legend : bool, optional
                凡例を追加するかどうか（デフォルトはTrue）。
            cbar_label : str | None, optional
                カラーバーのラベル。
            cbar_labelpad : int, optional
                カラーバーラベルのパディング。
            cmap : str
                使用するカラーマップの名前。
            reduce_c_function : callable
                フットプリントの集約関数（デフォルトはnp.mean）。
            hotspots : list[HotspotData] | None, optional
                ホットスポットデータのリスト。デフォルトはNone。
            hotspot_colors : dict[HotspotType, str] | None, optional
                ホットスポットの色を指定する辞書。
            hotspot_markers : dict[HotspotType, str] | None, optional
                ホットスポットの形状を指定する辞書。
                指定の例は {'bio': '^', 'gas': 'o', 'comb': 's'} （三角、丸、四角）など。
            lat_correction : float, optional
                緯度方向の補正係数（デフォルトは1）。
            lon_correction : float, optional
                経度方向の補正係数（デフォルトは1）。
            output_dir : str | None, optional
                プロット画像の保存先パス。
            output_filename : str
                プロット画像の保存ファイル名（拡張子を含む）。デフォルトは'footprint.png'。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
            show_fig : bool
                図の表示を許可するフラグ。デフォルトはTrue。
            satellite_image : ImageFile | None, optional
                使用する衛星画像。指定がない場合はデフォルトの画像が生成されます。
            xy_max : float, optional
                表示範囲の最大値（デフォルトは5000）。
        """
        # 1. 引数のバリデーション
        valid_extensions: list[str] = [".png", ".jpg", ".jpeg", ".pdf", ".svg"]
        _, file_extension = os.path.splitext(output_filename)
        if file_extension.lower() not in valid_extensions:
            quoted_extensions: list[str] = [f'"{ext}"' for ext in valid_extensions]
            self.logger.error(
                f"`output_filename`は有効な拡張子ではありません。プロットを保存するには、次のいずれかを指定してください: {','.join(quoted_extensions)}"
            )
            return

        # 2. フラグチェック
        if not self._got_satellite_image:
            raise ValueError(
                "`get_satellite_image_from_api`または`get_satellite_image_from_local`が実行されていません。"
            )

        # 3. 衛星画像の取得
        if satellite_image is None:
            satellite_image = Image.new("RGB", (2160, 2160), "lightgray")

        self.logger.info("プロットを作成中...")

        # 4. 座標変換のための定数計算（1回だけ）
        meters_per_lat: float = self.EARTH_RADIUS_METER * (
            math.pi / 180
        )  # 緯度1度あたりのメートル
        meters_per_lon: float = meters_per_lat * math.cos(
            math.radians(center_lat)
        )  # 経度1度あたりのメートル

        # 5. フットプリントデータの座標変換（まとめて1回で実行）
        x_deg = (
            np.array(x_list) / meters_per_lon * lon_correction
        )  # 補正係数も同時に適用
        y_deg = (
            np.array(y_list) / meters_per_lat * lat_correction
        )  # 補正係数も同時に適用

        # 6. 中心点からの相対座標を実際の緯度経度に変換
        lons = center_lon + x_deg
        lats = center_lat + y_deg

        # 7. 表示範囲の計算（変更なし）
        x_range: float = xy_max / meters_per_lon
        y_range: float = xy_max / meters_per_lat
        map_boundaries: tuple[float, float, float, float] = (
            center_lon - x_range,  # left_lon
            center_lon + x_range,  # right_lon
            center_lat - y_range,  # bottom_lat
            center_lat + y_range,  # top_lat
        )
        left_lon, right_lon, bottom_lat, top_lat = map_boundaries

        # 8. プロットの作成
        plt.rcParams["axes.edgecolor"] = "None"
        fig: plt.Figure = plt.figure(figsize=(10, 8), dpi=300)
        ax_data: plt.Axes = fig.add_axes([0.05, 0.1, 0.8, 0.8])

        # 9. フットプリントの描画
        # フットプリントの描画とカラーバー用の2つのhexbinを作成
        if c_list is not None:
            ax_data.hexbin(
                lons,
                lats,
                C=c_list,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=0.3,  # 実際のプロット用
                gridsize=100,
                linewidths=0,
                mincnt=100,
                extent=[left_lon, right_lon, bottom_lat, top_lat],
                reduce_C_function=reduce_c_function,
            )

        # カラーバー用の非表示hexbin（alpha=1.0）
        hidden_hexbin = ax_data.hexbin(
            lons,
            lats,
            C=c_list,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=1.0,  # カラーバー用
            gridsize=100,
            linewidths=0,
            mincnt=100,
            extent=[left_lon, right_lon, bottom_lat, top_lat],
            reduce_C_function=reduce_c_function,
            visible=False,  # プロットには表示しない
        )

        # 10. ホットスポットの描画
        spot_handles = []
        if hotspots is not None:
            default_colors: dict[HotspotType, str] = {
                "bio": "blue",
                "gas": "red",
                "comb": "green",
            }

            # デフォルトのマーカー形状を定義
            default_markers: dict[HotspotType, str] = {
                "bio": "o",
                "gas": "o",
                "comb": "o",
            }

            # 座標変換のための定数
            meters_per_lat: float = self.EARTH_RADIUS_METER * (math.pi / 180)
            meters_per_lon: float = meters_per_lat * math.cos(math.radians(center_lat))

            for spot_type, color in (hotspot_colors or default_colors).items():
                spots_lon = []
                spots_lat = []

                # 使用するマーカーを決定
                marker = (hotspot_markers or default_markers).get(spot_type, "o")

                for spot in hotspots:
                    if spot.type == spot_type:
                        # 変換前の緯度経度をログ出力
                        self.logger.debug(
                            f"Before - Type: {spot_type}, Lat: {spot.avg_lat:.6f}, Lon: {spot.avg_lon:.6f}"
                        )

                        # 中心からの相対距離を計算
                        dx: float = (spot.avg_lon - center_lon) * meters_per_lon
                        dy: float = (spot.avg_lat - center_lat) * meters_per_lat

                        # 補正前の相対座標をログ出力
                        self.logger.debug(
                            f"Relative - Type: {spot_type}, X: {dx:.2f}m, Y: {dy:.2f}m"
                        )

                        # 補正を適用
                        corrected_dx: float = dx * lon_correction
                        corrected_dy: float = dy * lat_correction

                        # 補正後の緯度経度を計算
                        adjusted_lon: float = center_lon + corrected_dx / meters_per_lon
                        adjusted_lat: float = center_lat + corrected_dy / meters_per_lat

                        # 変換後の緯度経度をログ出力
                        self.logger.debug(
                            f"After - Type: {spot_type}, Lat: {adjusted_lat:.6f}, Lon: {adjusted_lon:.6f}\n"
                        )

                        if (
                            left_lon <= adjusted_lon <= right_lon
                            and bottom_lat <= adjusted_lat <= top_lat
                        ):
                            spots_lon.append(adjusted_lon)
                            spots_lat.append(adjusted_lat)

                if spots_lon:
                    handle = ax_data.scatter(
                        spots_lon,
                        spots_lat,
                        c=color,
                        marker=marker,  # マーカー形状を指定
                        s=100,
                        alpha=0.7,
                        label=spot_type,  # "bio","gas","comb"
                        edgecolor="black",
                        linewidth=1,
                    )
                    spot_handles.append(handle)

        # 11. 背景画像の設定
        ax_img = ax_data.twiny().twinx()
        ax_img.imshow(
            satellite_image,
            extent=[left_lon, right_lon, bottom_lat, top_lat],
            aspect="equal",
        )

        # 12. 軸の設定
        for ax in [ax_data, ax_img]:
            ax.set_xlim(left_lon, right_lon)
            ax.set_ylim(bottom_lat, top_lat)
            ax.set_xticks([])
            ax.set_yticks([])

        ax_data.set_zorder(2)
        ax_data.patch.set_alpha(0)
        ax_img.set_zorder(1)

        # 13. カラーバーの追加
        if add_cbar:
            cbar_ax: plt.Axes = fig.add_axes([0.88, 0.1, 0.03, 0.8])
            cbar = fig.colorbar(hidden_hexbin, cax=cbar_ax)  # hidden_hexbinを使用
            # cbar_labelが指定されている場合のみラベルを設定
            if cbar_label:
                cbar.set_label(cbar_label, rotation=270, labelpad=cbar_labelpad)

        # 14. ホットスポットの凡例追加
        if add_legend and hotspots and spot_handles:
            ax_data.legend(
                handles=spot_handles,
                loc="upper center",  # 位置を上部中央に
                bbox_to_anchor=(0.55, -0.01),  # 図の下に配置
                ncol=len(spot_handles),  # ハンドルの数に応じて列数を設定
            )

        # 15. 画像の保存
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            output_path: str = os.path.join(output_dir, output_filename)
            self.logger.info("プロットを保存中...")
            try:
                fig.savefig(output_path, bbox_inches="tight")
                self.logger.info(f"プロットが正常に保存されました: {output_path}")
            except Exception as e:
                self.logger.error(f"プロットの保存中にエラーが発生しました: {str(e)}")
        # 16. 画像の表示
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def plot_flux_footprint_with_scale_checker(
        self,
        x_list: list[float],
        y_list: list[float],
        c_list: list[float] | None,
        center_lat: float,
        center_lon: float,
        check_points: list[tuple[float, float, str]] | None = None,
        vmin: float = 0,
        vmax: float = 100,
        add_cbar: bool = True,
        cbar_label: str | None = None,
        cbar_labelpad: int = 20,
        cmap: str = "jet",
        reduce_c_function: callable = np.mean,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dir: str | None = None,
        output_filename: str = "footprint-scale_checker.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: ImageFile | None = None,
        xy_max: float = 5000,
    ) -> None:
        """
        Staticな衛星画像上にフットプリントデータとホットスポットをプロットします。

        このメソッドは、指定されたフットプリントデータとホットスポットを可視化します。
        ホットスポットが指定されない場合は、フットプリントのみ作図します。

        Parameters:
        ------
            x_list : list[float]
                フットプリントのx座標リスト（メートル単位）。
            y_list : list[float]
                フットプリントのy座標リスト（メートル単位）。
            c_list : list[float] | None
                フットプリントの強度を示す値のリスト。
            center_lat : float
                プロットの中心となる緯度。
            center_lon : float
                プロットの中心となる経度。
            check_points : list[tuple[float, float, str]] | None
                確認用の地点リスト。各要素は (緯度, 経度, ラベル) のタプル。
                Noneの場合は中心から500m、1000m、2000m、3000mの位置に仮想的な点を配置。
            cmap : str
                使用するカラーマップの名前。
            vmin : float
                カラーバーの最小値。
            vmax : float
                カラーバーの最大値。
            reduce_c_function : callable, optional
                フットプリントの集約関数（デフォルトはnp.mean）。
            cbar_label : str, optional
                カラーバーのラベル。
            cbar_labelpad : int, optional
                カラーバーラベルのパディング。
            hotspots : list[HotspotData] | None
                ホットスポットデータのリスト。デフォルトはNone。
            hotspot_colors : dict[str, str] | None, optional
                ホットスポットの色を指定する辞書。
            lon_correction : float, optional
                経度方向の補正係数（デフォルトは1）。
            lat_correction : float, optional
                緯度方向の補正係数（デフォルトは1）。
            output_dir : str | None, optional
                プロット画像の保存先パス。
            output_filename : str
                プロット画像の保存ファイル名（拡張子を含む）。デフォルトは'footprint.png'。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
            show_fig : bool
                図の表示を許可するフラグ。デフォルトはTrue。
            satellite_image : ImageFile | None, optional
                使用する衛星画像。指定がない場合はデフォルトの画像が生成されます。
            xy_max : float, optional
                表示範囲の最大値（デフォルトは5000）。
        """
        if check_points is None:
            # デフォルトの確認ポイントを生成（従来の方式）
            default_points = [
                (500, "North", 90),  # 北 500m
                (1000, "East", 0),  # 東 1000m
                (2000, "South", 270),  # 南 2000m
                (3000, "West", 180),  # 西 3000m
            ]

            dummy_hotspots = []
            for distance, direction, angle in default_points:
                rad = math.radians(angle)
                meters_per_lat = self.EARTH_RADIUS_METER * (math.pi / 180)
                meters_per_lon = meters_per_lat * math.cos(math.radians(center_lat))

                dx = distance * math.cos(rad)
                dy = distance * math.sin(rad)

                delta_lon = dx / meters_per_lon
                delta_lat = dy / meters_per_lat

                hotspot = HotspotData(
                    avg_lat=center_lat + delta_lat,
                    avg_lon=center_lon + delta_lon,
                    delta_ch4=0.0,
                    delta_c2h6=0.0,
                    ratio=0.0,
                    type=f"{direction}_{distance}m",
                    section=0,
                    source="scale_check",
                    angle=0,
                    correlation=0,
                )
                dummy_hotspots.append(hotspot)
        else:
            # 指定された緯度経度を使用
            dummy_hotspots = []
            for lat, lon, label in check_points:
                hotspot = HotspotData(
                    avg_lat=lat,
                    avg_lon=lon,
                    delta_ch4=0.0,
                    delta_c2h6=0.0,
                    ratio=0.0,
                    type=label,
                    section=0,
                    source="scale_check",
                    angle=0,
                    correlation=0,
                )
                dummy_hotspots.append(hotspot)

        # カスタムカラーマップの作成
        hotspot_colors = {
            spot.type: plt.cm.tab10(i % 10) for i, spot in enumerate(dummy_hotspots)
        }

        # 既存のメソッドを呼び出してプロット
        self.plot_flux_footprint_with_hotspots(
            x_list=x_list,
            y_list=y_list,
            c_list=c_list,
            center_lat=center_lat,
            center_lon=center_lon,
            vmin=vmin,
            vmax=vmax,
            add_cbar=add_cbar,
            add_legend=True,
            cbar_label=cbar_label,
            cbar_labelpad=cbar_labelpad,
            cmap=cmap,
            reduce_c_function=reduce_c_function,
            hotspots=dummy_hotspots,
            hotspot_colors=hotspot_colors,
            lat_correction=lat_correction,
            lon_correction=lon_correction,
            output_dir=output_dir,
            output_filename=output_filename,
            save_fig=save_fig,
            show_fig=show_fig,
            satellite_image=satellite_image,
            xy_max=xy_max,
        )

    def _combine_all_csv(self, csv_dir_path: str, suffix: str = ".csv") -> pd.DataFrame:
        """
        指定されたディレクトリ内の全CSVファイルを読み込み、処理し、結合します。
        Monthlyシートを結合することを想定しています。

        Parameters:
        ------
            csv_dir_path : str
                CSVファイルが格納されているディレクトリのパス。
            suffix : str, optional
                読み込むファイルの拡張子。デフォルトは".csv"。

        Returns:
        ------
            pandas.DataFrame
                結合および処理済みのデータフレーム。

        Notes:
        ------
            - ディレクトリ内に少なくとも1つのCSVファイルが必要です。
        """
        csv_files = [f for f in os.listdir(csv_dir_path) if f.endswith(suffix)]
        if not csv_files:
            raise ValueError("指定されたディレクトリにCSVファイルが見つかりません。")

        df_array: list[pd.DataFrame] = []
        for csv_file in csv_files:
            file_path: str = os.path.join(csv_dir_path, csv_file)
            df: pd.DataFrame = self._prepare_csv(file_path)
            df_array.append(df)

        # 結合
        df_combined: pd.DataFrame = pd.concat(df_array, join="outer")
        df_combined = df_combined.loc[~df_combined.index.duplicated(), :]

        # 平日と休日の判定に使用するカラムを作成
        df_combined[self._col_weekday] = df_combined.index.map(
            FluxFootprintAnalyzer.is_weekday
        )  # 共通の関数を使用

        return df_combined

    def _prepare_csv(self, file_path: str) -> pd.DataFrame:
        """
        フラックスデータを含むCSVファイルを読み込み、処理します。

        Parameters:
        ------
            file_path : str
                CSVファイルのパス。

        Returns:
        ------
            pandas.DataFrame
                処理済みのデータフレーム。
        """
        # CSVファイルの最初の行を読み込み、ヘッダーを取得するための一時データフレームを作成
        temp: pd.DataFrame = pd.read_csv(file_path, header=None, nrows=1, skiprows=0)
        header = temp.loc[temp.index[0]]

        # 実際のデータを読み込み、必要な行をスキップし、欠損値を指定
        df: pd.DataFrame = pd.read_csv(
            file_path,
            header=None,
            skiprows=2,
            na_values=["#DIV/0!", "#VALUE!", "#REF!", "#N/A", "#NAME?", "NAN"],
            low_memory=False,
        )
        # 取得したヘッダーをデータフレームに設定
        df.columns = header

        # self._required_columnsのカラムが存在するか確認
        missing_columns: list[str] = [
            col for col in self._required_columns if col not in df.columns.tolist()
        ]
        if missing_columns:
            raise ValueError(
                f"必要なカラムが不足しています: {', '.join(missing_columns)}"
            )

        # "Date"カラムをインデックスに設定して返却
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna(subset=["Date"])
        df.set_index("Date", inplace=True)
        return df

    @staticmethod
    def _calculate_footprint_parameters(
        dUstar: float, dU: float, Z_d: float, phi_m: float, phi_c: float, n: float
    ) -> tuple[float, float, float, float, float]:
        """
        フットプリントパラメータを計算します。

        Parameters:
        ------
            dUstar : float
                摩擦速度
            dU : float
                風速
            Z_d : float
                地面修正後の測定高度
            phi_m : float
                運動量の安定度関数
            phi_c : float
                スカラーの安定度関数
            n : float
                安定度パラメータ

        Returns:
        ------
            tuple[float, float, float, float, float]
                m (べき指数),
                U (基準高度での風速),
                r (べき指数の補正項),
                mu (形状パラメータ),
                ksi (フラックス長さスケール)
        """
        KARMAN: float = 0.4  # フォン・カルマン定数
        # パラメータの計算
        m: float = dUstar / KARMAN * phi_m / dU
        U: float = dU / pow(Z_d, m)
        r: float = 2.0 + m - n
        mu: float = (1.0 + m) / r
        kz: float = KARMAN * dUstar * Z_d / phi_c
        k: float = kz / pow(Z_d, n)
        ksi: float = U * pow(Z_d, r) / r / r / k
        return m, U, r, mu, ksi

    @staticmethod
    def _calculate_ground_correction(
        z_m: float,
        wind_speed: np.ndarray,
        friction_velocity: np.ndarray,
        stability_parameter: np.ndarray,
    ) -> float:
        """
        地面修正量を計算します（Pennypacker and Baldocchi, 2016）。

        この関数は、与えられた気象データを使用して地面修正量を計算します。
        計算は以下のステップで行われます：
        1. 変位高さ（d）を計算
        2. 中立条件外のデータを除外
        3. 平均変位高さを計算
        4. 地面修正量を返す

        Parameters:
        ------
            z_m : float
                観測地点の高度
            wind_speed : np.ndarray
                風速データ配列 (WS vector)
            friction_velocity : np.ndarray
                摩擦速度データ配列 (u*)
            stability_parameter : np.ndarray
                安定度パラメータ配列 (z/L)

        Returns:
        ------
            float
                計算された地面修正量
        """
        KARMAN: float = 0.4  # フォン・カルマン定数
        z: float = z_m

        # 変位高さ（d）の計算
        displacement_height = 0.6 * (
            z / (0.6 + 0.1 * (np.exp((KARMAN * wind_speed) / friction_velocity)))
        )

        # 中立条件外のデータをマスク（中立条件：-0.1 < z/L < 0.1）
        neutral_condition_mask = (stability_parameter < -0.1) | (
            0.1 < stability_parameter
        )
        displacement_height[neutral_condition_mask] = np.nan

        # 平均変位高さを計算
        d: float = np.nanmean(displacement_height)

        # 地面修正量を返す
        return z - d

    @staticmethod
    def _calculate_stability_parameters(dzL: float) -> tuple[float, float, float]:
        """
        安定性パラメータを計算します。
        大気安定度に基づいて、運動量とスカラーの安定度関数、および安定度パラメータを計算します。

        Parameters:
        ------
            dzL : float
                無次元高度 (z/L)、ここで z は測定高度、L はモニン・オブコフ長

        Returns:
        ------
            tuple[float, float, float]
                phi_m : float
                    運動量の安定度関数
                phi_c : float
                    スカラーの安定度関数
                n : float
                    安定度パラメータ
        """
        phi_m: float = 0
        phi_c: float = 0
        n: float = 0
        if dzL > 0.0:
            # 安定成層の場合
            dzL = min(dzL, 2.0)
            phi_m = 1.0 + 5.0 * dzL
            phi_c = 1.0 + 5.0 * dzL
            n = 1.0 / (1.0 + 5.0 * dzL)
        else:
            # 不安定成層の場合
            phi_m = pow(1.0 - 16.0 * dzL, -0.25)
            phi_c = pow(1.0 - 16.0 * dzL, -0.50)
            n = (1.0 - 24.0 * dzL) / (1.0 - 16.0 * dzL)
        return phi_m, phi_c, n

    @staticmethod
    def filter_data(
        df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        months: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        指定された期間や月でデータをフィルタリングするメソッド。

        Parameters:
        ------
            df : pd.DataFrame
                フィルタリングするデータフレーム
            start_date : str | None
                フィルタリングの開始日（'YYYY-MM-DD'形式）。デフォルトはNone。
            end_date : str | None
                フィルタリングの終了日（'YYYY-MM-DD'形式）。デフォルトはNone。
            months : list[int] | None
                フィルタリングする月のリスト（例：[1, 2, 12]）。デフォルトはNone。

        Returns:
        ------
            pd.DataFrame
                フィルタリングされたデータフレーム

        Raises:
        ------
            ValueError
                インデックスがDatetimeIndexでない場合、または日付の形式が不正な場合
        """
        # インデックスの検証
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrameのインデックスはDatetimeIndexである必要があります"
            )

        filtered_df: pd.DataFrame = df.copy()

        # 日付形式の検証と変換
        try:
            if start_date is not None:
                start_date = pd.to_datetime(start_date)
            if end_date is not None:
                end_date = pd.to_datetime(end_date)
        except ValueError as e:
            raise ValueError(
                "日付の形式が不正です。'YYYY-MM-DD'形式で指定してください"
            ) from e

        # 期間でフィルタリング
        if start_date is not None or end_date is not None:
            filtered_df = filtered_df.loc[start_date:end_date]

        # 月のバリデーション
        if months is not None:
            if not all(isinstance(m, int) and 1 <= m <= 12 for m in months):
                raise ValueError(
                    "monthsは1から12までの整数のリストである必要があります"
                )
            filtered_df = filtered_df[filtered_df.index.month.isin(months)]

        # フィルタリング後のデータが空でないことを確認
        if filtered_df.empty:
            raise ValueError("フィルタリング後のデータが空になりました")

        return filtered_df

    @staticmethod
    def is_weekday(date: datetime) -> int:
        """
        指定された日付が平日であるかどうかを判定します。

        Parameters:
        ------
            date : datetime
                判定する日付。

        Returns:
        ------
            int
                平日であれば1、そうでなければ0。
        """
        return 1 if not jpholiday.is_holiday(date) and date.weekday() < 5 else 0

    @staticmethod
    def _prepare_plot_data(
        x80: float,
        ksi: float,
        mu: float,
        r: float,
        U: float,
        m: float,
        sigmaV: float,
        flux_value: float,
        plot_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        フットプリントのプロットデータを準備します。

        Parameters:
        ------
            x80 : float
                80%寄与距離
            ksi : float
                フラックス長さスケール
            mu : float
                形状パラメータ
            r : float
                べき指数
            U : float
                風速
            m : float
                風速プロファイルのべき指数
            sigmaV : float
                風速の標準偏差
            flux_value : float
                フラックス値
            plot_count : int
                生成するプロット数

        Returns:
        ------
            tuple[np.ndarray, np.ndarray, np.ndarray]
                x座標、y座標、フラックス値の配列のタプル
        """
        KARMAN: float = 0.4  # フォン・カルマン定数 (pp.210)
        x_lim: int = int(x80)

        """
        各ランで生成するプロット数
        多いほどメモリに付加がかかるため注意
        """
        plot_num: int = plot_count  # 各ランで生成するプロット数

        # x方向の距離配列を生成
        x_list: np.ndarray = np.arange(1, x_lim + 1, dtype="float64")

        # クロスウィンド積分フットプリント関数を計算
        f_list: np.ndarray = (
            ksi**mu * np.exp(-ksi / x_list) / math.gamma(mu) / x_list ** (1.0 + mu)
        )

        # プロット数に基づいてx座標を生成
        num_list: np.ndarray = np.round(f_list * plot_num).astype("int64")
        x1: np.ndarray = np.repeat(x_list, num_list)

        # 風速プロファイルを計算
        Ux: np.ndarray = (
            (math.gamma(mu) / math.gamma(1 / r))
            * ((r**2 * KARMAN) / U) ** (m / r)
            * U
            * x1 ** (m / r)
        )

        # y方向の分散を計算し、正規分布に従ってy座標を生成
        sigma_array: np.ndarray = sigmaV * x1 / Ux
        y1: np.ndarray = np.random.normal(0, sigma_array)

        # フラックス値の配列を生成
        flux1 = np.full_like(x1, flux_value)

        return x1, y1, flux1

    @staticmethod
    def _rotate_coordinates(
        x: np.ndarray, y: np.ndarray, radian: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        座標を指定された角度で回転させます。

        この関数は、与えられたx座標とy座標を、指定された角度（ラジアン）で回転させます。
        回転は原点を中心に反時計回りに行われます。

        Parameters:
        ------
            x : np.ndarray
                回転させるx座標の配列
            y : np.ndarray
                回転させるy座標の配列
            radian : float
                回転角度（ラジアン）

        Returns:
        ------
            tuple[np.ndarray, np.ndarray]
                回転後の(x_, y_)座標の組
        """
        radian1: float = (radian - (np.pi / 2)) * (-1)
        x_: np.ndarray = x * np.cos(radian1) - y * np.sin(radian1)
        y_: np.ndarray = x * np.sin(radian1) + y * np.cos(radian1)
        return x_, y_

    @staticmethod
    def _source_area_KM2001(
        ksi: float,
        mu: float,
        dU: float,
        sigmaV: float,
        Z_d: float,
        max_ratio: float = 0.8,
    ) -> float:
        """
        Kormann and Meixner (2001)のフットプリントモデルに基づいてソースエリアを計算します。

        このメソッドは、与えられたパラメータを使用して、フラックスの寄与距離を計算します。
        計算は反復的に行われ、寄与率が'max_ratio'に達するまで、または最大反復回数に達するまで続けられます。

        Parameters:
        ------
            ksi : float
                フラックス長さスケール
            mu : float
                形状パラメータ
            dU : float
                風速の変化率
            sigmaV : float
                風速の標準偏差
            Z_d : float
                ゼロ面変位高度
            max_ratio : float, optional
                寄与率の最大値。デフォルトは0.8。

        Returns:
        ------
            float
                80%寄与距離（メートル単位）。計算が収束しない場合はnp.nan。

        Notes:
        ------
            - 計算が収束しない場合（最大反復回数に達した場合）、結果はnp.nanとなります。
        """
        if max_ratio > 1:
            raise ValueError("max_ratio は0以上1以下である必要があります。")
        # 変数の初期値
        sum_f: float = 0.0  # 寄与率(0 < sum_f < 1.0)
        x1: float = 0.0
        dF_xd: float = 0.0

        x_d: float = ksi / (
            1.0 + mu
        )  # Eq. 22 (x_d : クロスウィンド積分フラックスフットプリント最大位置)

        dx: float = x_d / 100.0  # 等値線の拡がりの最大距離の100分の1(m)

        # 寄与率が80%に達するまでfを積算
        while sum_f < (max_ratio / 1):
            x1 += dx

            # Equation 21 (dF : クロスウィンド積分フットプリント)
            dF: float = (
                pow(ksi, mu) * math.exp(-ksi / x1) / math.gamma(mu) / pow(x1, 1.0 + mu)
            )

            sum_f += dF  # Footprint を加えていく (0.0 < dF < 1.0)
            dx *= 2.0  # 距離は2倍ずつ増やしていく

            if dx > 1.0:
                dx = 1.0  # 一気に、1 m 以上はインクリメントしない
            if x1 > Z_d * 1000.0:
                break  # ソースエリアが測定高度の1000倍以上となった場合、エラーとして止める

        x_dst: float = x1  # 寄与率が80%に達するまでの積算距離
        f_last: float = (
            pow(ksi, mu)
            * math.exp(-ksi / x_dst)
            / math.gamma(mu)
            / pow(x_dst, 1.0 + mu)
        )  # Page 214 just below the Eq. 21.

        # y方向の最大距離とその位置のxの距離
        dy: float = x_d / 100.0  # 等値線の拡がりの最大距離の100分の1
        y_dst: float = 0.0
        accumulated_y: float = 0.0  # y方向の積算距離を表す変数

        # 最大反復回数を設定
        MAX_ITERATIONS: int = 100000
        for _ in range(MAX_ITERATIONS):
            accumulated_y += dy
            if accumulated_y >= x_dst:
                break

            dF_xd = (
                pow(ksi, mu)
                * math.exp(-ksi / accumulated_y)
                / math.gamma(mu)
                / pow(accumulated_y, 1.0 + mu)
            )  # 式21の直下（214ページ）

            aa: float = math.log(x_dst * dF_xd / f_last / accumulated_y)
            sigma: float = sigmaV * accumulated_y / dU  # 215ページ8行目

            if 2.0 * aa >= 0:
                y_dst_new: float = sigma * math.sqrt(2.0 * aa)
                if y_dst_new <= y_dst:
                    break  # forループを抜ける
                y_dst = y_dst_new

            dy = min(dy * 2.0, 1.0)

        else:
            # ループが正常に終了しなかった場合（最大反復回数に達した場合）
            x_dst = np.nan

        return x_dst
