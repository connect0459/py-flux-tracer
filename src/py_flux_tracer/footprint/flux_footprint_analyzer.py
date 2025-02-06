import io
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from logging import DEBUG, INFO, Logger
from pathlib import Path
from typing import Literal

import jpholiday
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

from ..commons.utilities import setup_logger
from ..mobile.mobile_measurement_analyzer import HotspotData, HotspotType

DefaultColumnsNames = Literal[
    "datetime",
    "wind_direction",
    "wind_speed",
    "friction_velocity",
    "sigma_v",
    "stability",
]


@dataclass
class DefaultColumns:
    """
    デフォルトのカラム名定義

    以下のスネークケースの識別子を使用してカラムを指定します:
    - datetime: 日時
    - wind_direction: 風向 [度]
    - wind_speed: 風速 [m/s]
    - friction_velocity: 摩擦速度 [m/s]
    - sigma_v: 風速の標準偏差 [m/s]
    - stability: 安定度パラメータ [-]
    """

    # スネークケースで内部的な識別子を定義
    DATETIME: DefaultColumnsNames = "datetime"
    WIND_DIRECTION: DefaultColumnsNames = "wind_direction"
    WIND_SPEED: DefaultColumnsNames = "wind_speed"
    FRICTION_VELOCITY: DefaultColumnsNames = "friction_velocity"
    SIGMA_V: DefaultColumnsNames = "sigma_v"
    STABILITY: DefaultColumnsNames = "stability"

    # デフォルトのカラム名マッピング
    @property
    def defalut_mapping(self) -> dict[DefaultColumnsNames, str]:
        """デフォルトのカラム名マッピングを返す"""
        return {
            self.DATETIME: "Date",
            self.WIND_DIRECTION: "Wind direction",
            self.WIND_SPEED: "WS vector",
            self.FRICTION_VELOCITY: "u*",
            self.SIGMA_V: "sigmaV",
            self.STABILITY: "z/L",
        }


class FluxFootprintAnalyzer:
    """
    フラックスフットプリントを解析および可視化するクラス。

    このクラスは、フラックスデータの処理、フットプリントの計算、
    および結果を衛星画像上に可視化するメソッドを提供します。
    座標系と単位に関する重要な注意:
    - すべての距離はメートル単位で計算されます
    - 座標系の原点(0,0)は測定タワーの位置に対応します
    - x軸は東西方向(正が東)
    - y軸は南北方向(正が北)
    - 風向は北から時計回りに測定されたものを使用

    この実装は、Kormann and Meixner (2001) および Takano et al. (2021)に基づいています。
    """

    EARTH_RADIUS_METER: int = 6371000  # 地球の半径(メートル)
    # クラス内部で生成するカラム名
    COL_FFA_IS_WEEKDAY = "ffa_is_weekday"
    COL_FFA_RADIAN = "ffa_radian"
    COL_FFA_WIND_DIR_360 = "ffa_wind_direction_360"

    def __init__(
        self,
        z_m: float,
        na_values: list[str] | None = None,
        column_mapping: Mapping[DefaultColumnsNames, str] | None = None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        衛星画像を用いて FluxFootprintAnalyzer を初期化します。

        Parameters
        ----------
            z_m: float
                測定の高さ(メートル単位)
            na_values: list[str] | None, optional
                NaNと判定する値のパターン。デフォルト値は以下の通り:
                ["#DIV/0!", "#VALUE!", "#REF!", "#N/A", "#NAME?", "NAN", "nan"]
            column_mapping: Mapping[DefaultColumnsNames, str] | None, optional
                入力データのカラム名とデフォルトカラム名のマッピング。
                キーにスネークケースの識別子、値に実際のカラム名を指定します。
                デフォルト値はNoneで、その場合は以下のデフォルトマッピングを使用:
                ```python
                {
                    "datetime": "Date",              # 日時カラム
                    "wind_direction": "Wind direction", # 風向 [度]
                    "wind_speed": "WS vector",       # 風速 [m/s]
                    "friction_velocity": "u*",        # 摩擦速度 [m/s]
                    "sigma_v": "sigmaV",             # 風速の標準偏差 [m/s]
                    "stability": "z/L"               # 安定度パラメータ [-]
                }
                ```
                例えば、入力データのカラム名が異なる場合は以下のように指定:
                ```python
                {
                    "wind_direction": "WD",          # 風向カラム名が"WD"の場合
                    "wind_speed": "WS",              # 風速カラム名が"WS"の場合
                    "friction_velocity": "USTAR"      # 摩擦速度カラム名が"USTAR"の場合
                }
                ```
                指定されなかったキーはデフォルト値が使用されます。
            logger: Logger | None, optional
                使用するロガー。デフォルト値はNoneで、その場合は新しいロガーを生成
            logging_debug: bool, optional
                ログレベルを"DEBUG"に設定するかどうか。デフォルト値はFalseで、その場合はINFO以上のレベルのメッセージを出力

        Returns
        ----------
            None

        Examples
        --------
        >>> # 基本的な初期化(デフォルトのカラム名を使用)
        >>> analyzer = FluxFootprintAnalyzer(z_m=2.5)

        >>> # カスタムのカラム名マッピングを指定
        >>> custom_mapping = {
        ...     "wind_direction": "WD",      # 風向カラムが"WD"
        ...     "wind_speed": "WS",          # 風速カラムが"WS"
        ...     "friction_velocity": "USTAR"  # 摩擦速度カラムが"USTAR"
        ... }
        >>> analyzer = FluxFootprintAnalyzer(
        ...     z_m=3.0,
        ...     column_mapping=custom_mapping
        ... )
        """
        # デフォルトのカラム名を設定
        self._default_cols = DefaultColumns()
        # カラム名マッピングの作成
        self._cols: Mapping[DefaultColumnsNames, str] = self._create_column_mapping(
            column_mapping
        )
        # 必須カラムのリストを作成
        self._required_columns = [
            self._cols[self._default_cols.WIND_DIRECTION],
            self._cols[self._default_cols.WIND_SPEED],
            self._cols[self._default_cols.FRICTION_VELOCITY],
            self._cols[self._default_cols.SIGMA_V],
            self._cols[self._default_cols.STABILITY],
        ]
        self._z_m: float = z_m  # 測定高度
        if na_values is None:
            na_values = [
                "#DIV/0!",
                "#VALUE!",
                "#REF!",
                "#N/A",
                "#NAME?",
                "NAN",
                "nan",
            ]
        self._na_values: list[str] = na_values
        # 状態を管理するフラグ
        self._got_satellite_image: bool = False

        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

    def _create_column_mapping(
        self,
        mapping: Mapping[DefaultColumnsNames, str] | None,
    ) -> Mapping[DefaultColumnsNames, str]:
        """
        カラム名のマッピングを作成します。

        Parameters
        ----------
            mapping : Mapping[DefaultColumnsNames, str] | None
                ユーザー指定のカラム名マッピング。
                キーにスネークケースの識別子(例: "wind_speed")、
                値に実際のカラム名(例: "WS")を指定。

        Returns
        -------
            Mapping[DefaultColumnsNames, str]
                作成されたカラム名マッピング
        """
        # デフォルトのマッピングをコピー
        result = self._default_cols.defalut_mapping.copy()

        if mapping is None:
            return result

        # 指定されたマッピングで上書き
        for snake_case, actual_col in mapping.items():
            if snake_case in self._default_cols.defalut_mapping:
                result[snake_case] = actual_col
            else:
                self.logger.warning(f"Unknown column mapping key: {snake_case}")

        return result

    def check_required_columns(
        self,
        df: pd.DataFrame,
        col_datetime: str | None = None,
    ) -> bool:
        """
        データフレームに必要なカラムが存在するかチェックします。

        Parameters
        ----------
            df: pd.DataFrame
                チェック対象のデータフレーム
            col_datetime: str | None, optional
                日時カラム名。指定された場合はチェック対象から除外されます。デフォルト値はNoneです。

        Returns
        -------
            bool
                すべての必須カラムが存在する場合はTrue、存在しない場合はFalseを返します。

        Examples
        --------
        >>> import pandas as pd
        >>> # カスタムのカラム名を持つデータフレームを作成
        >>> df = pd.DataFrame({
        ...     'TIMESTAMP': ['2024-01-01'],
        ...     'WD': [180.0],
        ...     'WS': [2.5],
        ...     'USTAR': [0.3],
        ...     'SIGMAV': [0.5],
        ...     'ZL': [0.1]
        ... })
        >>> # カスタムのカラム名マッピングを定義
        >>> custom_mapping = {
        ...     "datetime": "TIMESTAMP",      # 日時カラム
        ...     "wind_direction": "WD",       # 風向カラム
        ...     "wind_speed": "WS",           # 風速カラム
        ...     "friction_velocity": "USTAR",  # 摩擦速度カラム
        ...     "sigma_v": "SIGMAV",          # 風速の標準偏差カラム
        ...     "stability": "ZL"             # 安定度パラメータカラム
        ... }
        >>> # カスタムマッピングを使用してアナライザーを初期化
        >>> analyzer = FluxFootprintAnalyzer(
        ...     z_m=2.5,
        ...     column_mapping=custom_mapping
        ... )
        >>> analyzer.check_required_columns(df)
        True
        """
        check_columns = [
            self._cols[self._default_cols.WIND_DIRECTION],
            self._cols[self._default_cols.WIND_SPEED],
            self._cols[self._default_cols.FRICTION_VELOCITY],
            self._cols[self._default_cols.SIGMA_V],
            self._cols[self._default_cols.STABILITY],
        ]

        missing_columns = [col for col in check_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(
                f"Required columns are missing: {missing_columns}."
                f"Available columns: {df.columns.tolist()}"
            )
            return False

        return True

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

        Parameters
        ----------
            df: pd.DataFrame
                分析対象のデータフレーム。フラックスデータを含む。
            col_flux: str
                フラックスデータの列名。計算に使用される。
            plot_count: int, optional
                生成するプロットの数。デフォルト値は10000。
            start_time: str, optional
                フットプリント計算に使用する開始時間。デフォルト値は"10:00"。
            end_time: str, optional
                フットプリント計算に使用する終了時間。デフォルト値は"16:00"。

        Examples
        --------
        >>> import pandas as pd
        >>> # カスタムのカラム名を持つデータフレームを作成
        >>> df = pd.DataFrame({
        ...     'TIMESTAMP': pd.date_range('2024-01-01', periods=24, freq='H'),
        ...     'WD': [180.0] * 24,
        ...     'WS': [2.5] * 24,
        ...     'USTAR': [0.3] * 24,
        ...     'SIGMAV': [0.5] * 24,
        ...     'ZL': [0.1] * 24,
        ...     'FCO2': [-2.0] * 24
        ... })
        >>> # カスタムのカラム名マッピングを定義
        >>> custom_mapping = {
        ...     "datetime": "TIMESTAMP",
        ...     "wind_direction": "WD",
        ...     "wind_speed": "WS",
        ...     "friction_velocity": "USTAR",
        ...     "sigma_v": "SIGMAV",
        ...     "stability": "ZL"
        ... }
        >>> analyzer = FluxFootprintAnalyzer(
        ...     z_m=2.5,
        ...     column_mapping=custom_mapping
        ... )
        >>> x, y, flux = analyzer.calculate_flux_footprint(df, 'FCO2')
        """
        # インデックスがdatetimeであることを確認し、必要に応じて変換
        df_internal: pd.DataFrame = df.copy()
        if not isinstance(df_internal.index, pd.DatetimeIndex):
            datetime_col = self._cols[self._default_cols.DATETIME]
            df_internal.set_index(datetime_col, inplace=True)
            df_internal.index = pd.to_datetime(df_internal.index)

        # 平日/休日の判定を追加
        df_internal[self.COL_FFA_IS_WEEKDAY] = df_internal.index.map(self.is_weekday)

        # 平日データの抽出と時間帯フィルタリング
        data_weekday = df_internal[df_internal[self.COL_FFA_IS_WEEKDAY] == 1].copy()
        data_weekday = data_weekday.between_time(start_time, end_time)
        data_weekday = data_weekday.dropna(subset=[col_flux])

        # 風向の360度系への変換
        wind_dir_col = self._cols[self._default_cols.WIND_DIRECTION]
        directions = [
            wind_direction if wind_direction >= 0 else wind_direction + 360
            for wind_direction in data_weekday[wind_dir_col]
        ]
        data_weekday[self.COL_FFA_WIND_DIR_360] = directions
        data_weekday[self.COL_FFA_RADIAN] = (
            data_weekday[self.COL_FFA_WIND_DIR_360] * np.pi / 180
        )

        # 欠測値の除去
        data_weekday = data_weekday.dropna(subset=[wind_dir_col, col_flux])

        # 数値型への変換
        numeric_columns: set[DefaultColumnsNames] = {
            self._default_cols.FRICTION_VELOCITY,
            self._default_cols.WIND_SPEED,
            self._default_cols.SIGMA_V,
            self._default_cols.STABILITY,
        }
        for col in numeric_columns:
            data_weekday[self._cols[col]] = pd.to_numeric(
                data_weekday[self._cols[col]], errors="coerce"
            )

        # 地面修正量dの計算
        z_m: float = self._z_m
        z_d: float = FluxFootprintAnalyzer._calculate_ground_correction(
            z_m=z_m,
            wind_speed=data_weekday[
                self._cols[self._default_cols.WIND_SPEED]
            ].to_numpy(),
            friction_velocity=data_weekday[
                self._cols[self._default_cols.FRICTION_VELOCITY]
            ].to_numpy(),
            stability_parameter=data_weekday[
                self._cols[self._default_cols.STABILITY]
            ].to_numpy(),
        )

        x_list: list[float] = []
        y_list: list[float] = []
        c_list: list[float] | None = []

        # tqdmを使用してプログレスバーを表示
        for i in tqdm(range(len(data_weekday)), desc="Calculating footprint"):
            d_u_star: float = data_weekday[
                self._cols[self._default_cols.FRICTION_VELOCITY]
            ].iloc[i]
            d_u: float = data_weekday[self._cols[self._default_cols.WIND_SPEED]].iloc[i]
            sigma_v: float = data_weekday[self._cols[self._default_cols.SIGMA_V]].iloc[
                i
            ]
            d_z_l: float = data_weekday[self._cols[self._default_cols.STABILITY]].iloc[
                i
            ]

            if pd.isna(d_u_star) or pd.isna(d_u) or pd.isna(sigma_v) or pd.isna(d_z_l):
                self.logger.warning(f"NaN fields are exist.: i = {i}")
                continue
            elif d_u_star < 5.0 and d_u_star != 0.0 and d_u > 0.1:
                phi_m, phi_c, n = FluxFootprintAnalyzer._calculate_stability_parameters(
                    d_z_l=d_z_l
                )
                m, u, r, mu, ksi = (
                    FluxFootprintAnalyzer._calculate_footprint_parameters(
                        d_u_star=d_u_star,
                        d_u=d_u,
                        z_d=z_d,
                        phi_m=phi_m,
                        phi_c=phi_c,
                        n=n,
                    )
                )

                # 80%ソースエリアの計算
                x80: float = FluxFootprintAnalyzer._source_area_kormann2001(
                    ksi=ksi, mu=mu, d_u=d_u, sigma_v=sigma_v, z_d=z_d, max_ratio=0.8
                )

                if not np.isnan(x80):
                    x1, y1, flux1 = FluxFootprintAnalyzer._prepare_plot_data(
                        x80=x80,
                        ksi=ksi,
                        mu=mu,
                        r=r,
                        u=u,
                        m=m,
                        sigma_v=sigma_v,
                        flux_value=data_weekday[col_flux].iloc[i],
                        plot_count=plot_count,
                    )
                    x1_rotated, y1__rotated = FluxFootprintAnalyzer._rotate_coordinates(
                        x=x1, y=y1, radian=data_weekday[self.COL_FFA_RADIAN].iloc[i]
                    )

                    x_list.extend(x1_rotated)
                    y_list.extend(y1__rotated)
                    c_list.extend(flux1)

        return (
            x_list,
            y_list,
            c_list,
        )

    def combine_all_data(
        self,
        data_source: str | pd.DataFrame,
        col_datetime: str = "Date",
        source_type: Literal["csv", "monthly"] = "csv",
    ) -> pd.DataFrame:
        """
        CSVファイルまたはMonthlyConverterからのデータを統合します。

        Parameters
        ----------
            data_source: str | pd.DataFrame
                CSVディレクトリパスまたはDataFrame形式のデータソース
            col_datetime: str, optional
                datetime型のカラム名。デフォルト値は"Date"
            source_type: Literal["csv", "monthly"], optional
                データソースの種類。"csv"または"monthly"を指定。デフォルト値は"csv"

        Returns
        ----------
            pd.DataFrame
                処理済みのデータフレーム。平日/休日の判定結果と欠損値を除去したデータ

        Examples
        --------
        >>> # CSVファイルからデータを読み込む場合
        >>> analyzer = FluxFootprintAnalyzer(z_m=2.5)
        >>> df = analyzer.combine_all_data(
        ...     data_source="path/to/csv/dir",
        ...     source_type="csv"
        ... )

        >>> # DataFrameから直接データを読み込む場合
        >>> analyzer = FluxFootprintAnalyzer(z_m=2.5)
        >>> input_df = pd.DataFrame({
        ...     "Date": pd.date_range("2024-01-01", periods=3),
        ...     "Wind direction": [180, 270, 90],
        ...     "WS vector": [2.5, 3.0, 1.5]
        ... })
        >>> df = analyzer.combine_all_data(
        ...     data_source=input_df,
        ...     source_type="monthly"
        ... )
        """
        col_weekday: str = self.COL_FFA_IS_WEEKDAY
        if source_type == "csv":
            # 既存のCSV処理ロジック
            if not isinstance(data_source, str):
                raise ValueError(
                    "source_type='csv'の場合、data_sourceはstr型である必要があります"
                )
            return self._combine_all_csv(
                csv_dir_path=data_source, col_datetime=col_datetime
            )
        elif source_type == "monthly":
            # MonthlyConverterからのデータを処理
            if not isinstance(data_source, pd.DataFrame):
                raise ValueError("monthly形式の場合、DataFrameを直接渡す必要があります")

            df: pd.DataFrame = data_source.copy()

            # required_columnsからDateを除外して欠損値チェックを行う
            check_columns: list[str] = [
                col for col in self._required_columns if col != col_datetime
            ]

            # インデックスがdatetimeであることを確認
            if (
                not isinstance(df.index, pd.DatetimeIndex)
                and col_datetime not in df.columns
            ):
                raise ValueError(f"DatetimeIndexまたは{col_datetime}カラムが必要です")

            if col_datetime in df.columns:
                df.set_index(col_datetime, inplace=True)

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
            df[col_weekday] = df.index.map(FluxFootprintAnalyzer.is_weekday)

            # Dateを除外したカラムで欠損値の処理
            df = df.dropna(subset=check_columns)

            # インデックスの重複を除去
            df = df.loc[~df.index.duplicated(), :]

            return df

    def get_satellite_image_from_api(
        self,
        api_key: str,
        center_lat: float,
        center_lon: float,
        output_filepath: str,
        scale: int = 1,
        size: tuple[int, int] = (2160, 2160),
        zoom: int = 13,
    ) -> Image.Image:
        """
        Google Maps Static APIを使用して衛星画像を取得します。

        Parameters
        ----------
            api_key: str
                Google Maps Static APIのキー
            center_lat: float
                中心の緯度
            center_lon: float
                中心の経度
            output_filepath: str
                画像の保存先パス。拡張子は'.png'のみ許可される
            scale: int, optional
                画像の解像度スケール。1または2を指定可能。デフォルトは1
            size: tuple[int, int], optional
                画像サイズ。(幅, 高さ)の形式で指定。デフォルトは(2160, 2160)
            zoom: int, optional
                ズームレベル。0から21の整数を指定可能。デフォルトは13

        Returns
        ----------
            Image.Image
                取得した衛星画像

        Raises
        ----------
            requests.RequestException
                API呼び出しに失敗した場合

        Example
        ----------
        >>> analyzer = FluxFootprintAnalyzer()
        >>> image = analyzer.get_satellite_image_from_api(
        ...     api_key="your_api_key",
        ...     center_lat=35.6895,
        ...     center_lon=139.6917,
        ...     output_filepath="satellite.png",
        ...     zoom=15
        ... )
        """
        # バリデーション
        if not output_filepath.endswith(".png"):
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
            image: Image.Image = Image.open(io.BytesIO(response.content))
            image.save(output_filepath)
            self._got_satellite_image = True
            self.logger.info(f"リモート画像を取得し、保存しました: {output_filepath}")
            return image
        except requests.RequestException as e:
            self.logger.error(f"衛星画像の取得に失敗しました: {e!s}")
            raise e

    def get_satellite_image_from_local(
        self,
        local_image_path: str,
        alpha: float = 1.0,
        grayscale: bool = False,
    ) -> Image.Image:
        """
        ローカルファイルから衛星画像を読み込みます。

        Parameters
        ----------
            local_image_path: str
                ローカル画像のパス
            alpha: float, optional
                画像の透過率。0.0から1.0の範囲で指定します。デフォルト値は1.0です。
            grayscale: bool, optional
                画像を白黒に変換するかどうかを指定します。デフォルト値はFalseです。

        Returns
        ----------
            Image.Image
                読み込んだ衛星画像(透過設定済み)

        Raises
        ----------
            FileNotFoundError
                指定されたパスにファイルが存在しない場合

        Example
        ----------
        >>> analyzer = FluxFootprintAnalyzer()
        >>> image = analyzer.get_satellite_image_from_local(
        ...     local_image_path="satellite.png",
        ...     alpha=0.7,
        ...     grayscale=True
        ... )
        """
        if not os.path.exists(local_image_path):
            raise FileNotFoundError(
                f"指定されたローカル画像が存在しません: {local_image_path}"
            )

        # 画像を読み込む
        image: Image.Image = Image.open(local_image_path)

        # 白黒変換が指定されている場合
        if grayscale:
            image = image.convert("L")  # グレースケールに変換

        # RGBAモードに変換
        image = image.convert("RGBA")

        # 透過率を設定
        data = image.getdata()
        new_data = [(r, g, b, int(255 * alpha)) for r, g, b, a in data]
        image.putdata(new_data)

        self._got_satellite_image = True
        self.logger.info(
            f"ローカル画像を使用しました(透過率: {alpha}, 白黒: {grayscale}): {local_image_path}"
        )
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
        reduce_c_function: Callable = np.mean,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dirpath: str | Path | None = None,
        output_filename: str = "footprint.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: Image.Image | None = None,
        xy_max: float = 5000,
    ) -> None:
        """
        フットプリントデータをプロットします。

        このメソッドは、指定されたフットプリントデータのみを可視化します。

        Parameters
        ----------
            x_list: list[float]
                フットプリントのx座標リスト(メートル単位)
            y_list: list[float]
                フットプリントのy座標リスト(メートル単位)
            c_list: list[float] | None
                フットプリントの強度を示す値のリスト
            center_lat: float
                プロットの中心となる緯度
            center_lon: float
                プロットの中心となる経度
            vmin: float
                カラーバーの最小値
            vmax: float
                カラーバーの最大値
            add_cbar: bool, optional
                カラーバーを追加するかどうか。デフォルト値はTrueです
            add_legend: bool, optional
                凡例を追加するかどうか。デフォルト値はTrueです
            cbar_label: str | None, optional
                カラーバーのラベル。デフォルト値はNoneです
            cbar_labelpad: int, optional
                カラーバーラベルのパディング。デフォルト値は20です
            cmap: str, optional
                使用するカラーマップの名前。デフォルト値は"jet"です
            reduce_c_function: Callable, optional
                フットプリントの集約関数。デフォルト値はnp.meanです
            lon_correction: float, optional
                経度方向の補正係数。デフォルト値は1です
            lat_correction: float, optional
                緯度方向の補正係数。デフォルト値は1です
            output_dirpath: str | Path | None, optional
                プロット画像の保存先パス。デフォルト値はNoneです
            output_filename: str, optional
                プロット画像の保存ファイル名(拡張子を含む)。デフォルト値は'footprint.png'です
            save_fig: bool, optional
                図の保存を許可するフラグ。デフォルト値はTrueです
            show_fig: bool, optional
                図の表示を許可するフラグ。デフォルト値はTrueです
            satellite_image: Image.Image | None, optional
                使用する衛星画像。指定がない場合はデフォルトの画像が生成されます
            xy_max: float, optional
                表示範囲の最大値。デフォルト値は5000です

        Returns
        ----------
            None
                戻り値はありません

        Example
        ----------
        >>> analyzer = FluxFootprintAnalyzer()
        >>> analyzer.plot_flux_footprint(
        ...     x_list=[0, 100, 200],
        ...     y_list=[0, 150, 250],
        ...     c_list=[1.0, 0.8, 0.6],
        ...     center_lat=35.0,
        ...     center_lon=135.0,
        ...     vmin=0.0,
        ...     vmax=1.0,
        ...     cmap="jet",
        ...     xy_max=1000
        ... )
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
            output_dirpath=output_dirpath,
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
        reduce_c_function: Callable = np.mean,
        dpi: float = 300,
        figsize: tuple[float, float] = (8, 8),
        constrained_layout: bool = False,
        hotspots: list[HotspotData] | None = None,
        hotspots_alpha: float = 0.7,
        hotspot_colors: dict[HotspotType, str] | None = None,
        hotspot_labels: dict[HotspotType, str] | None = None,
        hotspot_markers: dict[HotspotType, str] | None = None,
        hotspot_sizes: dict[str, tuple[tuple[float, float], float]] | None = None,
        hotspot_sorting_by_delta_ch4: bool = True,
        legend_alpha: float = 1.0,
        legend_bbox_to_anchor: tuple[float, float] = (0.55, -0.01),
        legend_loc: str = "upper center",
        legend_ncol: int | None = None,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dirpath: str | Path | None = None,
        output_filename: str = "footprint.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: Image.Image | None = None,
        satellite_image_aspect: Literal["auto", "equal"] = "auto",
        xy_max: float = 5000,
    ) -> None:
        """
        静的な衛星画像上にフットプリントデータとホットスポットをプロットします。

        このメソッドは、指定されたフットプリントデータとホットスポットを可視化します。
        ホットスポットが指定されない場合は、フットプリントのみ作図します。

        Parameters
        ----------
            x_list: list[float]
                フットプリントのx座標リスト(メートル単位)
            y_list: list[float]
                フットプリントのy座標リスト(メートル単位)
            c_list: list[float] | None
                フットプリントの強度を示す値のリスト
            center_lat: float
                プロットの中心となる緯度
            center_lon: float
                プロットの中心となる経度
            vmin: float
                カラーバーの最小値
            vmax: float
                カラーバーの最大値
            add_cbar: bool, optional
                カラーバーを追加するかどうか。デフォルトはTrue
            add_legend: bool, optional
                凡例を追加するかどうか。デフォルトはTrue
            cbar_label: str | None, optional
                カラーバーのラベル。デフォルトはNone
            cbar_labelpad: int, optional
                カラーバーラベルのパディング。デフォルトは20
            cmap: str, optional
                使用するカラーマップの名前。デフォルトは"jet"
            reduce_c_function: Callable, optional
                フットプリントの集約関数。デフォルトはnp.mean
            dpi: float, optional
                出力画像の解像度。デフォルトは300
            figsize: tuple[float, float], optional
                出力画像のサイズ。デフォルトは(8, 8)
            constrained_layout: bool, optional
                図のレイアウトを自動調整するかどうか。デフォルトはFalse
            hotspots: list[HotspotData] | None, optional
                ホットスポットデータのリスト。デフォルトはNone
            hotspots_alpha: float, optional
                ホットスポットの透明度。デフォルトは0.7
            hotspot_colors: dict[HotspotType, str] | None, optional
                ホットスポットの色を指定する辞書。デフォルトはNone
            hotspot_labels: dict[HotspotType, str] | None, optional
                ホットスポットの表示ラベルを指定する辞書。デフォルトはNone
            hotspot_markers: dict[HotspotType, str] | None, optional
                ホットスポットの形状を指定する辞書。デフォルトはNone
            hotspot_sizes: dict[str, tuple[tuple[float, float], float]] | None, optional
                ホットスポットのサイズ範囲とマーカーサイズを指定する辞書。デフォルトはNone
            hotspot_sorting_by_delta_ch4: bool, optional
                ホットスポットをΔCH4で昇順ソートするか。デフォルトはTrue
            legend_alpha: float, optional
                凡例の透過率。デフォルトは1.0
            legend_bbox_to_anchor: tuple[float, float], optional
                凡例の位置を微調整するためのアンカーポイント。デフォルトは(0.55, -0.01)
            legend_loc: str, optional
                凡例の基準位置。デフォルトは"upper center"
            legend_ncol: int | None, optional
                凡例の列数。デフォルトはNone
            lat_correction: float, optional
                緯度方向の補正係数。デフォルトは1
            lon_correction: float, optional
                経度方向の補正係数。デフォルトは1
            output_dirpath: str | Path | None, optional
                プロット画像の保存先パス。デフォルトはNone
            output_filename: str, optional
                プロット画像の保存ファイル名。デフォルトは"footprint.png"
            save_fig: bool, optional
                図の保存を許可するフラグ。デフォルトはTrue
            show_fig: bool, optional
                図の表示を許可するフラグ。デフォルトはTrue
            satellite_image: Image.Image | None, optional
                使用する衛星画像。デフォルトはNone
            satellite_image_aspect: Literal["auto", "equal"], optional
                衛星画像のアスペクト比。デフォルトは"auto"
            xy_max: float, optional
                表示範囲の最大値。デフォルトは5000

        Returns
        -------
            None
                戻り値はありません

        Example
        -------
        >>> analyzer = FluxFootprintAnalyzer()
        >>> analyzer.plot_flux_footprint_with_hotspots(
        ...     x_list=[0, 100, 200],
        ...     y_list=[0, 150, 250],
        ...     c_list=[1.0, 0.8, 0.6],
        ...     center_lat=35.0,
        ...     center_lon=135.0,
        ...     vmin=0.0,
        ...     vmax=1.0,
        ...     hotspots=[HotspotData(lat=35.001, lon=135.001, type="gas", delta_ch4=0.5)],
        ...     xy_max=1000
        ... )
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

        # 4. 座標変換のための定数計算(1回だけ)
        meters_per_lat: float = self.EARTH_RADIUS_METER * (
            math.pi / 180
        )  # 緯度1度あたりのメートル
        meters_per_lon: float = meters_per_lat * math.cos(
            math.radians(center_lat)
        )  # 経度1度あたりのメートル

        # 5. フットプリントデータの座標変換(まとめて1回で実行)
        x_deg = (
            np.array(x_list) / meters_per_lon * lon_correction
        )  # 補正係数も同時に適用
        y_deg = (
            np.array(y_list) / meters_per_lat * lat_correction
        )  # 補正係数も同時に適用

        # 6. 中心点からの相対座標を実際の緯度経度に変換
        lons = center_lon + x_deg
        lats = center_lat + y_deg

        # 7. 表示範囲の計算(変更なし)
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

        # 従来のロジック
        fig: Figure = plt.figure(
            figsize=figsize, dpi=dpi, constrained_layout=constrained_layout
        )
        ax_data: Axes = fig.add_axes((0.05, 0.1, 0.8, 0.8))

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
                extent=(left_lon, right_lon, bottom_lat, top_lat),
                reduce_C_function=reduce_c_function,
            )

        # カラーバー用の非表示hexbin(alpha=1.0)
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
            extent=(left_lon, right_lon, bottom_lat, top_lat),
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
                "bio": "^",  # 三角
                "gas": "o",  # 丸
                "comb": "s",  # 四角
            }

            # デフォルトのラベルを定義
            default_labels: dict[HotspotType, str] = {
                "bio": "bio",
                "gas": "gas",
                "comb": "comb",
            }

            # デフォルトのサイズ設定を定義
            default_sizes = {
                "small": ((0, 0.5), 50),
                "medium": ((0.5, 1.0), 100),
                "large": ((1.0, float("inf")), 200),
            }

            # ユーザー指定のサイズ設定を使用(指定がない場合はデフォルト値を使用)
            sizes = hotspot_sizes or default_sizes

            # 座標変換のための定数
            meters_per_lat: float = self.EARTH_RADIUS_METER * (math.pi / 180)
            meters_per_lon: float = meters_per_lat * math.cos(math.radians(center_lat))

            # ΔCH4で昇順ソート
            if hotspot_sorting_by_delta_ch4:
                hotspots = sorted(hotspots, key=lambda x: x.delta_ch4)

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
                    # 使用するラベルを決定
                    label = (hotspot_labels or default_labels).get(spot_type, spot_type)

                    spot_sizes = [
                        sizes[self._get_size_category(spot.delta_ch4, sizes)][1]
                        for spot in hotspots
                        if spot.type == spot_type
                        and left_lon
                        <= (center_lon + (spot.avg_lon - center_lon) * lon_correction)
                        <= right_lon
                        and bottom_lat
                        <= (center_lat + (spot.avg_lat - center_lat) * lat_correction)
                        <= top_lat
                    ]

                    handle = ax_data.scatter(
                        spots_lon,
                        spots_lat,
                        c=color,
                        marker=marker,  # マーカー形状を指定
                        # s=100,
                        s=spot_sizes,  # 修正したサイズリストを使用
                        alpha=hotspots_alpha,
                        label=label,
                        edgecolor="black",
                        linewidth=1,
                    )
                    spot_handles.append(handle)

        # 11. 背景画像の設定
        ax_img = ax_data.twiny().twinx()
        ax_img.imshow(
            satellite_image,
            extent=(left_lon, right_lon, bottom_lat, top_lat),
            aspect=satellite_image_aspect,
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
            cbar_ax: Axes = fig.add_axes((0.88, 0.1, 0.03, 0.8))
            cbar = fig.colorbar(hidden_hexbin, cax=cbar_ax)  # hidden_hexbinを使用
            # cbar_labelが指定されている場合のみラベルを設定
            if cbar_label:
                cbar.set_label(cbar_label, rotation=270, labelpad=cbar_labelpad)

        # 14. ホットスポットの凡例追加
        if add_legend and hotspots and spot_handles:
            # 列数を決定(指定がない場合はホットスポットの数を使用)
            ncol = legend_ncol if legend_ncol is not None else len(spot_handles)
            ax_data.legend(
                handles=spot_handles,
                loc=legend_loc,  # 凡例の基準位置を指定
                bbox_to_anchor=legend_bbox_to_anchor,  # アンカーポイントを指定
                ncol=ncol,  # 列数を指定
                framealpha=legend_alpha,
            )

        # 15. 画像の保存
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            self.logger.info("プロットを保存中...")
            try:
                fig.savefig(output_filepath, bbox_inches="tight")
                self.logger.info(f"プロットが正常に保存されました: {output_filepath}")
            except Exception as e:
                self.logger.error(f"プロットの保存中にエラーが発生しました: {e!s}")
        # 16. 画像の表示
        if show_fig:
            plt.show()
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
        reduce_c_function: Callable = np.mean,
        lat_correction: float = 1,
        lon_correction: float = 1,
        output_dirpath: str | Path | None = None,
        output_filename: str = "footprint-scale_checker.png",
        save_fig: bool = True,
        show_fig: bool = True,
        satellite_image: Image.Image | None = None,
        xy_max: float = 5000,
    ) -> None:
        """
        衛星画像上にフットプリントデータとスケールチェック用のポイントをプロットします。

        Parameters
        ----------
            x_list: list[float]
                フットプリントのx座標リスト(メートル単位)
            y_list: list[float]
                フットプリントのy座標リスト(メートル単位)
            c_list: list[float] | None
                フットプリントの強度を示す値のリスト
            center_lat: float
                プロットの中心となる緯度
            center_lon: float
                プロットの中心となる経度
            check_points: list[tuple[float, float, str]] | None, optional
                確認用の地点リスト。各要素は(緯度、経度、ラベル)のタプル。デフォルト値はNoneで、その場合は中心から500m、1000m、2000m、3000mの位置に仮想的な点を配置
            vmin: float, optional
                カラーバーの最小値。デフォルト値は0
            vmax: float, optional
                カラーバーの最大値。デフォルト値は100
            add_cbar: bool, optional
                カラーバーを追加するかどうか。デフォルト値はTrue
            cbar_label: str | None, optional
                カラーバーのラベル。デフォルト値はNone
            cbar_labelpad: int, optional
                カラーバーラベルのパディング。デフォルト値は20
            cmap: str, optional
                使用するカラーマップの名前。デフォルト値は"jet"
            reduce_c_function: Callable, optional
                フットプリントの集約関数。デフォルト値はnp.mean
            lat_correction: float, optional
                緯度方向の補正係数。デフォルト値は1
            lon_correction: float, optional
                経度方向の補正係数。デフォルト値は1
            output_dirpath: str | Path | None, optional
                プロット画像の保存先パス。デフォルト値はNone
            output_filename: str, optional
                プロット画像の保存ファイル名。デフォルト値は"footprint-scale_checker.png"
            save_fig: bool, optional
                図の保存を許可するフラグ。デフォルト値はTrue
            show_fig: bool, optional
                図の表示を許可するフラグ。デフォルト値はTrue
            satellite_image: Image.Image | None, optional
                使用する衛星画像。デフォルト値はNoneで、その場合はデフォルトの画像が生成されます
            xy_max: float, optional
                表示範囲の最大値。デフォルト値は5000

        Returns
        ----------
            None
                戻り値はありません

        Example
        ----------
        >>> analyzer = FluxFootprintAnalyzer(z_m=2.5)
        >>> analyzer.plot_flux_footprint_with_scale_checker(
        ...     x_list=[0, 100, 200],
        ...     y_list=[0, 150, 250],
        ...     c_list=[1.0, 0.8, 0.6],
        ...     center_lat=35.0,
        ...     center_lon=135.0,
        ...     check_points=[(35.001, 135.001, "Point A")],
        ...     vmin=0.0,
        ...     vmax=1.0,
        ...     cmap="jet",
        ...     xy_max=1000
        ... )
        """
        if check_points is None:
            # デフォルトの確認ポイントを生成(従来の方式)
            default_points = [
                (500, "North", 90),  # 北 500m
                (1000, "East", 0),  # 東 1000m
                (2000, "South", 270),  # 南 2000m
                (3000, "West", 180),  # 西 3000m
            ]

            dummy_hotspots = []
            for distance, _, angle in default_points:
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
                    delta_ratio=0.0,
                    type="scale_check",
                    section=0,
                    timestamp="scale_check",
                    angle=0,
                    correlation=0,
                )
                dummy_hotspots.append(hotspot)
        else:
            # 指定された緯度経度を使用
            dummy_hotspots = []
            for lat, lon, _ in check_points:
                hotspot = HotspotData(
                    avg_lat=lat,
                    avg_lon=lon,
                    delta_ch4=0.0,
                    delta_c2h6=0.0,
                    delta_ratio=0.0,
                    type="scale_check",
                    section=0,
                    timestamp="scale_check",
                    angle=0,
                    correlation=0,
                )
                dummy_hotspots.append(hotspot)

        # カスタムカラーマップの作成
        hotspot_colors = {
            spot.type: f"C{i % 10}" for i, spot in enumerate(dummy_hotspots)
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
            output_dirpath=output_dirpath,
            output_filename=output_filename,
            save_fig=save_fig,
            show_fig=show_fig,
            satellite_image=satellite_image,
            xy_max=xy_max,
        )

    def _combine_all_csv(
        self, csv_dir_path: str, col_datetime: str, suffix: str = ".csv"
    ) -> pd.DataFrame:
        """
        指定されたディレクトリ内の全CSVファイルを読み込み、処理し、結合します。
        Monthlyシートを結合することを想定しています。

        Parameters
        ----------
            csv_dir_path: str
                CSVファイルが格納されているディレクトリのパス。
            col_datetime: str
                datetimeカラムのカラム名。
            suffix: str, optional
                読み込むファイルの拡張子。デフォルトは".csv"。

        Returns
        ----------
            pandas.DataFrame
                結合および処理済みのデータフレーム。

        Notes
        ----------
            - ディレクトリ内に少なくとも1つのCSVファイルが必要です。
        """
        col_weekday: str = self.COL_FFA_IS_WEEKDAY
        csv_files = [f for f in os.listdir(csv_dir_path) if f.endswith(suffix)]
        if not csv_files:
            raise ValueError("指定されたディレクトリにCSVファイルが見つかりません。")

        df_array: list[pd.DataFrame] = []
        for csv_file in csv_files:
            filepath: str = os.path.join(csv_dir_path, csv_file)
            df: pd.DataFrame = self._prepare_csv(
                filepath=filepath, col_datetime=col_datetime
            )
            df_array.append(df)

        # 結合
        df_combined: pd.DataFrame = pd.concat(df_array, join="outer")
        df_combined = df_combined.loc[~df_combined.index.duplicated(), :]

        # 平日と休日の判定に使用するカラムを作成
        df_combined[col_weekday] = df_combined.index.map(
            FluxFootprintAnalyzer.is_weekday
        )  # 共通の関数を使用

        return df_combined

    def _prepare_csv(self, filepath: str, col_datetime: str) -> pd.DataFrame:
        """
        フラックスデータを含むCSVファイルを読み込み、処理します。

        Parameters
        ----------
            filepath: str
                CSVファイルのパス。
            col_datetime: str
                datetimeカラムのカラム名。

        Returns
        ----------
            pandas.DataFrame
                処理済みのデータフレーム。
        """
        # CSVファイルの最初の行を読み込み、ヘッダーを取得するための一時データフレームを作成
        temp: pd.DataFrame = pd.read_csv(filepath, header=None, nrows=1, skiprows=0)
        header = temp.loc[temp.index[0]]

        # 実際のデータを読み込み、必要な行をスキップし、欠損値を指定
        df: pd.DataFrame = pd.read_csv(
            filepath,
            header=None,
            skiprows=2,
            na_values=self._na_values,
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

        # {col_datetime}カラムをインデックスに設定して返却
        df[col_datetime] = pd.to_datetime(df[col_datetime])
        df = df.dropna(subset=[col_datetime])
        df.set_index(col_datetime, inplace=True)
        return df

    @staticmethod
    def _calculate_footprint_parameters(
        d_u_star: float, d_u: float, z_d: float, phi_m: float, phi_c: float, n: float
    ) -> tuple[float, float, float, float, float]:
        """
        フットプリントパラメータを計算します。

        Parameters
        ----------
            d_u_star: float
                摩擦速度
            d_u: float
                風速
            z_d: float
                地面修正後の測定高度
            phi_m: float
                運動量の安定度関数
            phi_c: float
                スカラーの安定度関数
            n: float
                安定度パラメータ

        Returns
        ----------
            tuple[float, float, float, float, float]
                m (べき指数),
                u (基準高度での風速),
                r (べき指数の補正項),
                mu (形状パラメータ),
                ksi (フラックス長さスケール)
        """
        const_karman: float = 0.4  # フォン・カルマン定数
        # パラメータの計算
        m: float = d_u_star / const_karman * phi_m / d_u
        u: float = d_u / pow(z_d, m)
        r: float = 2.0 + m - n
        mu: float = (1.0 + m) / r
        kz: float = const_karman * d_u_star * z_d / phi_c
        k: float = kz / pow(z_d, n)
        ksi: float = u * pow(z_d, r) / r / r / k
        return m, u, r, mu, ksi

    @staticmethod
    def _calculate_ground_correction(
        z_m: float,
        wind_speed: np.ndarray,
        friction_velocity: np.ndarray,
        stability_parameter: np.ndarray,
    ) -> float:
        """
        地面修正量を計算します(Pennypacker and Baldocchi, 2016)。

        この関数は、与えられた気象データを使用して地面修正量を計算します。
        計算は以下のステップで行われます:
        1. 変位高さ(d)を計算
        2. 中立条件外のデータを除外
        3. 平均変位高さを計算
        4. 地面修正量を返す

        Parameters
        ----------
            z_m: float
                観測地点の高度
            wind_speed: np.ndarray
                風速データ配列 (WS vector)
            friction_velocity: np.ndarray
                摩擦速度データ配列 (u*)
            stability_parameter: np.ndarray
                安定度パラメータ配列 (z/L)

        Returns
        ----------
            float
                計算された地面修正量
        """
        const_karman: float = 0.4  # フォン・カルマン定数
        z: float = z_m

        # 変位高さ(d)の計算
        displacement_height = 0.6 * (
            z / (0.6 + 0.1 * (np.exp((const_karman * wind_speed) / friction_velocity)))
        )

        # 中立条件外のデータをマスク(中立条件:-0.1 < z/L < 0.1)
        neutral_condition_mask = (stability_parameter < -0.1) | (
            0.1 < stability_parameter
        )
        displacement_height[neutral_condition_mask] = np.nan

        # 平均変位高さを計算
        d: float = float(np.nanmean(displacement_height))

        # 地面修正量を返す
        return z - d

    @staticmethod
    def _calculate_stability_parameters(d_z_l: float) -> tuple[float, float, float]:
        """
        安定性パラメータを計算します。
        大気安定度に基づいて、運動量とスカラーの安定度関数、および安定度パラメータを計算します。

        Parameters
        ----------
            d_z_l: float
                無次元高度 (z/L)、ここで z は測定高度、L はモニン・オブコフ長

        Returns
        ----------
            tuple[float, float, float]
                phi_m: float
                    運動量の安定度関数
                phi_c: float
                    スカラーの安定度関数
                n: float
                    安定度パラメータ
        """
        phi_m: float = 0
        phi_c: float = 0
        n: float = 0
        if d_z_l > 0.0:
            # 安定成層の場合
            d_z_l = min(d_z_l, 2.0)
            phi_m = 1.0 + 5.0 * d_z_l
            phi_c = 1.0 + 5.0 * d_z_l
            n = 1.0 / (1.0 + 5.0 * d_z_l)
        else:
            # 不安定成層の場合
            phi_m = pow(1.0 - 16.0 * d_z_l, -0.25)
            phi_c = pow(1.0 - 16.0 * d_z_l, -0.50)
            n = (1.0 - 24.0 * d_z_l) / (1.0 - 16.0 * d_z_l)
        return phi_m, phi_c, n

    @staticmethod
    def filter_data(
        df: pd.DataFrame,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
        months: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        指定された期間や月でデータをフィルタリングするメソッド。

        Parameters
        ----------
            df: pd.DataFrame
                フィルタリングするデータフレーム
            start_date: str | datetime | None, optional
                フィルタリングの開始日。'YYYY-MM-DD'形式の文字列またはdatetimeオブジェクト。指定しない場合は最初のデータから開始。
            end_date: str | datetime | None, optional
                フィルタリングの終了日。'YYYY-MM-DD'形式の文字列またはdatetimeオブジェクト。指定しない場合は最後のデータまで。
            months: list[int] | None, optional
                フィルタリングする月のリスト。1から12までの整数を含むリスト。指定しない場合は全ての月を対象。

        Returns
        ----------
            pd.DataFrame
                フィルタリングされたデータフレーム

        Raises
        ----------
            ValueError
                インデックスがDatetimeIndexでない場合、または日付の形式が不正な場合

        Examples
        ----------
        >>> import pandas as pd
        >>> df = pd.DataFrame(index=pd.date_range('2020-01-01', '2020-12-31'))
        >>> # 2020年1月から3月までのデータを抽出
        >>> filtered_df = FluxFootprintAnalyzer.filter_data(
        ...     df,
        ...     start_date='2020-01-01',
        ...     end_date='2020-03-31'
        ... )
        >>> # 冬季(12月、1月、2月)のデータのみを抽出
        >>> winter_df = FluxFootprintAnalyzer.filter_data(
        ...     df,
        ...     months=[12, 1, 2]
        ... )
        """
        # インデックスの検証
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrameのインデックスはDatetimeIndexである必要があります"
            )

        df_internal: pd.DataFrame = df.copy()

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
            df_internal = df_internal.loc[start_date:end_date]

        # 月のバリデーション
        if months is not None:
            if not all(isinstance(m, int) and 1 <= m <= 12 for m in months):
                raise ValueError(
                    "monthsは1から12までの整数のリストである必要があります"
                )
            df_internal = df_internal[
                pd.to_datetime(df_internal.index).month.isin(months)
            ]

        # フィルタリング後のデータが空でないことを確認
        if df_internal.empty:
            raise ValueError("フィルタリング後のデータが空になりました")

        return df_internal

    @staticmethod
    def is_weekday(date: datetime) -> int:
        """
        指定された日付が平日であるかどうかを判定します。

        Parameters
        ----------
            date: datetime
                判定対象の日付。土日祝日以外の日付を平日として判定します。

        Returns
        ----------
            int
                平日の場合は1、土日祝日の場合は0を返します。

        Examples
        --------
        >>> from datetime import datetime
        >>> date = datetime(2024, 1, 1)  # 2024年1月1日(祝日)
        >>> FluxFootprintAnalyzer.is_weekday(date)
        0
        >>> date = datetime(2024, 1, 4)  # 2024年1月4日(木曜)
        >>> FluxFootprintAnalyzer.is_weekday(date)
        1
        """
        return 1 if not jpholiday.is_holiday(date) and date.weekday() < 5 else 0

    @staticmethod
    def _prepare_plot_data(
        x80: float,
        ksi: float,
        mu: float,
        r: float,
        u: float,
        m: float,
        sigma_v: float,
        flux_value: float,
        plot_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        フットプリントのプロットデータを準備します。

        Parameters
        ----------
            x80: float
                80%寄与距離
            ksi: float
                フラックス長さスケール
            mu: float
                形状パラメータ
            r: float
                べき指数
            u: float
                風速
            m: float
                風速プロファイルのべき指数
            sigma_v: float
                風速の標準偏差
            flux_value: float
                フラックス値
            plot_count: int
                生成するプロット数

        Returns
        ----------
            tuple[np.ndarray, np.ndarray, np.ndarray]
                x座標、y座標、フラックス値の配列のタプル
        """
        const_karman: float = 0.4  # フォン・カルマン定数 (pp.210)
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
        u_x: np.ndarray = (
            (math.gamma(mu) / math.gamma(1 / r))
            * ((r**2 * const_karman) / u) ** (m / r)
            * u
            * x1 ** (m / r)
        )

        # y方向の分散を計算し、正規分布に従ってy座標を生成
        sigma_array: np.ndarray = sigma_v * x1 / u_x
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

        この関数は、与えられたx座標とy座標を、指定された角度(ラジアン)で回転させます。
        回転は原点を中心に反時計回りに行われます。

        Parameters
        ----------
            x: np.ndarray
                回転させるx座標の配列
            y: np.ndarray
                回転させるy座標の配列
            radian: float
                回転角度(ラジアン)

        Returns
        ----------
            tuple[np.ndarray, np.ndarray]
                回転後の(x_, y_)座標の組
        """
        radian1: float = (radian - (np.pi / 2)) * (-1)
        x_rotated: np.ndarray = x * np.cos(radian1) - y * np.sin(radian1)
        y_rotated: np.ndarray = x * np.sin(radian1) + y * np.cos(radian1)
        return x_rotated, y_rotated

    @staticmethod
    def _source_area_kormann2001(
        ksi: float,
        mu: float,
        d_u: float,
        sigma_v: float,
        z_d: float,
        max_ratio: float = 0.8,
    ) -> float:
        """
        Kormann and Meixner (2001)のフットプリントモデルに基づいてソースエリアを計算します。

        このメソッドは、与えられたパラメータを使用して、フラックスの寄与距離を計算します。
        計算は反復的に行われ、寄与率が'max_ratio'に達するまで、または最大反復回数に達するまで続けられます。

        Parameters
        ----------
            ksi: float
                フラックス長さスケール
            mu: float
                形状パラメータ
            d_u: float
                風速の変化率
            sigma_v: float
                風速の標準偏差
            z_d: float
                ゼロ面変位高度
            max_ratio: float, optional
                寄与率の最大値。デフォルトは0.8。

        Returns
        ----------
            float
                80%寄与距離(メートル単位)。計算が収束しない場合はnp.nan。

        Notes
        ----------
            - 計算が収束しない場合(最大反復回数に達した場合)、結果はnp.nanとなります。
        """
        if max_ratio > 1:
            raise ValueError("max_ratio は0以上1以下である必要があります。")
        # 変数の初期値
        sum_f: float = 0.0  # 寄与率(0 < sum_f < 1.0)
        x1: float = 0.0
        d_f_xd: float = 0.0

        x_d: float = ksi / (
            1.0 + mu
        )  # Eq. 22 (x_d: クロスウィンド積分フラックスフットプリント最大位置)

        dx: float = x_d / 100.0  # 等値線の拡がりの最大距離の100分の1(m)

        # 寄与率が80%に達するまでfを積算
        while sum_f < (max_ratio / 1):
            x1 += dx

            # Equation 21 (d_f: クロスウィンド積分フットプリント)
            d_f: float = (
                pow(ksi, mu) * math.exp(-ksi / x1) / math.gamma(mu) / pow(x1, 1.0 + mu)
            )

            sum_f += d_f  # Footprint を加えていく (0.0 < d_f < 1.0)
            dx *= 2.0  # 距離は2倍ずつ増やしていく

            if dx > 1.0:
                dx = 1.0  # 一気に、1 m 以上はインクリメントしない
            if x1 > z_d * 1000.0:
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
        max_iterations: int = 100000
        for _ in range(max_iterations):
            accumulated_y += dy
            if accumulated_y >= x_dst:
                break

            d_f_xd = (
                pow(ksi, mu)
                * math.exp(-ksi / accumulated_y)
                / math.gamma(mu)
                / pow(accumulated_y, 1.0 + mu)
            )  # 式21の直下(214ページ)

            aa: float = math.log(x_dst * d_f_xd / f_last / accumulated_y)
            sigma: float = sigma_v * accumulated_y / d_u  # 215ページ8行目

            if 2.0 * aa >= 0:
                y_dst_new: float = sigma * math.sqrt(2.0 * aa)
                if y_dst_new <= y_dst:
                    break  # forループを抜ける
                y_dst = y_dst_new

            dy = min(dy * 2.0, 1.0)

        else:
            # ループが正常に終了しなかった場合(最大反復回数に達した場合)
            x_dst = np.nan

        return x_dst

    @staticmethod
    def _get_size_category(
        value: float, sizes: dict[str, tuple[tuple[float, float], float]]
    ) -> str:
        """
        サイズカテゴリを決定します。

        Parameters
        ----------
            value: float
                サイズを決定するための値。
            sizes: dict[str, tuple[tuple[float, float], float]]
                サイズカテゴリの辞書。キーはカテゴリ名、値は最小値と最大値のタプルおよびサイズ。

        Returns
        ----------
            str
                指定された値に基づいて決定されたサイズカテゴリ。デフォルトは"small"。
        """
        for category, ((min_val, max_val), _) in sizes.items():
            if min_val < value <= max_val:
                return category
        return "small"  # デフォルトのカテゴリ
