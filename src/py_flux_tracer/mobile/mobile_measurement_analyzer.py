import math
import os
from dataclasses import dataclass
from datetime import timedelta
from logging import DEBUG, INFO, Logger
from pathlib import Path
from typing import Literal, get_args

import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
from geopy.distance import geodesic
from matplotlib import gridspec
from tqdm import tqdm

from ..commons.utilities import setup_logger
from .correcting_utils import BiasRemovalConfig, CorrectingUtils, H2OCorrectionConfig

# ホットスポットの種類を表す型エイリアス
HotspotType = Literal["bio", "gas", "comb", "scale_check"]


@dataclass
class HotspotData:
    """
    ホットスポットの情報を保持するデータクラス

    Parameters
    ----------
        timestamp: str
            タイムスタンプ
        angle: float
            中心からの角度
        avg_lat: float
            平均緯度
        avg_lon: float
            平均経度
        correlation: float
            ΔC2H6/ΔCH4相関係数
        delta_ch4: float
            CH4の増加量
        delta_c2h6: float
            C2H6の増加量
        delta_ratio: float
            ΔC2H6/ΔCH4の比率
        section: int
            所属する区画番号
        type: HotspotType
            ホットスポットの種類
    """

    timestamp: str
    angle: float
    avg_lat: float
    avg_lon: float
    correlation: float
    delta_ch4: float
    delta_c2h6: float
    delta_ratio: float
    section: int
    type: HotspotType

    def __post_init__(self):
        """
        __post_init__で各プロパティをバリデーション
        """
        # タイムスタンプが空でないことを確認
        if not self.timestamp.strip():
            raise ValueError(f"'timestamp' must not be empty: {self.timestamp}")

        # 角度は-180~180度の範囲内であることを確認
        if not -180 <= self.angle <= 180:
            raise ValueError(
                f"'angle' must be between -180 and 180 degrees: {self.angle}"
            )

        # 緯度は-90から90度の範囲内であることを確認
        if not -90 <= self.avg_lat <= 90:
            raise ValueError(
                f"'avg_lat' must be between -90 and 90 degrees: {self.avg_lat}"
            )

        # 経度は-180から180度の範囲内であることを確認
        if not -180 <= self.avg_lon <= 180:
            raise ValueError(
                f"'avg_lon' must be between -180 and 180 degrees: {self.avg_lon}"
            )

        # ΔCH4はfloat型であり、0以上を許可
        if not isinstance(self.delta_c2h6, float) or self.delta_ch4 < 0:
            raise ValueError(
                f"'delta_ch4' must be a non-negative value and at least 0: {self.delta_ch4}"
            )

        # ΔC2H6はfloat型のみを許可
        if not isinstance(self.delta_c2h6, float):
            raise ValueError(f"'delta_c2h6' must be a float value: {self.delta_c2h6}")

        # 比率は0または正の値であることを確認
        # if self.delta_ratio < 0:
        #     raise ValueError(
        #         f"'delta_ratio' must be 0 or a positive value: {self.delta_ratio}"
        #     )
        # エラーが出たため暫定的にfloat型の確認のみに変更
        if not isinstance(self.delta_ratio, float):
            raise ValueError(f"'delta_ratio' must be a float value: {self.delta_ratio}")

        # 相関係数は-1から1の範囲内であることを確認
        if not -1 <= self.correlation <= 1 and str(self.correlation) != "nan":
            raise ValueError(
                f"'correlation' must be between -1 and 1: {self.correlation}"
            )

        # セクション番号は0または正の整数であることを確認
        if not isinstance(self.section, int) or self.section < 0:
            raise ValueError(
                f"'section' must be a non-negative integer: {self.section}"
            )


RollingMethod = Literal["quantile", "mean"]


@dataclass
class HotspotParams:
    """ホットスポット解析のパラメータ設定

    Parameters
    ----------
        col_ch4_ppm: str, optional
            CH4濃度のカラム名。デフォルトは"ch4_ppm"。
        col_c2h6_ppb: str, optional
            C2H6濃度のカラム名。デフォルトは"c2h6_ppb"。
        col_h2o_ppm: str, optional
            H2O濃度のカラム名。デフォルトは"h2o_ppm"。
        ch4_ppm_delta_min: float, optional
            CH4濃度変化量の下限閾値。この値未満のデータは除外される。デフォルトは0.05。
        ch4_ppm_delta_max: float, optional
            CH4濃度変化量の上限閾値。この値を超えるデータは除外される。デフォルトは無限大。
        c2h6_ppb_delta_min: float, optional
            C2H6濃度変化量の下限閾値。この値未満のデータは除外される。デフォルトは0.0。
        c2h6_ppb_delta_max: float, optional
            C2H6濃度変化量の上限閾値。この値を超えるデータは除外される。デフォルトは1000.0。
        h2o_ppm_min: float, optional
            H2O濃度の下限閾値。この値未満のデータは除外される。デフォルトは2000。
        rolling_method: RollingMethod, optional
            移動計算の方法。"quantile"は下位パーセンタイル値を使用し、"mean"は移動平均を使用する。デフォルトは"quantile"。
        quantile_value: float, optional
            quantileメソッド使用時の下位パーセンタイル値(0-1)。デフォルトは0.05。

    Examples
    --------
    >>> params = HotspotParams(
    ...     col_ch4_ppm="ch4_ppm",
    ...     ch4_ppm_delta_min=0.1,
    ...     rolling_method="mean"
    ... )
    >>> analyzer = MobileMeasurementAnalyzer(hotspot_params=params)
    """

    col_ch4_ppm: str = "ch4_ppm"
    col_c2h6_ppb: str = "c2h6_ppb"
    col_h2o_ppm: str = "h2o_ppm"
    ch4_ppm_delta_min: float = 0.05
    ch4_ppm_delta_max: float = float("inf")
    c2h6_ppb_delta_min: float = 0.0
    c2h6_ppb_delta_max: float = 1000.0
    h2o_ppm_min: float = 2000
    rolling_method: RollingMethod = "quantile"
    quantile_value: float = 0.05

    def __post_init__(self) -> None:
        """パラメータの検証を行います。

        Raises
        ----------
            ValueError: quantile_value が0以上1以下でない場合
            ValueError: 下限値が上限値を超える場合
        """
        if not 0 <= self.quantile_value <= 1:
            raise ValueError(
                f"'quantile_value' must be between 0 and 1, got {self.quantile_value}"
            )

        if math.isinf(self.ch4_ppm_delta_min) or math.isinf(self.c2h6_ppb_delta_min):
            raise ValueError(
                "Lower threshold values cannot be set to infinity: 'ch4_ppm_delta_min', 'c2h6_ppb_delta_min'."
            )

        if self.ch4_ppm_delta_min > self.ch4_ppm_delta_max:
            raise ValueError(
                "'ch4_ppm_delta_min' must be less than or equal to 'ch4_ppm_delta_max'"
            )

        if self.c2h6_ppb_delta_min > self.c2h6_ppb_delta_max:
            raise ValueError(
                "'c2h6_ppb_delta_min' must be less than or equal to 'c2h6_ppb_delta_max'"
            )


@dataclass
class MobileMeasurementConfig:
    """
    MobileMeasurementAnalyzerのconfigsに与える設定の値を保持するデータクラス

    Parameters
    ----------
        fs: float
            サンプリング周波数(Hz)
        lag: float
            測器の遅れ時間(秒)
        path: Path | str
            ファイルパス
        bias_removal: BiasRemovalConfig | None, optional
            バイアス除去の設定。未指定の場合は補正を実施しません。
        h2o_correction: H2OCorrectionConfig | None, optional
            水蒸気補正の設定。未指定の場合は補正を実施しません。

    Examples
    --------
    >>> # 水蒸気補正の設定
    >>> h2o_config = H2OCorrectionConfig(
    ...     coef_b=0.001,
    ...     coef_c=0.0001,
    ...     h2o_ppm_threshold=2000,
    ...     target_h2o_ppm=10000
    ... )
    >>> # バイアス除去の設定
    >>> bias_config = BiasRemovalConfig(
    ...     quantile_value=0.05,
    ...     base_ch4_ppm=2.0,
    ...     base_c2h6_ppb=0
    ... )
    >>> # 車載観測の設定
    >>> analyzer_config = MobileMeasurementConfig(
    ...     fs=1.0,  # サンプリング周波数 1Hz
    ...     lag=2.0,  # 遅れ時間 2秒
    ...     path="data.csv",
    ...     bias_removal=bias_config,
    ...     h2o_correction=h2o_config
    ... )
    """

    fs: float
    lag: float
    path: Path | str
    bias_removal: BiasRemovalConfig | None = None
    h2o_correction: H2OCorrectionConfig | None = None

    def __post_init__(self) -> None:
        """
        インスタンス生成後に入力値の検証を行います。
        """
        # fsが有効かを確認
        if not isinstance(self.fs, int | float) or self.fs <= 0:
            raise ValueError(
                f"Invalid sampling frequency: {self.fs}. Must be a positive float."
            )
        # lagが0以上のfloatかを確認
        if not isinstance(self.lag, int | float) or self.lag < 0:
            raise ValueError(
                f"Invalid lag value: {self.lag}. Must be a non-negative float."
            )
        # 拡張子の確認
        supported_extensions: list[str] = [".txt", ".csv"]
        extension = Path(self.path).suffix
        if extension not in supported_extensions:
            raise ValueError(
                f"Unsupported file extension: '{extension}'. Supported: {supported_extensions}"
            )
        # ファイルの存在確認
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"'{self.path}'")

    @classmethod
    def validate_and_create(
        cls,
        fs: float,
        lag: float,
        path: Path | str,
        bias_removal: BiasRemovalConfig | None = None,
        h2o_correction: H2OCorrectionConfig | None = None,
    ) -> "MobileMeasurementConfig":
        """
        入力値を検証し、MobileMeasurementConfigインスタンスを生成するファクトリメソッドです。

        Parameters
        ----------
            fs: float
                サンプリング周波数(Hz)。正の値である必要があります。
            lag: float
                遅延時間(秒)。0以上の値である必要があります。
            path: Path | str
                入力ファイルのパス。サポートされている拡張子は.txtと.csvです。
            bias_removal: BiasRemovalConfig | None, optional
                バイアス除去の設定。未指定の場合は補正を実施しません。
            h2o_correction: H2OCorrectionConfig | None, optional
                水蒸気補正の設定。未指定の場合は補正を実施しません。

        Returns
        -------
            MobileMeasurementConfig
                検証された入力設定を持つMobileMeasurementConfigオブジェクト

        Examples
        --------
        >>> config = MobileMeasurementConfig.validate_and_create(
        ...     fs=1.0,
        ...     lag=2.0,
        ...     path="data.csv"
        ... )
        """
        return cls(
            fs=fs,
            lag=lag,
            path=path,
            bias_removal=bias_removal,
            h2o_correction=h2o_correction,
        )


class MobileMeasurementAnalyzer:
    """
    車載濃度観測で得られた測定データを解析するクラス
    """

    EARTH_RADIUS_METERS: float = 6371000  # 地球の半径(メートル)

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        configs: list[MobileMeasurementConfig] | list[tuple[float, float, str | Path]],
        num_sections: int = 4,
        ch4_enhance_threshold: float = 0.1,
        correlation_threshold: float = 0.7,
        hotspot_area_meter: float = 50,
        hotspot_params: HotspotParams | None = None,
        window_minutes: float = 5,
        columns_rename_dict: dict[str, str] | None = None,
        na_values: list[str] | None = None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        測定データ解析クラスを初期化します。

        Parameters
        ----------
            center_lat: float
                中心緯度
            center_lon: float
                中心経度
            configs: list[MobileMeasurementConfig] | list[tuple[float, float, str | Path]]
                入力ファイルのリスト
            num_sections: int, optional
                分割する区画数。デフォルト値は4です。
            ch4_enhance_threshold: float, optional
                CH4増加の閾値(ppm)。デフォルト値は0.1です。
            correlation_threshold: float, optional
                相関係数の閾値。デフォルト値は0.7です。
            hotspot_area_meter: float, optional
                ホットスポットの検出に使用するエリアの半径(メートル)。デフォルト値は50メートルです。
            hotspot_params: HotspotParams | None, optional
                ホットスポット解析のパラメータ設定。未指定の場合はデフォルト設定を使用します。
            window_minutes: float, optional
                移動窓の大きさ(分)。デフォルト値は5分です。
            columns_rename_dict: dict[str, str] | None, optional
                元のデータファイルのヘッダーを汎用的な単語に変換するための辞書型データ。未指定の場合は以下のデフォルト値を使用します:
                ```py
                {
                    "Time Stamp": "timestamp",
                    "CH4 (ppm)": "ch4_ppm",
                    "C2H6 (ppb)": "c2h6_ppb",
                    "H2O (ppm)": "h2o_ppm",
                    "Latitude": "latitude",
                    "Longitude": "longitude",
                }
                ```
            na_values: list[str] | None, optional
                NaNと判定する値のパターン。未指定の場合は["No Data", "nan"]を使用します。
            logger: Logger | None, optional
                使用するロガー。未指定の場合は新しいロガーを作成します。
            logging_debug: bool, optional
                ログレベルを"DEBUG"に設定するかどうか。デフォルト値はFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer(
        ...     center_lat=35.6895,
        ...     center_lon=139.6917,
        ...     configs=[
        ...         MobileMeasurementConfig(fs=1.0, lag=0.0, path="data1.csv"),
        ...         MobileMeasurementConfig(fs=1.0, lag=0.0, path="data2.csv")
        ...     ],
        ...     num_sections=6,
        ...     ch4_enhance_threshold=0.2
        ... )
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)
        # デフォルト値を使用
        if columns_rename_dict is None:
            columns_rename_dict = {
                "Time Stamp": "timestamp",
                "CH4 (ppm)": "ch4_ppm",
                "C2H6 (ppb)": "c2h6_ppb",
                "H2O (ppm)": "h2o_ppm",
                "Latitude": "latitude",
                "Longitude": "longitude",
            }
        if na_values is None:
            na_values = ["No Data", "nan"]
        # プライベートなプロパティ
        self._center_lat: float = center_lat
        self._center_lon: float = center_lon
        self._ch4_enhance_threshold: float = ch4_enhance_threshold
        self._correlation_threshold: float = correlation_threshold
        self._hotspot_area_meter: float = hotspot_area_meter
        self._column_mapping: dict[str, str] = columns_rename_dict
        self._na_values: list[str] = na_values
        self._hotspot_params = hotspot_params or HotspotParams()
        self._num_sections: int = num_sections
        # セクションの範囲
        section_size: float = 360 / num_sections
        self._section_size: float = section_size
        self._sections = MobileMeasurementAnalyzer._initialize_sections(
            num_sections, section_size
        )
        # window_sizeをデータポイント数に変換(分→秒→データポイント数)
        self._window_size: int = MobileMeasurementAnalyzer._calculate_window_size(
            window_minutes
        )
        # 入力設定の標準化
        normalized_input_configs: list[MobileMeasurementConfig] = (
            MobileMeasurementAnalyzer._normalize_configs(configs)
        )
        self._configs: list[MobileMeasurementConfig] = normalized_input_configs
        # 複数ファイルのデータを読み込み結合
        self.df: pd.DataFrame = self._load_all_combined_data(normalized_input_configs)

    @property
    def hotspot_params(self) -> HotspotParams:
        """ホットスポット解析のパラメータ設定を取得"""
        return self._hotspot_params

    @hotspot_params.setter
    def hotspot_params(self, params: HotspotParams) -> None:
        """ホットスポット解析のパラメータ設定を更新"""
        self._hotspot_params = params

    def analyze_delta_ch4_stats(self, hotspots: list[HotspotData]) -> None:
        """
        各タイプのホットスポットについてΔCH4の統計情報を計算し、結果を表示します。

        Parameters
        ----------
            hotspots: list[HotspotData]
                分析対象のホットスポットリスト
        """
        # タイプごとにホットスポットを分類
        hotspots_by_type: dict[HotspotType, list[HotspotData]] = {
            "bio": [h for h in hotspots if h.type == "bio"],
            "gas": [h for h in hotspots if h.type == "gas"],
            "comb": [h for h in hotspots if h.type == "comb"],
        }

        # 統計情報を計算し、表示
        for spot_type, spots in hotspots_by_type.items():
            if spots:
                delta_ch4_values = [spot.delta_ch4 for spot in spots]
                max_value = max(delta_ch4_values)
                mean_value = sum(delta_ch4_values) / len(delta_ch4_values)
                median_value = sorted(delta_ch4_values)[len(delta_ch4_values) // 2]
                print(f"{spot_type}タイプのホットスポットの統計情報:")
                print(f"  最大値: {max_value}")
                print(f"  平均値: {mean_value}")
                print(f"  中央値: {median_value}")
            else:
                print(f"{spot_type}タイプのホットスポットは存在しません。")

    def analyze_hotspots(
        self,
        duplicate_check_mode: Literal["none", "time_window", "time_all"] = "none",
        min_time_threshold_seconds: float = 300,
        max_time_threshold_hours: float = 12,
    ) -> list[HotspotData]:
        """
        ホットスポットを検出して分析します。

        Parameters
        ----------
            duplicate_check_mode: Literal["none", "time_window", "time_all"], optional
                重複チェックのモード。デフォルトは"none"。
                    - "none": 重複チェックを行わない
                    - "time_window": 指定された時間窓内の重複のみを除外
                    - "time_all": すべての時間範囲で重複チェックを行う
            min_time_threshold_seconds: float, optional
                重複とみなす最小時間の閾値。デフォルトは300秒。
            max_time_threshold_hours: float, optional
                重複チェックを一時的に無視する最大時間の閾値。デフォルトは12時間。

        Returns
        ----------
            list[HotspotData]
                検出されたホットスポットのリスト

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> # 重複チェックなしでホットスポットを検出
        >>> hotspots = analyzer.analyze_hotspots()
        >>>
        >>> # 時間窓内の重複を除外してホットスポットを検出
        >>> hotspots = analyzer.analyze_hotspots(
        ...     duplicate_check_mode="time_window",
        ...     min_time_threshold_seconds=600,
        ...     max_time_threshold_hours=24
        ... )
        """
        all_hotspots: list[HotspotData] = []
        params: HotspotParams = self._hotspot_params

        # 各データソースに対して解析を実行
        # パラメータの計算
        df_processed: pd.DataFrame = (
            MobileMeasurementAnalyzer._calculate_hotspots_parameters(
                df=self.df,
                window_size=self._window_size,
                col_ch4_ppm=params.col_ch4_ppm,
                col_c2h6_ppb=params.col_c2h6_ppb,
                col_h2o_ppm=params.col_h2o_ppm,
                ch4_ppm_delta_min=params.ch4_ppm_delta_min,
                ch4_ppm_delta_max=params.ch4_ppm_delta_max,
                c2h6_ppb_delta_min=params.c2h6_ppb_delta_min,
                c2h6_ppb_delta_max=params.c2h6_ppb_delta_max,
                h2o_ppm_threshold=params.h2o_ppm_min,
                rolling_method=params.rolling_method,
                quantile_value=params.quantile_value,
            )
        )

        # ホットスポットの検出
        hotspots: list[HotspotData] = self._detect_hotspots(
            df=df_processed,
            ch4_enhance_threshold=self._ch4_enhance_threshold,
        )
        all_hotspots.extend(hotspots)

        # 重複チェックモードに応じて処理
        if duplicate_check_mode != "none":
            unique_hotspots = MobileMeasurementAnalyzer.remove_hotspots_duplicates(
                all_hotspots,
                check_time_all=(duplicate_check_mode == "time_all"),
                min_time_threshold_seconds=min_time_threshold_seconds,
                max_time_threshold_hours=max_time_threshold_hours,
                hotspot_area_meter=self._hotspot_area_meter,
            )
            self.logger.info(
                f"重複除外: {len(all_hotspots)} → {len(unique_hotspots)} ホットスポット"
            )
            return unique_hotspots

        return all_hotspots

    def calculate_measurement_stats(
        self,
        col_latitude: str = "latitude",
        col_longitude: str = "longitude",
        print_summary_individual: bool = True,
        print_summary_total: bool = True,
    ) -> tuple[float, timedelta]:
        """
        各ファイルの測定時間と走行距離を計算し、合計を返します。

        Parameters
        ----------
            col_latitude: str, optional
                緯度情報が格納されているカラム名。デフォルト値は"latitude"です。
            col_longitude: str, optional
                経度情報が格納されているカラム名。デフォルト値は"longitude"です。
            print_summary_individual: bool, optional
                個別ファイルの統計を表示するかどうか。デフォルト値はTrueです。
            print_summary_total: bool, optional
                合計統計を表示するかどうか。デフォルト値はTrueです。

        Returns
        ----------
            tuple[float, timedelta]
                総距離(km)と総時間のタプル

        Examples
        ----------
        >>> analyzer = MobileMeasurementAnalyzer(config_list)
        >>> total_distance, total_time = analyzer.calculate_measurement_stats()
        >>> print(f"総距離: {total_distance:.2f}km")
        >>> print(f"総時間: {total_time}")
        """
        total_distance: float = 0.0
        total_time: timedelta = timedelta()
        individual_stats: list[dict] = []  # 個別の統計情報を保存するリスト

        # プログレスバーを表示しながら計算
        for config in tqdm(self._configs, desc="Calculating", unit="file"):
            df, source_name = self._load_data(config=config)
            # 時間の計算
            time_spent = df.index[-1] - df.index[0]

            # 距離の計算
            distance_km = 0.0
            for i in range(len(df) - 1):
                lat1, lon1 = df.iloc[i][[col_latitude, col_longitude]]
                lat2, lon2 = df.iloc[i + 1][[col_latitude, col_longitude]]
                distance_km += (
                    MobileMeasurementAnalyzer._calculate_distance(
                        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
                    )
                    / 1000
                )

            # 合計に加算
            total_distance += distance_km
            total_time += time_spent

            # 統計情報を保存
            if print_summary_individual:
                average_speed = distance_km / (time_spent.total_seconds() / 3600)
                individual_stats.append(
                    {
                        "source": source_name,
                        "distance": distance_km,
                        "time": time_spent,
                        "speed": average_speed,
                    }
                )

        # 計算完了後に統計情報を表示
        if print_summary_individual:
            self.logger.info("=== Individual Stats ===")
            for stat in individual_stats:
                print(f"File        : {stat['source']}")
                print(f"  Distance  : {stat['distance']:.2f} km")
                print(f"  Time      : {stat['time']}")
                print(f"  Avg. Speed: {stat['speed']:.1f} km/h\n")

        # 合計を表示
        if print_summary_total:
            average_speed_total: float = total_distance / (
                total_time.total_seconds() / 3600
            )
            self.logger.info("=== Total Stats ===")
            print(f"  Distance  : {total_distance:.2f} km")
            print(f"  Time      : {total_time}")
            print(f"  Avg. Speed: {average_speed_total:.1f} km/h\n")

        return total_distance, total_time

    def create_hotspots_map(
        self,
        hotspots: list[HotspotData],
        output_dirpath: str | Path | None = None,
        output_filename: str = "hotspots_map.html",
        center_marker_color: str = "green",
        center_marker_label: str = "Center",
        plot_center_marker: bool = True,
        radius_meters: float = 3000,
        save_fig: bool = True,
    ) -> None:
        """
        ホットスポットの分布を地図上にプロットして保存します。

        Parameters
        ----------
            hotspots: list[HotspotData]
                プロットするホットスポットのリスト
            output_dirpath: str | Path | None, optional
                保存先のディレクトリパス。未指定の場合はカレントディレクトリに保存されます。
            output_filename: str, optional
                保存するファイル名。デフォルト値は"hotspots_map.html"です。
            center_marker_color: str, optional
                中心を示すマーカーの色。デフォルト値は"green"です。
            center_marker_label: str, optional
                中心を示すマーカーのラベル。デフォルト値は"Center"です。
            plot_center_marker: bool, optional
                中心を示すマーカーを表示するかどうか。デフォルト値はTrueです。
            radius_meters: float, optional
                区画分けを示す線の長さ(メートル)。デフォルト値は3000です。
            save_fig: bool, optional
                図を保存するかどうか。デフォルト値はTrueです。

        Returns
        -------
            None

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer(center_lat=35.6895, center_lon=139.6917, configs=[])
        >>> hotspots = [HotspotData(...)]  # ホットスポットのリスト
        >>> analyzer.create_hotspots_map(
        ...     hotspots=hotspots,
        ...     output_dirpath="results",
        ...     radius_meters=5000
        ... )
        """
        # 地図の作成
        m = folium.Map(
            location=[self._center_lat, self._center_lon],
            zoom_start=15,
            tiles="OpenStreetMap",
        )

        # ホットスポットの種類ごとに異なる色でプロット
        for spot in hotspots:
            # NaN値チェックを追加
            if math.isnan(spot.avg_lat) or math.isnan(spot.avg_lon):
                continue

            # default type
            color = "black"
            # タイプに応じて色を設定
            if spot.type == "comb":
                color = "green"
            elif spot.type == "gas":
                color = "red"
            elif spot.type == "bio":
                color = "blue"

            # CSSのgrid layoutを使用してHTMLタグを含むテキストをフォーマット
            popup_html = f"""
            <div style='font-family: Arial; font-size: 12px; display: grid; grid-template-columns: auto auto auto; gap: 5px;'>
                <b>Date</b> <span>:</span> <span>{spot.timestamp}</span>
                <b>Lat</b> <span>:</span> <span>{spot.avg_lat:.3f}</span>
                <b>Lon</b> <span>:</span> <span>{spot.avg_lon:.3f}</span>
                <b>ΔCH<sub>4</sub></b> <span>:</span> <span>{spot.delta_ch4:.3f}</span>
                <b>ΔC<sub>2</sub>H<sub>6</sub></b> <span>:</span> <span>{spot.delta_c2h6:.3f}</span>
                <b>Ratio</b> <span>:</span> <span>{spot.delta_ratio:.3f}</span>
                <b>Type</b> <span>:</span> <span>{spot.type}</span>
                <b>Section</b> <span>:</span> <span>{spot.section}</span>
            </div>
            """

            # ポップアップのサイズを指定
            popup = folium.Popup(
                folium.Html(popup_html, script=True),
                max_width=200,  # 最大幅(ピクセル)
            )

            folium.CircleMarker(
                location=[spot.avg_lat, spot.avg_lon],
                radius=8,
                color=color,
                fill=True,
                popup=popup,
            ).add_to(m)

        # 中心点のマーカー
        if plot_center_marker:
            folium.Marker(
                [self._center_lat, self._center_lon],
                popup=center_marker_label,
                icon=folium.Icon(color=center_marker_color, icon="info-sign"),
            ).add_to(m)

        # 区画の境界線を描画
        for section in range(self._num_sections):
            start_angle = math.radians(-180 + section * self._section_size)

            const_r = self.EARTH_RADIUS_METERS

            # 境界線の座標を計算
            lat1 = self._center_lat
            lon1 = self._center_lon
            lat2 = math.degrees(
                math.asin(
                    math.sin(math.radians(lat1)) * math.cos(radius_meters / const_r)
                    + math.cos(math.radians(lat1))
                    * math.sin(radius_meters / const_r)
                    * math.cos(start_angle)
                )
            )
            lon2 = self._center_lon + math.degrees(
                math.atan2(
                    math.sin(start_angle)
                    * math.sin(radius_meters / const_r)
                    * math.cos(math.radians(lat1)),
                    math.cos(radius_meters / const_r)
                    - math.sin(math.radians(lat1)) * math.sin(math.radians(lat2)),
                )
            )

            # 境界線を描画
            folium.PolyLine(
                locations=[[lat1, lon1], [lat2, lon2]],
                color="black",
                weight=1,
                opacity=0.5,
            ).add_to(m)

        # 地図を保存
        if save_fig and output_dirpath is None:
            raise ValueError(
                "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
            )
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            m.save(str(output_filepath))
            self.logger.info(f"地図を保存しました: {output_filepath}")

    def export_hotspots_to_csv(
        self,
        hotspots: list[HotspotData],
        output_dirpath: str | Path | None = None,
        output_filename: str = "hotspots.csv",
    ) -> None:
        """
        ホットスポットの情報をCSVファイルに出力します。

        Parameters
        ----------
            hotspots: list[HotspotData]
                出力するホットスポットのリスト
            output_dirpath: str | Path | None, optional
                出力先ディレクトリのパス。未指定の場合はValueErrorが発生します。
            output_filename: str, optional
                出力ファイル名。デフォルト値は"hotspots.csv"です。

        Returns
        -------
            None
                戻り値はありません。

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> hotspots = analyzer.analyze_hotspots()
        >>> analyzer.export_hotspots_to_csv(
        ...     hotspots=hotspots,
        ...     output_dirpath="output",
        ...     output_filename="hotspots_20240101.csv"
        ... )
        """
        # 日時の昇順でソート
        sorted_hotspots = sorted(hotspots, key=lambda x: x.timestamp)

        # 出力用のデータを作成
        records = []
        for spot in sorted_hotspots:
            record = {
                "timestamp": spot.timestamp,
                "type": spot.type,
                "delta_ch4": spot.delta_ch4,
                "delta_c2h6": spot.delta_c2h6,
                "delta_ratio": spot.delta_ratio,
                "correlation": spot.correlation,
                "angle": spot.angle,
                "section": spot.section,
                "latitude": spot.avg_lat,
                "longitude": spot.avg_lon,
            }
            records.append(record)

        # DataFrameに変換してCSVに出力
        if output_dirpath is None:
            raise ValueError(
                "output_dirpath が指定されていません。有効なディレクトリパスを指定してください。"
            )
        os.makedirs(output_dirpath, exist_ok=True)
        output_filepath: str = os.path.join(output_dirpath, output_filename)
        df: pd.DataFrame = pd.DataFrame(records)
        df.to_csv(output_filepath, index=False)
        self.logger.info(
            f"ホットスポット情報をCSVファイルに出力しました: {output_filepath}"
        )

    @staticmethod
    def extract_source_name_from_path(path: str | Path) -> str:
        """
        ファイルパスからソース名(拡張子なしのファイル名)を抽出します。

        Parameters
        ----------
            path: str | Path
                ソース名を抽出するファイルパス
                例: "/path/to/Pico100121_241017_092120+.txt"

        Returns
        ----------
            str
                抽出されたソース名
                例: "Pico100121_241017_092120+"

        Examples:
        ----------
            >>> path = "/path/to/data/Pico100121_241017_092120+.txt"
            >>> MobileMeasurementAnalyzer.extract_source_from_path(path)
            'Pico100121_241017_092120+'
        """
        # Pathオブジェクトに変換
        path_obj: Path = Path(path)
        # stem属性で拡張子なしのファイル名を取得
        source_name: str = path_obj.stem
        return source_name

    def get_preprocessed_data(
        self,
    ) -> pd.DataFrame:
        """
        データ前処理を行い、CH4とC2H6の相関解析に必要な形式に整えます。
        コンストラクタで読み込んだすべてのデータを前処理し、結合したDataFrameを返します。

        Returns
        ----------
            pd.DataFrame
                前処理済みの結合されたDataFrame
        """
        return self.df.copy()

    def get_section_size(self) -> float:
        """
        セクションのサイズを取得するメソッド。
        このメソッドは、解析対象のデータを区画に分割する際の
        各区画の角度範囲を示すサイズを返します。

        Returns
        ----------
            float
                1セクションのサイズ(度単位)
        """
        return self._section_size

    def plot_ch4_delta_histogram(
        self,
        hotspots: list[HotspotData],
        output_dirpath: str | Path | None,
        output_filename: str = "ch4_delta_histogram.png",
        dpi: float | None = 350,
        figsize: tuple[float, float] = (8, 6),
        fontsize: float = 20,
        hotspot_colors: dict[HotspotType, str] | None = None,
        xlabel: str = "Δ$\\mathregular{CH_{4}}$ (ppm)",
        ylabel: str = "Frequency",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
        yscale_log: bool = True,
        print_bins_analysis: bool = False,
    ) -> None:
        """
        CH4の増加量(ΔCH4)の積み上げヒストグラムをプロットします。

        Parameters
        ----------
            hotspots: list[HotspotData]
                プロットするホットスポットのリスト
            output_dirpath: str | Path | None
                保存先のディレクトリパス
            output_filename: str, optional
                保存するファイル名。デフォルト値は"ch4_delta_histogram.png"です。
            dpi: float | None, optional
                解像度。デフォルト値は350です。
            figsize: tuple[float, float], optional
                図のサイズ。デフォルト値は(8, 6)です。
            fontsize: float, optional
                フォントサイズ。デフォルト値は20です。
            hotspot_colors: dict[HotspotType, str] | None, optional
                ホットスポットの色を定義する辞書。未指定の場合は以下のデフォルト値を使用します:
                {
                    "bio": "blue",
                    "gas": "red",
                    "comb": "green",
                }
            xlabel: str, optional
                x軸のラベル。デフォルト値は"Δ$\\mathregular{CH_{4}}$ (ppm)"です。
            ylabel: str, optional
                y軸のラベル。デフォルト値は"Frequency"です。
            xlim: tuple[float, float] | None, optional
                x軸の範囲。未指定の場合は自動設定されます。
            ylim: tuple[float, float] | None, optional
                y軸の範囲。未指定の場合は自動設定されます。
            save_fig: bool, optional
                図の保存を許可するフラグ。デフォルト値はTrueです。
            show_fig: bool, optional
                図の表示を許可するフラグ。デフォルト値はTrueです。
            yscale_log: bool, optional
                y軸をlogスケールにするかどうか。デフォルト値はTrueです。
            print_bins_analysis: bool, optional
                ビンごとの内訳を表示するかどうか。デフォルト値はFalseです。

        Returns
        -------
            None

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer(...)
        >>> hotspots = analyzer.detect_hotspots()
        >>> analyzer.plot_ch4_delta_histogram(
        ...     hotspots=hotspots,
        ...     output_dirpath="results",
        ...     xlim=(0, 5),
        ...     ylim=(0, 100)
        ... )
        """
        if hotspot_colors is None:
            hotspot_colors = {
                "bio": "blue",
                "gas": "red",
                "comb": "green",
            }
        plt.rcParams["font.size"] = fontsize
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # ホットスポットからデータを抽出
        all_ch4_deltas = []
        all_types = []
        for spot in hotspots:
            all_ch4_deltas.append(spot.delta_ch4)
            all_types.append(spot.type)

        # データをNumPy配列に変換
        all_ch4_deltas = np.array(all_ch4_deltas)
        all_types = np.array(all_types)

        # 0.1刻みのビンを作成
        if xlim is not None:
            bins = np.arange(xlim[0], xlim[1] + 0.1, 0.1)
        else:
            max_val = np.ceil(np.max(all_ch4_deltas) * 10) / 10
            bins = np.arange(0, max_val + 0.1, 0.1)

        # タイプごとのヒストグラムデータを計算
        hist_data = {}
        # HotspotTypeのリテラル値を使用してイテレーション
        for type_name in get_args(HotspotType):  # typing.get_argsをインポート
            mask = all_types == type_name
            if np.any(mask):
                counts, _ = np.histogram(all_ch4_deltas[mask], bins=bins)
                hist_data[type_name] = counts

        # ビンごとの内訳を表示
        if print_bins_analysis:
            self.logger.info("各ビンの内訳:")
            print(f"{'Bin Range':15} {'bio':>8} {'gas':>8} {'comb':>8} {'Total':>8}")
            print("-" * 50)

            for i in range(len(bins) - 1):
                bin_start = bins[i]
                bin_end = bins[i + 1]
                bio_count = hist_data.get("bio", np.zeros(len(bins) - 1))[i]
                gas_count = hist_data.get("gas", np.zeros(len(bins) - 1))[i]
                comb_count = hist_data.get("comb", np.zeros(len(bins) - 1))[i]
                total = bio_count + gas_count + comb_count

                if total > 0:  # 合計が0のビンは表示しない
                    print(
                        f"{bin_start:4.1f}-{bin_end:<8.1f}"
                        f"{int(bio_count):8d}"
                        f"{int(gas_count):8d}"
                        f"{int(comb_count):8d}"
                        f"{int(total):8d}"
                    )

        # 積み上げヒストグラムを作成
        bottom = np.zeros_like(hist_data.get("bio", np.zeros(len(bins) - 1)))

        # HotspotTypeのリテラル値を使用してイテレーション
        for type_name in get_args(HotspotType):
            if type_name in hist_data:
                plt.bar(
                    bins[:-1],
                    hist_data[type_name],
                    width=np.diff(bins)[0],
                    bottom=bottom,
                    color=hotspot_colors[type_name],
                    label=type_name,
                    alpha=0.6,
                    align="edge",
                )
                bottom += hist_data[type_name]

        if yscale_log:
            plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # 軸の範囲を設定
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

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

    def plot_mapbox(
        self,
        df: pd.DataFrame,
        col_conc: str,
        mapbox_access_token: str,
        sort_conc_column: bool = True,
        output_dirpath: str | Path | None = None,
        output_filename: str = "mapbox_plot.html",
        col_lat: str = "latitude",
        col_lon: str = "longitude",
        colorscale: str = "Jet",
        center_lat: float | None = None,
        center_lon: float | None = None,
        zoom: float = 12,
        width: int = 700,
        height: int = 700,
        tick_font_family: str = "Arial",
        title_font_family: str = "Arial",
        tick_font_size: int = 12,
        title_font_size: int = 14,
        marker_size: int = 4,
        colorbar_title: str | None = None,
        value_range: tuple[float, float] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
    ) -> None:
        """
        Mapbox上にデータをプロットします。

        Parameters
        ----------
            df: pd.DataFrame
                プロットするデータを含むDataFrame
            col_conc: str
                カラーマッピングに使用する列名
            mapbox_access_token: str
                Mapboxのアクセストークン
            sort_conc_column: bool, optional
                濃度列をソートするかどうか。デフォルトはTrue
            output_dirpath: str | Path | None, optional
                出力ディレクトリのパス。デフォルトはNone
            output_filename: str, optional
                出力ファイル名。デフォルトは"mapbox_plot.html"
            col_lat: str, optional
                緯度の列名。デフォルトは"latitude"
            col_lon: str, optional
                経度の列名。デフォルトは"longitude"
            colorscale: str, optional
                使用するカラースケール。デフォルトは"Jet"
            center_lat: float | None, optional
                中心緯度。デフォルトはNoneで、self._center_latを使用
            center_lon: float | None, optional
                中心経度。デフォルトはNoneで、self._center_lonを使用
            zoom: float, optional
                マップの初期ズームレベル。デフォルトは12
            width: int, optional
                プロットの幅(ピクセル)。デフォルトは700
            height: int, optional
                プロットの高さ(ピクセル)。デフォルトは700
            tick_font_family: str, optional
                カラーバーの目盛りフォントファミリー。デフォルトは"Arial"
            title_font_family: str, optional
                カラーバーのタイトルフォントファミリー。デフォルトは"Arial"
            tick_font_size: int, optional
                カラーバーの目盛りフォントサイズ。デフォルトは12
            title_font_size: int, optional
                カラーバーのタイトルフォントサイズ。デフォルトは14
            marker_size: int, optional
                マーカーのサイズ。デフォルトは4
            colorbar_title: str | None, optional
                カラーバーのタイトル。デフォルトはNoneでcol_concを使用
            value_range: tuple[float, float] | None, optional
                カラーマッピングの範囲。デフォルトはNoneでデータの最小値と最大値を使用
            save_fig: bool, optional
                図を保存するかどうか。デフォルトはTrue
            show_fig: bool, optional
                図を表示するかどうか。デフォルトはTrue

        Returns
        -------
            None

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> df = pd.DataFrame({
        ...     'latitude': [35.681236, 35.681237],
        ...     'longitude': [139.767125, 139.767126],
        ...     'concentration': [1.2, 1.5]
        ... })
        >>> analyzer.plot_mapbox(
        ...     df=df,
        ...     col_conc='concentration',
        ...     mapbox_access_token='your_token_here'
        ... )
        """
        df_mapping: pd.DataFrame = df.copy().dropna(subset=[col_conc])
        if sort_conc_column:
            df_mapping = df_mapping.sort_values(col_conc)
        # 中心座標の設定
        center_lat = center_lat if center_lat is not None else self._center_lat
        center_lon = center_lon if center_lon is not None else self._center_lon

        # カラーマッピングの範囲を設定
        cmin, cmax = 0, 0
        if value_range is None:
            cmin = df_mapping[col_conc].min()
            cmax = df_mapping[col_conc].max()
        else:
            cmin, cmax = value_range

        # カラーバーのタイトルを設定
        title_text = colorbar_title if colorbar_title is not None else col_conc

        # Scattermapboxのデータを作成
        scatter_data = go.Scattermapbox(
            lat=df_mapping[col_lat],
            lon=df_mapping[col_lon],
            text=df_mapping[col_conc].astype(str),
            hoverinfo="text",
            mode="markers",
            marker={
                "color": df_mapping[col_conc],
                "size": marker_size,
                "reversescale": False,
                "autocolorscale": False,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
                "colorbar": {
                    "tickformat": "3.2f",
                    "outlinecolor": "black",
                    "outlinewidth": 1.5,
                    "ticks": "outside",
                    "ticklen": 7,
                    "tickwidth": 1.5,
                    "tickcolor": "black",
                    "tickfont": {
                        "family": tick_font_family,
                        "color": "black",
                        "size": tick_font_size,
                    },
                    "title": {
                        "text": title_text,
                        "side": "top",
                    },  # カラーバーのタイトルを設定
                    "titlefont": {
                        "family": title_font_family,
                        "color": "black",
                        "size": title_font_size,
                    },
                },
            },
        )

        # レイアウトの設定
        layout = go.Layout(
            width=width,
            height=height,
            showlegend=False,
            mapbox={
                "accesstoken": mapbox_access_token,
                "center": {"lat": center_lat, "lon": center_lon},
                "zoom": zoom,
            },
        )

        # 図の作成
        fig = go.Figure(data=[scatter_data], layout=layout)

        # 図の保存
        if save_fig:
            # 保存時の出力ディレクトリチェック
            if output_dirpath is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirpathを指定する必要があります。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath = os.path.join(output_dirpath, output_filename)
            pyo.plot(fig, filename=output_filepath, auto_open=False)
            self.logger.info(f"Mapboxプロットを保存しました: {output_filepath}")
        # 図の表示
        if show_fig:
            pyo.iplot(fig)

    def plot_scatter_c2c1(
        self,
        hotspots: list[HotspotData],
        output_dirpath: str | Path | None = None,
        output_filename: str = "scatter_c2c1.png",
        figsize: tuple[float, float] = (4, 4),
        dpi: float | None = 350,
        hotspot_colors: dict[HotspotType, str] | None = None,
        hotspot_labels: dict[HotspotType, str] | None = None,
        fontsize: float = 12,
        xlim: tuple[float, float] = (0, 2.0),
        ylim: tuple[float, float] = (0, 50),
        xlabel: str = "Δ$\\mathregular{CH_{4}}$ (ppm)",
        ylabel: str = "Δ$\\mathregular{C_{2}H_{6}}$ (ppb)",
        xscale_log: bool = False,
        yscale_log: bool = False,
        add_legend: bool = True,
        save_fig: bool = True,
        show_fig: bool = True,
        add_ratio_labels: bool = True,
        ratio_labels: dict[float, tuple[float, float, str]] | None = None,
    ) -> None:
        """
        検出されたホットスポットのΔC2H6とΔCH4の散布図をプロットします。

        Parameters
        ----------
            hotspots: list[HotspotData]
                プロットするホットスポットのリスト
            output_dirpath: str | Path | None, optional
                保存先のディレクトリパス。未指定の場合はNoneとなります。
            output_filename: str, optional
                保存するファイル名。デフォルト値は"scatter_c2c1.png"です。
            figsize: tuple[float, float], optional
                図のサイズ。デフォルト値は(4, 4)です。
            dpi: float | None, optional
                解像度。デフォルト値は350です。
            hotspot_colors: dict[HotspotType, str] | None, optional
                ホットスポットの色を定義する辞書。未指定の場合は{"bio": "blue", "gas": "red", "comb": "green"}が使用されます。
            hotspot_labels: dict[HotspotType, str] | None, optional
                ホットスポットのラベルを定義する辞書。未指定の場合は{"bio": "bio", "gas": "gas", "comb": "comb"}が使用されます。
            fontsize: float, optional
                フォントサイズ。デフォルト値は12です。
            xlim: tuple[float, float], optional
                x軸の範囲。デフォルト値は(0, 2.0)です。
            ylim: tuple[float, float], optional
                y軸の範囲。デフォルト値は(0, 50)です。
            xlabel: str, optional
                x軸のラベル。デフォルト値は"Δ$\\mathregular{CH_{4}}$ (ppm)"です。
            ylabel: str, optional
                y軸のラベル。デフォルト値は"Δ$\\mathregular{C_{2}H_{6}}$ (ppb)"です。
            xscale_log: bool, optional
                x軸を対数スケールにするかどうか。デフォルト値はFalseです。
            yscale_log: bool, optional
                y軸を対数スケールにするかどうか。デフォルト値はFalseです。
            add_legend: bool, optional
                凡例を追加するかどうか。デフォルト値はTrueです。
            save_fig: bool, optional
                図の保存を許可するフラグ。デフォルト値はTrueです。
            show_fig: bool, optional
                図の表示を許可するフラグ。デフォルト値はTrueです。
            add_ratio_labels: bool, optional
                比率線を表示するかどうか。デフォルト値はTrueです。
            ratio_labels: dict[float, tuple[float, float, str]] | None, optional
                比率線とラベルの設定。キーは比率値、値は(x位置, y位置, ラベルテキスト)のタプル。
                未指定の場合は以下のデフォルト値が使用されます:
                {0.001: (1.25, 2, "0.001"), 0.005: (1.25, 8, "0.005"),
                0.010: (1.25, 15, "0.01"), 0.020: (1.25, 30, "0.02"),
                0.030: (1.0, 40, "0.03"), 0.076: (0.20, 42, "0.076 (Osaka)")}

        Returns
        -------
            None

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> hotspots = analyzer.analyze_hotspots()
        >>> analyzer.plot_scatter_c2c1(
        ...     hotspots=hotspots,
        ...     output_dirpath="output",
        ...     xlim=(0, 5),
        ...     ylim=(0, 100)
        ... )
        """
        # デフォルト値の設定
        if hotspot_colors is None:
            hotspot_colors = {
                "bio": "blue",
                "gas": "red",
                "comb": "green",
            }
        if hotspot_labels is None:
            hotspot_labels = {
                "bio": "bio",
                "gas": "gas",
                "comb": "comb",
            }
        if ratio_labels is None:
            ratio_labels = {
                0.001: (1.25, 2, "0.001"),
                0.005: (1.25, 8, "0.005"),
                0.010: (1.25, 15, "0.01"),
                0.020: (1.25, 30, "0.02"),
                0.030: (1.0, 40, "0.03"),
                0.076: (0.20, 42, "0.076 (Osaka)"),
            }
        plt.rcParams["font.size"] = fontsize
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # タイプごとのデータを収集
        type_data: dict[HotspotType, list[tuple[float, float]]] = {
            "bio": [],
            "gas": [],
            "comb": [],
        }
        for spot in hotspots:
            type_data[spot.type].append((spot.delta_ch4, spot.delta_c2h6))

        # タイプごとにプロット(データが存在する場合のみ)
        for spot_type, data in type_data.items():
            if data:  # データが存在する場合のみプロット
                ch4_values, c2h6_values = zip(*data, strict=True)
                plt.plot(
                    ch4_values,
                    c2h6_values,
                    "o",
                    c=hotspot_colors[spot_type],
                    alpha=0.5,
                    ms=2,
                    label=hotspot_labels[spot_type],
                )

        # プロット後、軸の設定前に比率の線を追加
        x = np.array([0, 5])
        base_ch4 = 0.0
        base = 0.0

        # 各比率に対して線を引く
        if ratio_labels is not None:
            if not add_ratio_labels:
                raise ValueError(
                    "ratio_labels に基づいて比率線を描画する場合は、 add_ratio_labels = True を指定してください。"
                )
            for ratio, (x_pos, y_pos, label) in ratio_labels.items():
                y = (x - base_ch4) * 1000 * ratio + base
                plt.plot(x, y, "-", c="black", alpha=0.5)
                plt.text(x_pos, y_pos, label)

        # 軸の設定
        if xscale_log:
            plt.xscale("log")
        if yscale_log:
            plt.yscale("log")

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if add_legend:
            plt.legend()

        # グラフの保存または表示
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            output_filepath: str = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, bbox_inches="tight")
            self.logger.info(f"散布図を保存しました: {output_filepath}")
        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_conc_timeseries(
        self,
        output_dirpath: str | Path | None = None,
        output_filename: str = "timeseries.png",
        figsize: tuple[float, float] = (8, 4),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
        col_ch4: str = "ch4_ppm",
        col_c2h6: str = "c2h6_ppb",
        col_h2o: str = "h2o_ppm",
        ylim_ch4: tuple[float, float] | None = None,
        ylim_c2h6: tuple[float, float] | None = None,
        ylim_h2o: tuple[float, float] | None = None,
        yscale_log_ch4: bool = False,
        yscale_log_c2h6: bool = False,
        yscale_log_h2o: bool = False,
        font_size: float = 12,
        label_pad: float = 10,
        line_color: str = "black",
    ) -> None:
        """
        CH4、C2H6、H2Oの時系列データをプロットします。

        Parameters
        ----------
            output_dirpath: str | Path | None, optional
                保存先のディレクトリを指定します。save_fig=Trueの場合は必須です。デフォルト値はNoneです。
            output_filename: str, optional
                保存するファイル名を指定します。デフォルト値は"timeseries.png"です。
            figsize: tuple[float, float], optional
                図のサイズを指定します。デフォルト値は(8, 4)です。
            dpi: float | None, optional
                図の解像度を指定します。デフォルト値は350です。
            save_fig: bool, optional
                図を保存するかどうかを指定します。デフォルト値はTrueです。
            show_fig: bool, optional
                図を表示するかどうかを指定します。デフォルト値はTrueです。
            col_ch4: str, optional
                CH4データの列名を指定します。デフォルト値は"ch4_ppm"です。
            col_c2h6: str, optional
                C2H6データの列名を指定します。デフォルト値は"c2h6_ppb"です。
            col_h2o: str, optional
                H2Oデータの列名を指定します。デフォルト値は"h2o_ppm"です。
            ylim_ch4: tuple[float, float] | None, optional
                CH4プロットのy軸範囲を指定します。デフォルト値はNoneです。
            ylim_c2h6: tuple[float, float] | None, optional
                C2H6プロットのy軸範囲を指定します。デフォルト値はNoneです。
            ylim_h2o: tuple[float, float] | None, optional
                H2Oプロットのy軸範囲を指定します。デフォルト値はNoneです。
            yscale_log_ch4: bool, optional
                CH4データのy軸を対数スケールで表示するかどうかを指定します。デフォルト値はFalseです。
            yscale_log_c2h6: bool, optional
                C2H6データのy軸を対数スケールで表示するかどうかを指定します。デフォルト値はFalseです。
            yscale_log_h2o: bool, optional
                H2Oデータのy軸を対数スケールで表示するかどうかを指定します。デフォルト値はFalseです。
            font_size: float, optional
                プロット全体のフォントサイズを指定します。デフォルト値は12です。
            label_pad: float, optional
                y軸ラベルとプロットの間隔を指定します。デフォルト値は10です。
            line_color: str, optional
                プロットする線の色を指定します。デフォルト値は"black"です。

        Returns
        -------
            None

        Examples
        --------
        >>> analyzer = MobileMeasurementAnalyzer(df)
        >>> analyzer.plot_conc_timeseries(
        ...     output_dirpath="output",
        ...     ylim_ch4=(1.8, 2.5),
        ...     ylim_c2h6=(0, 100),
        ...     ylim_h2o=(0, 20000)
        ... )
        """
        # プロットパラメータの設定
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.labelsize": font_size,
                "axes.titlesize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
            }
        )
        df_internal: pd.DataFrame = self.df.copy()

        # プロットの作成
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # CH4プロット
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(df_internal.index, df_internal[col_ch4], c=line_color)
        if ylim_ch4:
            ax1.set_ylim(ylim_ch4)
        if yscale_log_ch4:
            ax1.set_yscale("log")
        ax1.set_ylabel("$\\mathregular{CH_{4}}$ (ppm)", labelpad=label_pad)
        ax1.grid(True, alpha=0.3)

        # C2H6プロット
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(df_internal.index, df_internal[col_c2h6], c=line_color)
        if ylim_c2h6:
            ax2.set_ylim(ylim_c2h6)
        if yscale_log_c2h6:
            ax2.set_yscale("log")
        ax2.set_ylabel("$\\mathregular{C_{2}H_{6}}$ (ppb)", labelpad=label_pad)
        ax2.grid(True, alpha=0.3)

        # H2Oプロット
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(df_internal.index, df_internal[col_h2o], c=line_color)
        if ylim_h2o:
            ax3.set_ylim(ylim_h2o)
        if yscale_log_h2o:
            ax3.set_yscale("log")
        ax3.set_ylabel("$\\mathregular{H_{2}O}$ (ppm)", labelpad=label_pad)
        ax3.grid(True, alpha=0.3)

        # x軸のフォーマット調整
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # 軸のラベルとグリッド線の調整
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.grid(True, alpha=0.3)

        # サブプロット間の間隔調整
        plt.subplots_adjust(wspace=0.38, hspace=0.38)

        # 図の保存
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, bbox_inches="tight")

        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def plot_conc_timeseries_with_hotspots(
        self,
        hotspots: list[HotspotData] | None = None,
        output_dirpath: str | Path | None = None,
        output_filename: str = "timeseries_with_hotspots.png",
        figsize: tuple[float, float] = (8, 6),
        dpi: float | None = 350,
        save_fig: bool = True,
        show_fig: bool = True,
        col_ch4: str = "ch4_ppm",
        col_c2h6: str = "c2h6_ppb",
        col_h2o: str = "h2o_ppm",
        add_legend: bool = True,
        legend_bbox_to_anchor: tuple[float, float] = (0.5, 0.05),
        legend_ncol: int | None = None,
        font_size: float = 12,
        label_pad: float = 10,
        line_color: str = "black",
        hotspot_colors: dict[HotspotType, str] | None = None,
        hotspot_markerscale: float = 1,
        hotspot_size: int = 10,
        time_margin_minutes: float = 2.0,
        ylim_ch4: tuple[float, float] | None = None,
        ylim_c2h6: tuple[float, float] | None = None,
        ylim_h2o: tuple[float, float] | None = None,
        ylim_ratio: tuple[float, float] | None = None,
        yscale_log_ch4: bool = False,
        yscale_log_c2h6: bool = False,
        yscale_log_h2o: bool = False,
        yscale_log_ratio: bool = False,
        ylabel_ch4: str = "$\\mathregular{CH_{4}}$ (ppm)",
        ylabel_c2h6: str = "$\\mathregular{C_{2}H_{6}}$ (ppb)",
        ylabel_h2o: str = "$\\mathregular{H_{2}O}$ (ppm)",
        ylabel_ratio: str = "ΔC$_2$H$_6$/ΔCH$_4$\n(ppb ppm$^{-1}$)",
    ) -> None:
        """
        時系列データとホットスポットをプロットします。

        Parameters
        ----------
            hotspots: list[HotspotData] | None, optional
                表示するホットスポットのリスト。デフォルトはNoneで、この場合ホットスポットは表示されません。
            output_dirpath: str | Path | None, optional
                出力先ディレクトリのパス。save_fig=Trueの場合は必須です。デフォルトはNoneです。
            output_filename: str, optional
                保存するファイル名。デフォルトは"timeseries_with_hotspots.png"です。
            figsize: tuple[float, float], optional
                図のサイズ。デフォルトは(8, 6)です。
            dpi: float | None, optional
                図の解像度。デフォルトは350です。
            save_fig: bool, optional
                図を保存するかどうか。デフォルトはTrueです。
            show_fig: bool, optional
                図を表示するかどうか。デフォルトはTrueです。
            col_ch4: str, optional
                CH4データのカラム名。デフォルトは"ch4_ppm"です。
            col_c2h6: str, optional
                C2H6データのカラム名。デフォルトは"c2h6_ppb"です。
            col_h2o: str, optional
                H2Oデータのカラム名。デフォルトは"h2o_ppm"です。
            add_legend: bool, optional
                ホットスポットの凡例を表示するかどうか。デフォルトはTrueです。
            legend_bbox_to_anchor: tuple[float, float], optional
                ホットスポットの凡例の位置。デフォルトは(0.5, 0.05)です。
            legend_ncol: int | None, optional
                凡例のカラム数。デフォルトはNoneで、この場合ホットスポットの種類数に基づいて一行で表示します。
            font_size: float, optional
                基本フォントサイズ。デフォルトは12です。
            label_pad: float, optional
                y軸ラベルのパディング。デフォルトは10です。
            line_color: str, optional
                線の色。デフォルトは"black"です。
            hotspot_colors: dict[HotspotType, str] | None, optional
                ホットスポットタイプごとの色指定。デフォルトはNoneで、{"bio": "blue", "gas": "red", "comb": "green"}が使用されます。
            hotspot_markerscale: float, optional
                ホットスポットの凡例でのサイズ。hotspot_sizeに対する相対値。デフォルトは1です。
            hotspot_size: int, optional
                ホットスポットの図でのサイズ。デフォルトは10です。
            time_margin_minutes: float, optional
                プロットの時間軸の余白(分)。デフォルトは2.0分です。
            ylim_ch4: tuple[float, float] | None, optional
                CH4プロットのy軸範囲。デフォルトはNoneで、自動設定されます。
            ylim_c2h6: tuple[float, float] | None, optional
                C2H6プロットのy軸範囲。デフォルトはNoneで、自動設定されます。
            ylim_h2o: tuple[float, float] | None, optional
                H2Oプロットのy軸範囲。デフォルトはNoneで、自動設定されます。
            ylim_ratio: tuple[float, float] | None, optional
                比率プロットのy軸範囲。デフォルトはNoneで、自動設定されます。
            yscale_log_ch4: bool, optional
                CH4データのy軸を対数スケールで表示するかどうか。デフォルトはFalseです。
            yscale_log_c2h6: bool, optional
                C2H6データのy軸を対数スケールで表示するかどうか。デフォルトはFalseです。
            yscale_log_h2o: bool, optional
                H2Oデータのy軸を対数スケールで表示するかどうか。デフォルトはFalseです。
            yscale_log_ratio: bool, optional
                比率データのy軸を対数スケールで表示するかどうか。デフォルトはFalseです。
            ylabel_ch4: str, optional
                CH4プロットのy軸ラベル。デフォルトは"$\\mathregular{CH_{4}}$ (ppm)"です。
            ylabel_c2h6: str, optional
                C2H6プロットのy軸ラベル。デフォルトは"$\\mathregular{C_{2}H_{6}}$ (ppb)"です。
            ylabel_h2o: str, optional
                H2Oプロットのy軸ラベル。デフォルトは"$\\mathregular{H_{2}O}$ (ppm)"です。
            ylabel_ratio: str, optional
                比率プロットのy軸ラベル。デフォルトは"ΔC$_2$H$_6$/ΔCH$_4$\\n(ppb ppm$^{-1}$)"です。

        Examples
        --------
        基本的な使用方法:
        >>> analyzer = MobileMeasurementAnalyzer(df)
        >>> analyzer.plot_conc_timeseries_with_hotspots()

        ホットスポットを指定して保存する:
        >>> hotspots = [HotspotData(...), HotspotData(...)]
        >>> analyzer.plot_conc_timeseries_with_hotspots(
        ...     hotspots=hotspots,
        ...     output_dirpath="output",
        ...     save_fig=True
        ... )

        カスタマイズした表示:
        >>> analyzer.plot_conc_timeseries_with_hotspots(
        ...     figsize=(12, 8),
        ...     ylim_ch4=(1.8, 2.5),
        ...     yscale_log_c2h6=True,
        ...     hotspot_colors={"bio": "purple", "gas": "orange"}
        ... )
        """
        if hotspot_colors is None:
            hotspot_colors = {"bio": "blue", "gas": "red", "comb": "green"}
        # プロットパラメータの設定
        plt.rcParams.update(
            {
                "font.size": font_size,
                "axes.labelsize": font_size,
                "axes.titlesize": font_size,
                "xtick.labelsize": font_size,
                "ytick.labelsize": font_size,
            }
        )

        df_internal: pd.DataFrame = self.df.copy()

        # プロットの作成
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # サブプロットのグリッドを作成 (4行1列)
        gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

        # 時間軸の範囲を設定(余白付き)
        time_min = df_internal.index.min()
        time_max = df_internal.index.max()
        time_margin = pd.Timedelta(minutes=time_margin_minutes)
        plot_time_min = time_min - time_margin
        plot_time_max = time_max + time_margin

        # CH4プロット
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df_internal.index, df_internal[col_ch4], c=line_color)
        if ylim_ch4:
            ax1.set_ylim(ylim_ch4)
        if yscale_log_ch4:
            ax1.set_yscale("log")
        ax1.set_ylabel(ylabel_ch4, labelpad=label_pad)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(plot_time_min, plot_time_max)

        # C2H6プロット
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(df_internal.index, df_internal[col_c2h6], c=line_color)
        if ylim_c2h6:
            ax2.set_ylim(ylim_c2h6)
        if yscale_log_c2h6:
            ax2.set_yscale("log")
        ax2.set_ylabel(ylabel_c2h6, labelpad=label_pad)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(plot_time_min, plot_time_max)

        # H2Oプロット
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(df_internal.index, df_internal[col_h2o], c=line_color)
        if ylim_h2o:
            ax3.set_ylim(ylim_h2o)
        if yscale_log_h2o:
            ax3.set_yscale("log")
        ax3.set_ylabel(ylabel_h2o, labelpad=label_pad)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(plot_time_min, plot_time_max)

        # ホットスポットの比率プロット
        ax4 = fig.add_subplot(gs[3])

        if hotspots:
            # ホットスポットをDataFrameに変換
            hotspot_df = pd.DataFrame(
                [
                    {
                        "timestamp": pd.to_datetime(spot.timestamp),
                        "delta_ratio": spot.delta_ratio,
                        "type": spot.type,
                    }
                    for spot in hotspots
                ]
            )

            # タイプごとにプロット
            for spot_type in set(hotspot_df["type"]):
                type_data = hotspot_df[hotspot_df["type"] == spot_type]

                # 点をプロット
                ax4.scatter(
                    type_data["timestamp"],
                    type_data["delta_ratio"],
                    c=hotspot_colors.get(spot_type, "black"),
                    label=spot_type,
                    alpha=0.6,
                    s=hotspot_size,
                )

        ax4.set_ylabel(ylabel_ratio, labelpad=label_pad)
        if ylim_ratio:
            ax4.set_ylim(ylim_ratio)
        if yscale_log_ratio:
            ax4.set_yscale("log")
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(plot_time_min, plot_time_max)  # 他のプロットと同じ時間範囲を設定

        # 凡例を図の下部に配置
        if hotspots and add_legend:
            ncol = (
                legend_ncol if legend_ncol is not None else len(set(hotspot_df["type"]))
            )
            # markerscaleは元のサイズに対する倍率を指定するため、
            # 目的のサイズ(100)をプロットのマーカーサイズで割ることで、適切な倍率を計算しています
            fig.legend(
                bbox_to_anchor=legend_bbox_to_anchor,
                loc="upper center",
                ncol=ncol,
                columnspacing=1.0,
                handletextpad=0.5,
                markerscale=hotspot_markerscale,
            )

        # x軸のフォーマット調整(全てのサブプロットで共通)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.tick_params(axis="both", which="major", labelsize=font_size)
            ax.grid(True, alpha=0.3)

        # サブプロット間の間隔調整と凡例のためのスペース確保
        plt.subplots_adjust(hspace=0.38, bottom=0.12)  # bottomを0.15から0.12に変更

        # 図の保存
        if save_fig:
            if output_dirpath is None:
                raise ValueError(
                    "save_fig = True の場合、 output_dirpath を指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dirpath, exist_ok=True)
            output_filepath = os.path.join(output_dirpath, output_filename)
            plt.savefig(output_filepath, bbox_inches="tight", dpi=dpi)

        if show_fig:
            plt.show()
        plt.close(fig=fig)

    def _detect_hotspots(
        self,
        df: pd.DataFrame,
        ch4_enhance_threshold: float,
    ) -> list[HotspotData]:
        """
        シンプル化したホットスポット検出

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム
            ch4_enhance_threshold: float
                CH4増加の閾値

        Returns
        ----------
            list[HotspotData]
                検出されたホットスポットのリスト
        """
        hotspots: list[HotspotData] = []

        # CH4増加量が閾値を超えるデータポイントを抽出
        enhanced_mask = df["ch4_ppm_delta"] >= ch4_enhance_threshold

        if enhanced_mask.any():
            timestamp = df["timestamp"][enhanced_mask]
            lat = df["latitude"][enhanced_mask]
            lon = df["longitude"][enhanced_mask]
            delta_ratio = df["c2c1_ratio_delta"][enhanced_mask]
            delta_ch4 = df["ch4_ppm_delta"][enhanced_mask]
            delta_c2h6 = df["c2h6_ppb_delta"][enhanced_mask]

            # 各ポイントに対してホットスポットを作成
            for i in range(len(lat)):
                if pd.notna(delta_ratio.iloc[i]):
                    current_lat = lat.iloc[i]
                    current_lon = lon.iloc[i]
                    correlation = df["c1c2_correlation"].iloc[i]

                    # 比率に基づいてタイプを決定
                    spot_type: HotspotType = "bio"
                    if delta_ratio.iloc[i] >= 100:
                        spot_type = "comb"
                    elif delta_ratio.iloc[i] >= 5:
                        spot_type = "gas"

                    angle: float = MobileMeasurementAnalyzer._calculate_angle(
                        lat=current_lat,
                        lon=current_lon,
                        center_lat=self._center_lat,
                        center_lon=self._center_lon,
                    )
                    section: int = self._determine_section(angle)
                    # 値を取得してpd.Timestampに変換
                    timestamp_raw = pd.Timestamp(str(timestamp.values[i]))

                    hotspots.append(
                        HotspotData(
                            timestamp=timestamp_raw.strftime("%Y-%m-%d %H:%M:%S"),
                            angle=angle,
                            avg_lat=current_lat,
                            avg_lon=current_lon,
                            delta_ch4=delta_ch4.iloc[i],
                            delta_c2h6=delta_c2h6.iloc[i],
                            correlation=max(-1, min(1, correlation)),
                            delta_ratio=delta_ratio.iloc[i],
                            section=section,
                            type=spot_type,
                        )
                    )

        return hotspots

    def _determine_section(self, angle: float) -> int:
        """
        角度に基づいて所属する区画を特定します。

        Parameters
        ----------
            angle: float
                計算された角度

        Returns
        ----------
            int
                区画番号(0-based-index)
        """
        for section_num, (start, end) in self._sections.items():
            if start <= angle < end:
                return section_num
        # -180度の場合は最後の区画に含める
        return self._num_sections - 1

    def _load_all_combined_data(
        self, input_configs: list[MobileMeasurementConfig]
    ) -> pd.DataFrame:
        """
        全入力ファイルのデータを読み込み、結合したデータフレームを返します。

        Parameters
        ----------
            input_configs: list[MobileMeasurementConfig]
                読み込むファイルの設定リスト。

        Returns
        ----------
            pd.DataFrame
                読み込まれたすべてのデータを結合したデータフレーム。
        """
        dfs: list[pd.DataFrame] = []
        for config in input_configs:
            df, _ = self._load_data(config)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _load_data(
        self,
        config: MobileMeasurementConfig,
        columns_to_shift: list[str] | None = None,
        col_timestamp: str = "timestamp",
        col_latitude: str = "latitude",
        col_longitude: str = "longitude",
    ) -> tuple[pd.DataFrame, str]:
        """
        測定データを読み込み、前処理を行うメソッド。

        Parameters
        ----------
            config: MobileMeasurementConfig
                入力ファイルの設定を含むオブジェクト。ファイルパス、遅れ時間、サンプリング周波数、補正タイプなどの情報を持つ。
            columns_to_shift: list[str] | None, optional
                シフトを適用するカラム名のリスト。Noneの場合のデフォルトは["ch4_ppm", "c2h6_ppb", "h2o_ppm"]で、これらのカラムに対して遅れ時間の補正が行われる。
            col_timestamp: str, optional
                タイムスタンプのカラム名。デフォルトは"timestamp"。
            col_latitude: str, optional
                緯度のカラム名。デフォルトは"latitude"。
            col_longitude: str, optional
                経度のカラム名。デフォルトは"longitude"。

        Returns
        ----------
            tuple[pd.DataFrame, str]
                読み込まれたデータフレームとそのソース名を含むタプル。データフレームは前処理が施されており、ソース名はファイル名から抽出されたもの。
        """
        if columns_to_shift is None:
            columns_to_shift = ["ch4_ppm", "c2h6_ppb", "h2o_ppm"]
        source_name: str = MobileMeasurementAnalyzer.extract_source_name_from_path(
            config.path
        )
        df: pd.DataFrame = pd.read_csv(config.path, na_values=self._na_values)

        # カラム名の標準化(測器に依存しない汎用的な名前に変更)
        df = df.rename(columns=self._column_mapping)
        df[col_timestamp] = pd.to_datetime(df[col_timestamp])
        # インデックスを設定(元のtimestampカラムは保持)
        df = df.set_index(col_timestamp, drop=False)

        if config.lag < 0:
            raise ValueError(
                f"Invalid lag value: {config.lag}. Must be a non-negative float."
            )

        # サンプリング周波数に応じてシフト量を調整
        shift_periods: int = -int(config.lag * config.fs)  # fsを掛けて補正

        # 遅れ時間の補正
        for col in columns_to_shift:
            df[col] = df[col].shift(shift_periods)

        # 緯度経度とシフト対象カラムのnanを一度に削除
        df = df.dropna(subset=[col_latitude, col_longitude, *columns_to_shift])

        # 水蒸気補正の適用
        if config.h2o_correction is not None and all(
            x is not None
            for x in [
                config.h2o_correction.coef_b,
                config.h2o_correction.coef_c,
            ]
        ):
            h2o_correction: H2OCorrectionConfig = config.h2o_correction
            df = CorrectingUtils.correct_h2o_interference(
                df=df,
                coef_b=float(h2o_correction.coef_b),  # type: ignore
                coef_c=float(h2o_correction.coef_c),  # type: ignore
                h2o_ppm_threshold=h2o_correction.h2o_ppm_threshold,
                target_h2o_ppm=h2o_correction.target_h2o_ppm,
            )

        # バイアス除去の適用
        if config.bias_removal is not None:
            bias_removal: BiasRemovalConfig = config.bias_removal
            df = CorrectingUtils.remove_bias(
                df=df,
                quantile_value=bias_removal.quantile_value,
                base_ch4_ppm=bias_removal.base_ch4_ppm,
                base_c2h6_ppb=bias_removal.base_c2h6_ppb,
            )

        return df, source_name

    @staticmethod
    def _calculate_angle(
        lat: float, lon: float, center_lat: float, center_lon: float
    ) -> float:
        """
        中心からの角度を計算

        Parameters
        ----------
            lat: float
                対象地点の緯度
            lon: float
                対象地点の経度
            center_lat: float
                中心の緯度
            center_lon: float
                中心の経度

        Returns
        ----------
            float
                真北を0°として時計回りの角度(-180°から180°)
        """
        d_lat: float = lat - center_lat
        d_lon: float = lon - center_lon
        # arctanを使用して角度を計算(ラジアン)
        angle_rad: float = math.atan2(d_lon, d_lat)
        # ラジアンから度に変換(-180から180の範囲)
        angle_deg: float = math.degrees(angle_rad)
        return angle_deg

    @classmethod
    def _calculate_distance(
        cls, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        2点間の距離をメートル単位で計算(Haversine formula)

        Parameters
        ----------
            lat1: float
                地点1の緯度
            lon1: float
                地点1の経度
            lat2: float
                地点2の緯度
            lon2: float
                地点2の経度

        Returns
        ----------
            float
                2地点間の距離(メートル)
        """
        const_r = cls.EARTH_RADIUS_METERS

        # 緯度経度をラジアンに変換
        lat1_rad: float = math.radians(lat1)
        lon1_rad: float = math.radians(lon1)
        lat2_rad: float = math.radians(lat2)
        lon2_rad: float = math.radians(lon2)

        # 緯度と経度の差分
        dlat: float = lat2_rad - lat1_rad
        dlon: float = lon2_rad - lon1_rad

        # Haversine formula
        a: float = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c: float = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return const_r * c  # メートル単位での距離

    @staticmethod
    def _calculate_hotspots_parameters(
        df: pd.DataFrame,
        window_size: int,
        col_ch4_ppm: str,
        col_c2h6_ppb: str,
        col_h2o_ppm: str,
        ch4_ppm_delta_min: float = 0.05,
        ch4_ppm_delta_max: float = float("inf"),
        c2h6_ppb_delta_min: float = 0.0,
        c2h6_ppb_delta_max: float = 1000.0,
        h2o_ppm_threshold: float = 2000,
        rolling_method: RollingMethod = "quantile",
        quantile_value: float = 0.05,
    ) -> pd.DataFrame:
        """
        ホットスポットのパラメータを計算します。
        このメソッドは、指定されたデータフレームに対して移動平均(または指定されたquantile)や相関を計算し、
        各種のデルタ値や比率を追加します。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム
            window_size: int
                移動窓のサイズ
            col_ch4_ppm: str
                CH4濃度のカラム名
            col_c2h6_ppb: str
                C2H6濃度のカラム名
            col_h2o_ppm: str
                H2O濃度のカラム名
            ch4_ppm_delta_min: float, optional
                CH4濃度の下限閾値。この値未満のデータは除外されます。デフォルト値は0.05です。
            ch4_ppm_delta_max: float, optional
                CH4濃度の上限閾値。この値を超えるデータは除外されます。デフォルト値は無限大です。
            c2h6_ppb_delta_min: float, optional
                C2H6濃度の下限閾値。この値未満のデータは除外されます。デフォルト値は0.0です。
            c2h6_ppb_delta_max: float, optional
                C2H6濃度の上限閾値。この値を超えるデータは除外されます。デフォルト値は1000.0です。
            h2o_ppm_threshold: float, optional
                H2Oの閾値。デフォルト値は2000です。
            rolling_method: RollingMethod, optional
                バックグラウンド値の移動計算に使用する方法を指定します。'quantile'または'mean'を指定できます。デフォルト値は'quantile'です。
            quantile_value: float, optional
                使用するquantileの値。デフォルト値は0.05です。

        Returns
        ----------
            pd.DataFrame
                計算されたパラメータを含むデータフレーム

        Raises
        ----------
            ValueError
                quantile_value が0未満または1を超える場合に発生します。
        """
        # 引数のバリデーション
        if quantile_value < 0 or quantile_value > 1:
            raise ValueError(
                "quantile_value は0以上1以下の float で指定する必要があります。"
            )

        # データのコピーを作成
        df_internal: pd.DataFrame = df.copy()

        # 移動相関の計算
        df_internal["c1c2_correlation"] = (
            df_internal[col_ch4_ppm]
            .rolling(window=window_size)
            .corr(df_internal[col_c2h6_ppb])
        )

        # バックグラウンド値の計算(指定されたパーセンタイルまたは移動平均)
        if rolling_method == "quantile":
            df_internal["ch4_ppm_mv"] = (
                df_internal[col_ch4_ppm]
                .rolling(window=window_size, center=True, min_periods=1)
                .quantile(quantile_value)
            )
            df_internal["c2h6_ppb_mv"] = (
                df_internal[col_c2h6_ppb]
                .rolling(window=window_size, center=True, min_periods=1)
                .quantile(quantile_value)
            )
        elif rolling_method == "mean":
            df_internal["ch4_ppm_mv"] = (
                df_internal[col_ch4_ppm]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )
            df_internal["c2h6_ppb_mv"] = (
                df_internal[col_c2h6_ppb]
                .rolling(window=window_size, center=True, min_periods=1)
                .mean()
            )

        # デルタ値の計算
        df_internal["ch4_ppm_delta"] = (
            df_internal[col_ch4_ppm] - df_internal["ch4_ppm_mv"]
        )
        df_internal["c2h6_ppb_delta"] = (
            df_internal[col_c2h6_ppb] - df_internal["c2h6_ppb_mv"]
        )

        # C2H6/CH4の比率計算
        df_internal["c2c1_ratio"] = df_internal[col_c2h6_ppb] / df_internal[col_ch4_ppm]
        # デルタ値に基づく比の計算とフィルタリング
        df_internal["c2c1_ratio_delta"] = (
            df_internal["c2h6_ppb_delta"] / df_internal["ch4_ppm_delta"]
        )

        # フィルタリング条件の適用
        df_internal.loc[
            (df_internal["ch4_ppm_delta"] < ch4_ppm_delta_min)
            | (df_internal["ch4_ppm_delta"] > ch4_ppm_delta_max),
            "c2c1_ratio_delta",
        ] = np.nan
        df_internal.loc[
            (df_internal["c2h6_ppb_delta"] < c2h6_ppb_delta_min)
            | (df_internal["c2h6_ppb_delta"] > c2h6_ppb_delta_max),
            "c2h6_ppb_delta",
        ] = np.nan
        # c2h6_ppb_delta は0未満のものを一律0とする
        df_internal.loc[df_internal["c2h6_ppb_delta"] < 0, "c2c1_ratio_delta"] = 0.0

        # 水蒸気濃度によるフィルタリング
        df_internal.loc[
            df_internal[col_h2o_ppm] < h2o_ppm_threshold, [col_ch4_ppm, col_c2h6_ppb]
        ] = np.nan

        # 欠損値の除去
        df_internal = df_internal.dropna(subset=[col_ch4_ppm, col_c2h6_ppb])

        return df_internal

    @staticmethod
    def _calculate_window_size(window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Parameters
        ----------
            window_minutes: float
                時間窓の大きさ(分)

        Returns
        ----------
            int
                データポイント数
        """
        return int(60 * window_minutes)

    @staticmethod
    def _initialize_sections(
        num_sections: int, section_size: float
    ) -> dict[int, tuple[float, float]]:
        """
        指定された区画数と区画サイズに基づいて、区画の範囲を初期化します。

        Parameters
        ----------
            num_sections: int
                初期化する区画の数。
            section_size: float
                各区画の角度範囲のサイズ。

        Returns
        ----------
            dict[int, tuple[float, float]]
                区画番号(0-based-index)とその範囲の辞書。各区画は-180度から180度の範囲に分割されます。
        """
        sections: dict[int, tuple[float, float]] = {}
        for i in range(num_sections):
            # -180から180の範囲で区画を設定
            start_angle = -180 + i * section_size
            end_angle = -180 + (i + 1) * section_size
            sections[i] = (start_angle, end_angle)
        return sections

    @staticmethod
    def _is_duplicate_spot(
        current_lat: float,
        current_lon: float,
        current_time: str,
        used_positions: list[tuple[float, float, str, float]],
        check_time_all: bool,
        min_time_threshold_seconds: float,
        max_time_threshold_hours: float,
        hotspot_area_meter: float,
    ) -> bool:
        """
        与えられた地点が既存の地点と重複しているかを判定します。

        Parameters
        ----------
            current_lat: float
                判定する地点の緯度
            current_lon: float
                判定する地点の経度
            current_time: str
                判定する地点の時刻
            used_positions: list[tuple[float, float, str, float]]
                既存の地点情報のリスト (lat, lon, time, value)
            check_time_all: bool
                時間に関係なく重複チェックを行うかどうか
            min_time_threshold_seconds: float
                重複とみなす最小時間の閾値(秒)
            max_time_threshold_hours: float
                重複チェックを一時的に無視する最大時間の閾値(時間)
            hotspot_area_meter: float
                重複とみなす距離の閾値(m)

        Returns
        ----------
            bool
                重複している場合はTrue、そうでない場合はFalse
        """
        for used_lat, used_lon, used_time, _ in used_positions:
            # 距離チェック
            distance = MobileMeasurementAnalyzer._calculate_distance(
                lat1=current_lat, lon1=current_lon, lat2=used_lat, lon2=used_lon
            )

            if distance < hotspot_area_meter:
                # 時間差の計算(秒単位)
                time_diff = pd.Timedelta(
                    pd.to_datetime(current_time) - pd.to_datetime(used_time)
                ).total_seconds()
                time_diff_abs = abs(time_diff)

                if check_time_all:
                    # 時間に関係なく、距離が近ければ重複とみなす
                    return True
                else:
                    # 時間窓による判定を行う
                    if time_diff_abs <= min_time_threshold_seconds:
                        # Case 1: 最小時間閾値以内は重複とみなす
                        return True
                    elif time_diff_abs > max_time_threshold_hours * 3600:
                        # Case 2: 最大時間閾値を超えた場合は重複チェックをスキップ
                        continue
                    # Case 3: その間の時間差の場合は、距離が近ければ重複とみなす
                    return True

        return False

    @staticmethod
    def _normalize_configs(
        configs: list[MobileMeasurementConfig] | list[tuple[float, float, str | Path]],
    ) -> list[MobileMeasurementConfig]:
        """
        入力設定を標準化

        Parameters
        ----------
            configs: list[MobileMeasurementConfig] | list[tuple[float, float, str | Path]]
                入力設定のリスト

        Returns
        ----------
            list[MobileMeasurementConfig]
                標準化された入力設定のリスト
        """
        normalized: list[MobileMeasurementConfig] = []
        for inp in configs:
            if isinstance(inp, MobileMeasurementConfig):
                normalized.append(inp)  # すでに検証済みのため、そのまま追加
            else:
                fs, lag, path = inp
                normalized.append(
                    MobileMeasurementConfig.validate_and_create(
                        fs=fs, lag=lag, path=path
                    )
                )
        return normalized

    def remove_c2c1_ratio_duplicates(
        self,
        df: pd.DataFrame,
        min_time_threshold_seconds: float = 300,
        max_time_threshold_hours: float = 12.0,
        check_time_all: bool = True,
        hotspot_area_meter: float = 50.0,
        col_ch4_ppm: str = "ch4_ppm",
        col_ch4_ppm_mv: str = "ch4_ppm_mv",
        col_ch4_ppm_delta: str = "ch4_ppm_delta",
    ):
        """
        メタン濃度の増加が閾値を超えた地点から、重複を除外してユニークなホットスポットを抽出する関数。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム。必須カラム:
                - latitude: 緯度
                - longitude: 経度
                - ch4_ppm: メタン濃度(ppm)
                - ch4_ppm_mv: メタン濃度の移動平均(ppm)
                - ch4_ppm_delta: メタン濃度の増加量(ppm)
            min_time_threshold_seconds: float, optional
                重複とみなす最小時間差(秒)。デフォルト値は300秒(5分)。
            max_time_threshold_hours: float, optional
                別ポイントとして扱う最大時間差(時間)。デフォルト値は12時間。
            check_time_all: bool, optional
                時間閾値を超えた場合の重複チェックを継続するかどうか。デフォルト値はTrue。
            hotspot_area_meter: float, optional
                重複とみなす距離の閾値(メートル)。デフォルト値は50メートル。
            col_ch4_ppm: str, optional
                メタン濃度のカラム名。デフォルト値は"ch4_ppm"。
            col_ch4_ppm_mv: str, optional
                メタン濃度移動平均のカラム名。デフォルト値は"ch4_ppm_mv"。
            col_ch4_ppm_delta: str, optional
                メタン濃度増加量のカラム名。デフォルト値は"ch4_ppm_delta"。

        Returns
        ----------
            pd.DataFrame
                ユニークなホットスポットのデータフレーム。

        Examples
        ----------
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> df = pd.read_csv("measurement_data.csv")
        >>> unique_spots = analyzer.remove_c2c1_ratio_duplicates(
        ...     df,
        ...     min_time_threshold_seconds=300,
        ...     hotspot_area_meter=50.0
        ... )
        """
        df_data: pd.DataFrame = df.copy()
        # メタン濃度の増加が閾値を超えた点を抽出
        mask = (
            df_data[col_ch4_ppm] - df_data[col_ch4_ppm_mv] > self._ch4_enhance_threshold
        )
        hotspot_candidates = df_data[mask].copy()

        # ΔCH4の降順でソート
        sorted_hotspots = hotspot_candidates.sort_values(
            by=col_ch4_ppm_delta, ascending=False
        )
        used_positions = []
        unique_hotspots = pd.DataFrame()

        for _, spot in sorted_hotspots.iterrows():
            should_add = True
            for used_lat, used_lon, used_time in used_positions:
                # 距離チェック
                distance = geodesic(
                    (spot.latitude, spot.longitude), (used_lat, used_lon)
                ).meters

                if distance < hotspot_area_meter:
                    # 時間差の計算(秒単位)
                    time_diff = pd.Timedelta(
                        spot.name - pd.to_datetime(used_time)
                    ).total_seconds()
                    time_diff_abs = abs(time_diff)

                    # 時間差に基づく判定
                    if check_time_all:
                        # 時間に関係なく、距離が近ければ重複とみなす
                        # ΔCH4が大きい方を残す(現在のスポットは必ず小さい)
                        should_add = False
                        break
                    else:
                        # 時間窓による判定を行う
                        if time_diff_abs <= min_time_threshold_seconds:
                            # Case 1: 最小時間閾値以内は重複とみなす
                            should_add = False
                            break
                        elif time_diff_abs > max_time_threshold_hours * 3600:
                            # Case 2: 最大時間閾値を超えた場合は重複チェックをスキップ
                            continue
                        # Case 3: その間の時間差の場合は、距離が近ければ重複とみなす
                        should_add = False
                        break

            if should_add:
                unique_hotspots = pd.concat([unique_hotspots, pd.DataFrame([spot])])
                used_positions.append((spot.latitude, spot.longitude, spot.name))

        return unique_hotspots

    @staticmethod
    def remove_hotspots_duplicates(
        hotspots: list[HotspotData],
        check_time_all: bool,
        min_time_threshold_seconds: float = 300,
        max_time_threshold_hours: float = 12,
        hotspot_area_meter: float = 50,
    ) -> list[HotspotData]:
        """
        重複するホットスポットを除外します。

        このメソッドは、与えられたホットスポットのリストから重複を検出し、
        一意のホットスポットのみを返します。重複の判定は、指定された
        時間および距離の閾値に基づいて行われます。

        Parameters
        ----------
            hotspots: list[HotspotData]
                重複を除外する対象のホットスポットのリスト
            check_time_all: bool
                時間に関係なく重複チェックを行うかどうか
            min_time_threshold_seconds: float, optional
                重複とみなす最小時間の閾値(秒)。デフォルト値は300秒
            max_time_threshold_hours: float, optional
                重複チェックを一時的に無視する最大時間の閾値(時間)。デフォルト値は12時間
            hotspot_area_meter: float, optional
                重複とみなす距離の閾値(メートル)。デフォルト値は50メートル

        Returns
        ----------
            list[HotspotData]
                重複を除去したホットスポットのリスト

        Examples
        ----------
        >>> hotspots = [HotspotData(...), HotspotData(...)]  # ホットスポットのリスト
        >>> analyzer = MobileMeasurementAnalyzer()
        >>> unique_spots = analyzer.remove_hotspots_duplicates(
        ...     hotspots=hotspots,
        ...     check_time_all=True,
        ...     min_time_threshold_seconds=300,
        ...     max_time_threshold_hours=12,
        ...     hotspot_area_meter=50
        ... )
        """
        # ΔCH4の降順でソート
        sorted_hotspots: list[HotspotData] = sorted(
            hotspots, key=lambda x: x.delta_ch4, reverse=True
        )
        used_positions_by_type: dict[
            HotspotType, list[tuple[float, float, str, float]]
        ] = {
            "bio": [],
            "gas": [],
            "comb": [],
        }
        unique_hotspots: list[HotspotData] = []

        for spot in sorted_hotspots:
            is_duplicate = MobileMeasurementAnalyzer._is_duplicate_spot(
                current_lat=spot.avg_lat,
                current_lon=spot.avg_lon,
                current_time=spot.timestamp,
                used_positions=used_positions_by_type[spot.type],
                check_time_all=check_time_all,
                min_time_threshold_seconds=min_time_threshold_seconds,
                max_time_threshold_hours=max_time_threshold_hours,
                hotspot_area_meter=hotspot_area_meter,
            )

            if not is_duplicate:
                unique_hotspots.append(spot)
                used_positions_by_type[spot.type].append(
                    (spot.avg_lat, spot.avg_lon, spot.timestamp, spot.delta_ch4)
                )

        return unique_hotspots
