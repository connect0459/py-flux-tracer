import os
import math
import folium
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import get_args, Literal
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from geopy.distance import geodesic
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO
from ..commons.hotspot_data import HotspotData, HotspotType
from .correcting_utils import CorrectingUtils, CORRECTION_TYPES_PATTERN


@dataclass
class EmissionData:
    """
    ホットスポットの排出量データを格納するクラス。

    Parameters:
    ------
        source : str
            データソース（日時）
        type : HotspotType
            ホットスポットの種類（`HotspotType`を参照）
        section : str | int | float
            セクション情報
        latitude : float
            緯度
        longitude : float
            経度
        delta_ch4 : float
            CH4の増加量 (ppm)
        delta_c2h6 : float
            C2H6の増加量 (ppb)
        ratio : float
            C2H6/CH4比
        emission_rate : float
            排出量 (L/min)
        daily_emission : float
            日排出量 (L/day)
        annual_emission : float
            年間排出量 (L/year)
    """

    source: str
    type: HotspotType
    section: str | int | float
    latitude: float
    longitude: float
    delta_ch4: float
    delta_c2h6: float
    ratio: float
    emission_rate: float
    daily_emission: float
    annual_emission: float

    def __post_init__(self) -> None:
        """
        Initialize時のバリデーションを行います。

        Raises:
        ------
            ValueError: 入力値が不正な場合
        """
        # sourceのバリデーション
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Source must be a non-empty string")

        # typeのバリデーションは型システムによって保証されるため削除
        # HotspotTypeはLiteral["bio", "gas", "comb"]として定義されているため、
        # 不正な値は型チェック時に検出されます

        # sectionのバリデーション（Noneは許可）
        if self.section is not None and not isinstance(self.section, (str, int, float)):
            raise ValueError("Section must be a string, int, float, or None")

        # 緯度のバリデーション
        if (
            not isinstance(self.latitude, (int, float))
            or not -90 <= self.latitude <= 90
        ):
            raise ValueError("Latitude must be a number between -90 and 90")

        # 経度のバリデーション
        if (
            not isinstance(self.longitude, (int, float))
            or not -180 <= self.longitude <= 180
        ):
            raise ValueError("Longitude must be a number between -180 and 180")

        # delta_ch4のバリデーション
        if not isinstance(self.delta_ch4, (int, float)) or self.delta_ch4 < 0:
            raise ValueError("Delta CH4 must be a non-negative number")

        # delta_c2h6のバリデーション
        if not isinstance(self.delta_c2h6, (int, float)) or self.delta_c2h6 < 0:
            raise ValueError("Delta C2H6 must be a non-negative number")

        # ratioのバリデーション
        if not isinstance(self.ratio, (int, float)) or self.ratio < 0:
            raise ValueError("Ratio must be a non-negative number")

        # emission_rateのバリデーション
        if not isinstance(self.emission_rate, (int, float)) or self.emission_rate < 0:
            raise ValueError("Emission rate must be a non-negative number")

        # daily_emissionのバリデーション
        expected_daily = self.emission_rate * 60 * 24
        if not math.isclose(self.daily_emission, expected_daily, rel_tol=1e-10):
            raise ValueError(
                f"Daily emission ({self.daily_emission}) does not match "
                f"calculated value from emission rate ({expected_daily})"
            )

        # annual_emissionのバリデーション
        expected_annual = self.daily_emission * 365
        if not math.isclose(self.annual_emission, expected_annual, rel_tol=1e-10):
            raise ValueError(
                f"Annual emission ({self.annual_emission}) does not match "
                f"calculated value from daily emission ({expected_annual})"
            )

        # NaN値のチェック
        numeric_fields = [
            self.latitude,
            self.longitude,
            self.delta_ch4,
            self.delta_c2h6,
            self.ratio,
            self.emission_rate,
            self.daily_emission,
            self.annual_emission,
        ]
        if any(math.isnan(x) for x in numeric_fields):
            raise ValueError("Numeric fields cannot contain NaN values")

    def to_dict(self) -> dict:
        """
        データクラスの内容を辞書形式に変換します。

        Returns:
        ------
            dict: データクラスの属性と値を含む辞書
        """
        return {
            "source": self.source,
            "type": self.type,
            "section": self.section,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "delta_ch4": self.delta_ch4,
            "delta_c2h6": self.delta_c2h6,
            "ratio": self.ratio,
            "emission_rate": self.emission_rate,
            "daily_emission": self.daily_emission,
            "annual_emission": self.annual_emission,
        }


@dataclass
class MSAInputConfig:
    """入力ファイルの設定を保持するデータクラス

    Parameters:
    ------
        fs : float
            サンプリング周波数（Hz）
        lag : float
            測器の遅れ時間（秒）
        path : Path | str
            ファイルパス
        correction_type : str | None
            適用する補正式の種類を表す文字列
    """

    fs: float  # サンプリング周波数（Hz）
    lag: float  # 測器の遅れ時間（秒）
    path: Path | str  # ファイルパス
    correction_type: str | None = None  # 適用する補正式の種類を表す文字列

    def __post_init__(self) -> None:
        """
        インスタンス生成後に入力値の検証を行います。

        Raises:
        ------
            ValueError: 遅延時間が負の値である場合、またはサポートされていないファイル拡張子の場合。
        """
        # fsが有効かを確認
        if not isinstance(self.fs, (int, float)) or self.fs <= 0:
            raise ValueError(
                f"Invalid sampling frequency: {self.fs}. Must be a positive float."
            )
        # lagが0以上のfloatかを確認
        if not isinstance(self.lag, (int, float)) or self.lag < 0:
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
        # 与えられたcorrection_typeがNoneでない場合、CORRECTION_TYPES_PATTERNに含まれているかを検証します
        if self.correction_type is not None:
            if not isinstance(self.correction_type, str):
                raise ValueError(
                    f"Invalid correction_type: {self.correction_type}. Must be a str instance."
                )
            if self.correction_type not in CORRECTION_TYPES_PATTERN:
                raise ValueError(
                    f"Invalid correction_type: {self.correction_type}. Must be one of {CORRECTION_TYPES_PATTERN}."
                )

    @classmethod
    def validate_and_create(
        cls,
        fs: float,
        lag: float,
        path: Path | str,
        correction_type: str | None,
    ) -> "MSAInputConfig":
        """
        入力値を検証し、MSAInputConfigインスタンスを生成するファクトリメソッドです。

        指定された遅延時間、サンプリング周波数、およびファイルパスが有効であることを確認し、
        有効な場合に新しいMSAInputConfigオブジェクトを返します。

        Parameters:
        ------
            fs : float
                サンプリング周波数。正のfloatである必要があります。
            lag : float
                遅延時間。0以上のfloatである必要があります。
            path : Path | str
                入力ファイルのパス。サポートされている拡張子は.txtと.csvです。
            correction_type : str | None
                適用する補正式の種類を表す文字列。

        Returns:
        ------
            MSAInputConfig
                検証された入力設定を持つMSAInputConfigオブジェクト。
        """
        return cls(fs=fs, lag=lag, path=path, correction_type=correction_type)


class MobileSpatialAnalyzer:
    """
    移動観測で得られた測定データを解析するクラス
    """

    EARTH_RADIUS_METERS: float = 6371000  # 地球の半径（メートル）

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        inputs: list[MSAInputConfig] | list[tuple[float, float, str | Path]],
        num_sections: int = 4,
        ch4_enhance_threshold: float = 0.1,
        correlation_threshold: float = 0.7,
        hotspot_area_meter: float = 50,
        window_minutes: float = 5,
        column_mapping: dict[str, str] = {
            "Time Stamp": "timestamp",
            "CH4 (ppm)": "ch4_ppm",
            "C2H6 (ppb)": "c2h6_ppb",
            "H2O (ppm)": "h2o_ppm",
            "Latitude": "latitude",
            "Longitude": "longitude",
        },
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        測定データ解析クラスの初期化

        Parameters:
        ------
            center_lat : float
                中心緯度
            center_lon : float
                中心経度
            inputs : list[MSAInputConfig] | list[tuple[float, float, str | Path]]
                入力ファイルのリスト
            num_sections : int
                分割する区画数。デフォルトは4。
            ch4_enhance_threshold : float
                CH4増加の閾値(ppm)。デフォルトは0.1。
            correlation_threshold : float
                相関係数の閾値。デフォルトは0.7。
            hotspot_area_meter : float
                ホットスポットの検出に使用するエリアの半径（メートル）。デフォルトは50メートル。
            window_minutes : float
                移動窓の大きさ（分）。デフォルトは5分。
            column_mapping : dict[str, str]
                元のデータファイルのヘッダーを汎用的な単語に変換するための辞書型データ。
                - timestamp,ch4_ppm,c2h6_ppm,h2o_ppm,latitude,longitudeをvalueに、それぞれに対応するカラム名をcolに指定してください。
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug : bool
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。

        Returns:
        ------
            None
                初期化処理が完了したことを示します。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MobileSpatialAnalyzer.setup_logger(logger, log_level)
        # プライベートなプロパティ
        self._center_lat: float = center_lat
        self._center_lon: float = center_lon
        self._ch4_enhance_threshold: float = ch4_enhance_threshold
        self._correlation_threshold: float = correlation_threshold
        self._hotspot_area_meter: float = hotspot_area_meter
        self._column_mapping: dict[str, str] = column_mapping
        self._num_sections: int = num_sections
        # セクションの範囲
        section_size: float = 360 / num_sections
        self._section_size: float = section_size
        self._sections = MobileSpatialAnalyzer._initialize_sections(
            num_sections, section_size
        )
        # window_sizeをデータポイント数に変換（分→秒→データポイント数）
        self._window_size: int = MobileSpatialAnalyzer._calculate_window_size(
            window_minutes
        )
        # 入力設定の標準化
        normalized_input_configs: list[MSAInputConfig] = (
            MobileSpatialAnalyzer._normalize_inputs(inputs)
        )
        # 複数ファイルのデータを読み込み
        self._data: dict[str, pd.DataFrame] = self._load_all_data(
            normalized_input_configs
        )

    def analyze_delta_ch4_stats(self, hotspots: list[HotspotData]) -> None:
        """
        各タイプのホットスポットについてΔCH4の統計情報を計算し、結果を表示します。

        Parameters:
        ------
            hotspots : list[HotspotData]
                分析対象のホットスポットリスト

        Returns:
        ------
            None
                統計情報の表示が完了したことを示します。
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
        duplicate_check_mode: str = "none",
        min_time_threshold_seconds: float = 300,
        max_time_threshold_hours: float = 12,
    ) -> list[HotspotData]:
        """
        ホットスポットを検出して分析します。

        Parameters:
        ------
            duplicate_check_mode : str
                重複チェックのモード（"none","time_window","time_all"）。
                - "none": 重複チェックを行わない。
                - "time_window": 指定された時間窓内の重複のみを除外。
                - "time_all": すべての時間範囲で重複チェックを行う。
            min_time_threshold_seconds : float
                重複とみなす最小時間の閾値（秒）。デフォルトは300秒。
            max_time_threshold_hours : float
                重複チェックを一時的に無視する最大時間の閾値（時間）。デフォルトは12時間。

        Returns:
        ------
            list[HotspotData]
                検出されたホットスポットのリスト。
        """
        # 不正な入力値に対するエラーチェック
        valid_modes = {"none", "time_window", "time_all"}
        if duplicate_check_mode not in valid_modes:
            raise ValueError(
                f"無効な重複チェックモード: {duplicate_check_mode}. 有効な値は {valid_modes} です。"
            )

        all_hotspots: list[HotspotData] = []

        # 各データソースに対して解析を実行
        for _, df in self._data.items():
            # パラメータの計算
            df = MobileSpatialAnalyzer._calculate_hotspots_parameters(
                df, self._window_size
            )

            # ホットスポットの検出
            hotspots: list[HotspotData] = self._detect_hotspots(
                df,
                ch4_enhance_threshold=self._ch4_enhance_threshold,
            )
            all_hotspots.extend(hotspots)

        # 重複チェックモードに応じて処理
        if duplicate_check_mode != "none":
            unique_hotspots = MobileSpatialAnalyzer.remove_hotspots_duplicates(
                all_hotspots,
                check_time_all=duplicate_check_mode == "time_all",
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
        print_individual_stats: bool = True,
        print_total_stats: bool = True,
    ) -> tuple[float, timedelta]:
        """
        各ファイルの測定時間と走行距離を計算し、合計を返します。

        Parameters:
        ------
            print_individual_stats : bool
                個別ファイルの統計を表示するかどうか。デフォルトはTrue。
            print_total_stats : bool
                合計統計を表示するかどうか。デフォルトはTrue。

        Returns:
        ------
            tuple[float, timedelta]
                総距離(km)と総時間のタプル
        """
        total_distance: float = 0.0
        total_time: timedelta = timedelta()
        individual_stats: list[dict] = []  # 個別の統計情報を保存するリスト

        # プログレスバーを表示しながら計算
        for source_name, df in tqdm(
            self._data.items(), desc="Calculating", unit="file"
        ):
            # 時間の計算
            time_spent = df.index[-1] - df.index[0]

            # 距離の計算
            distance_km = 0.0
            for i in range(len(df) - 1):
                lat1, lon1 = df.iloc[i][["latitude", "longitude"]]
                lat2, lon2 = df.iloc[i + 1][["latitude", "longitude"]]
                distance_km += (
                    MobileSpatialAnalyzer._calculate_distance(
                        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
                    )
                    / 1000
                )

            # 合計に加算
            total_distance += distance_km
            total_time += time_spent

            # 統計情報を保存
            if print_individual_stats:
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
        if print_individual_stats:
            self.logger.info("=== Individual Stats ===")
            for stat in individual_stats:
                print(f"File         : {stat['source']}")
                print(f"  Distance   : {stat['distance']:.2f} km")
                print(f"  Time       : {stat['time']}")
                print(f"  Avg. Speed : {stat['speed']:.1f} km/h\n")

        # 合計を表示
        if print_total_stats:
            average_speed_total: float = total_distance / (
                total_time.total_seconds() / 3600
            )
            self.logger.info("=== Total Stats ===")
            print(f"  Distance   : {total_distance:.2f} km")
            print(f"  Time       : {total_time}")
            print(f"  Avg. Speed : {average_speed_total:.1f} km/h\n")

        return total_distance, total_time

    def create_hotspots_map(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path | None = None,
        output_filename: str = "hotspots_map.html",
        center_marker_label: str = "Center",
        plot_center_marker: bool = True,
        radius_meters: float = 3000,
        save_fig: bool = True,
    ) -> None:
        """
        ホットスポットの分布を地図上にプロットして保存

        Parameters:
        ------
            hotspots : list[HotspotData]
                プロットするホットスポットのリスト
            output_dir : str | Path
                保存先のディレクトリパス
            output_filename : str
                保存するファイル名。デフォルトは"hotspots_map"。
            center_marker_label : str
                中心を示すマーカーのラベルテキスト。デフォルトは"Center"。
            plot_center_marker : bool
                中心を示すマーカーの有無。デフォルトはTrue。
            radius_meters : float
                区画分けを示す線の長さ。デフォルトは3000。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
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
                <b>Date</b> <span>:</span> <span>{spot.source}</span>
                <b>Lat</b> <span>:</span> <span>{spot.avg_lat:.3f}</span>
                <b>Lon</b> <span>:</span> <span>{spot.avg_lon:.3f}</span>
                <b>ΔCH<sub>4</sub></b> <span>:</span> <span>{spot.delta_ch4:.3f}</span>
                <b>ΔC<sub>2</sub>H<sub>6</sub></b> <span>:</span> <span>{spot.delta_c2h6:.3f}</span>
                <b>Ratio</b> <span>:</span> <span>{spot.ratio:.3f}</span>
                <b>Type</b> <span>:</span> <span>{spot.type}</span>
                <b>Section</b> <span>:</span> <span>{spot.section}</span>
            </div>
            """

            # ポップアップのサイズを指定
            popup = folium.Popup(
                folium.Html(popup_html, script=True),
                max_width=200,  # 最大幅（ピクセル）
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
                icon=folium.Icon(color="green", icon="info-sign"),
            ).add_to(m)

        # 区画の境界線を描画
        for section in range(self._num_sections):
            start_angle = math.radians(-180 + section * self._section_size)

            R = self.EARTH_RADIUS_METERS

            # 境界線の座標を計算
            lat1 = self._center_lat
            lon1 = self._center_lon
            lat2 = math.degrees(
                math.asin(
                    math.sin(math.radians(lat1)) * math.cos(radius_meters / R)
                    + math.cos(math.radians(lat1))
                    * math.sin(radius_meters / R)
                    * math.cos(start_angle)
                )
            )
            lon2 = self._center_lon + math.degrees(
                math.atan2(
                    math.sin(start_angle)
                    * math.sin(radius_meters / R)
                    * math.cos(math.radians(lat1)),
                    math.cos(radius_meters / R)
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
        if save_fig and output_dir is None:
            raise ValueError(
                "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
            )
            output_path: str = os.path.join(output_dir, output_filename)
            m.save(str(output_path))
            self.logger.info(f"地図を保存しました: {output_path}")

    def export_hotspots_to_csv(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path | None = None,
        output_filename: str = "hotspots.csv",
    ) -> None:
        """
        ホットスポットの情報をCSVファイルに出力します。

        Parameters:
        ------
            hotspots : list[HotspotData]
                出力するホットスポットのリスト
            output_dir : str | Path | None
                出力先ディレクトリ
            output_filename : str
                出力ファイル名
        """
        # 日時の昇順でソート
        sorted_hotspots = sorted(hotspots, key=lambda x: x.source)

        # 出力用のデータを作成
        records = []
        for spot in sorted_hotspots:
            record = {
                "source": spot.source,
                "type": spot.type,
                "delta_ch4": spot.delta_ch4,
                "delta_c2h6": spot.delta_c2h6,
                "ratio": spot.ratio,
                "correlation": spot.correlation,
                "angle": spot.angle,
                "section": spot.section,
                "latitude": spot.avg_lat,
                "longitude": spot.avg_lon,
            }
            records.append(record)

        # DataFrameに変換してCSVに出力
        if output_dir is None:
            raise ValueError(
                "output_dirが指定されていません。有効なディレクトリパスを指定してください。"
            )
        output_path: str = os.path.join(output_dir, output_filename)
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        self.logger.info(
            f"ホットスポット情報をCSVファイルに出力しました: {output_path}"
        )

    def get_preprocessed_data(
        self,
    ) -> pd.DataFrame:
        """
        データ前処理を行い、CH4とC2H6の相関解析に必要な形式に整えます。
        コンストラクタで読み込んだすべてのデータを前処理し、結合したDataFrameを返します。

        Returns:
        ------
            pd.DataFrame
                前処理済みの結合されたDataFrame
        """
        processed_dfs: list[pd.DataFrame] = []

        # 各データソースに対して解析を実行
        for source_name, df in self._data.items():
            # パラメータの計算
            processed_df = MobileSpatialAnalyzer._calculate_hotspots_parameters(
                df, self._window_size
            )
            # ソース名を列として追加
            processed_df["source"] = source_name
            processed_dfs.append(processed_df)

        # すべてのDataFrameを結合
        if not processed_dfs:
            raise ValueError("処理対象のデータが存在しません。")

        combined_df = pd.concat(processed_dfs, axis=0)
        return combined_df

    def get_section_size(self) -> float:
        """
        セクションのサイズを取得するメソッド。
        このメソッドは、解析対象のデータを区画に分割する際の
        各区画の角度範囲を示すサイズを返します。

        Returns:
        ------
            float
                1セクションのサイズ（度単位）
        """
        return self._section_size

    def plot_ch4_delta_histogram(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path | None,
        output_filename: str = "ch4_delta_histogram.png",
        dpi: int = 200,
        figsize: tuple[int, int] = (8, 6),
        fontsize: float = 20,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        save_fig: bool = True,
        show_fig: bool = True,
        yscale_log: bool = True,
        print_bins_analysis: bool = False,
    ) -> None:
        """
        CH4の増加量（ΔCH4）の積み上げヒストグラムをプロットします。

        Parameters:
        ------
            hotspots : list[HotspotData]
                プロットするホットスポットのリスト
            output_dir : str | Path | None
                保存先のディレクトリパス
            output_filename : str
                保存するファイル名。デフォルトは"ch4_delta_histogram.png"。
            dpi : int
                解像度。デフォルトは200。
            figsize : tuple[int, int]
                図のサイズ。デフォルトは(8, 6)。
            fontsize : float
                フォントサイズ。デフォルトは20。
            xlim : tuple[float, float] | None
                x軸の範囲。Noneの場合は自動設定。
            ylim : tuple[float, float] | None
                y軸の範囲。Noneの場合は自動設定。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
            show_fig : bool
                図の表示を許可するフラグ。デフォルトはTrue。
            yscale_log : bool
                y軸をlogにするかどうか。デフォルトはTrue。
            print_bins_analysis : bool
                ビンごとの内訳を表示するオプション。
        """
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

        # 色の定義をHotspotTypeを使用して型安全に定義
        colors: dict[HotspotType, str] = {"bio": "blue", "gas": "red", "comb": "green"}

        # HotspotTypeのリテラル値を使用してイテレーション
        for type_name in get_args(HotspotType):
            if type_name in hist_data:
                plt.bar(
                    bins[:-1],
                    hist_data[type_name],
                    width=np.diff(bins)[0],
                    bottom=bottom,
                    color=colors[type_name],
                    label=type_name,
                    alpha=0.6,
                    align="edge",
                )
                bottom += hist_data[type_name]

        if yscale_log:
            plt.yscale("log")
        plt.xlabel("Δ$\\mathregular{CH_{4}}$ (ppm)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # 軸の範囲を設定
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # グラフの保存または表示
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dir, exist_ok=True)
            output_path: str = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            self.logger.info(f"ヒストグラムを保存しました: {output_path}")
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def plot_mapbox(
        self,
        df: pd.DataFrame,
        col: str,
        mapbox_access_token: str,
        sort_value_column: bool = True,
        output_dir: str | Path | None = None,
        output_filename: str = "mapbox_plot.html",
        lat_column: str = "latitude",
        lon_column: str = "longitude",
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
        Plotlyを使用してMapbox上にデータをプロットします。

        Parameters:
        ------
            df : pd.DataFrame
                プロットするデータを含むDataFrame
            col : str
                カラーマッピングに使用する列名
            mapbox_access_token : str
                Mapboxのアクセストークン
            sort_value_column : bool
                value_columnをソートするか否か。デフォルトはTrue。
            output_dir : str | Path | None
                出力ディレクトリのパス
            output_filename : str
                出力ファイル名。デフォルトは"mapbox_plot.html"
            lat_column : str
                緯度の列名。デフォルトは"latitude"
            lon_column : str
                経度の列名。デフォルトは"longitude"
            colorscale : str
                使用するカラースケール。デフォルトは"Jet"
            center_lat : float | None
                中心緯度。デフォルトはNoneで、self._center_latを使用
            center_lon : float | None
                中心経度。デフォルトはNoneで、self._center_lonを使用
            zoom : float
                マップの初期ズームレベル。デフォルトは12
            width : int
                プロットの幅（ピクセル）。デフォルトは700
            height : int
                プロットの高さ（ピクセル）。デフォルトは700
            tick_font_family : str
                カラーバーの目盛りフォントファミリー。デフォルトは"Arial"
            title_font_family : str
                カラーバーのラベルフォントファミリー。デフォルトは"Arial"
            tick_font_size : int
                カラーバーの目盛りフォントサイズ。デフォルトは12
            title_font_size : int
                カラーバーのラベルフォントサイズ。デフォルトは14
            marker_size : int
                マーカーのサイズ。デフォルトは4
            colorbar_title : str | None
                カラーバーのラベル
            value_range : tuple[float, float] | None
                カラーマッピングの範囲。デフォルトはNoneで、データの最小値と最大値を使用
            save_fig : bool
                図を保存するかどうか。デフォルトはTrue
            show_fig : bool
                図を表示するかどうか。デフォルトはTrue
        """
        df_mapping: pd.DataFrame = df.copy().dropna(subset=[col])
        if sort_value_column:
            df_mapping = df_mapping.sort_values(col)
        # 中心座標の設定
        center_lat = center_lat if center_lat is not None else self._center_lat
        center_lon = center_lon if center_lon is not None else self._center_lon

        # カラーマッピングの範囲を設定
        cmin, cmax = 0, 0
        if value_range is None:
            cmin = df_mapping[col].min()
            cmax = df_mapping[col].max()
        else:
            cmin, cmax = value_range

        # カラーバーのタイトルを設定
        title_text = colorbar_title if colorbar_title is not None else col

        # Scattermapboxのデータを作成
        scatter_data = go.Scattermapbox(
            lat=df_mapping[lat_column],
            lon=df_mapping[lon_column],
            text=df_mapping[col].astype(str),
            hoverinfo="text",
            mode="markers",
            marker=dict(
                color=df_mapping[col],
                size=marker_size,
                reversescale=False,
                autocolorscale=False,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                colorbar=dict(
                    tickformat="3.2f",
                    outlinecolor="black",
                    outlinewidth=1.5,
                    ticks="outside",
                    ticklen=7,
                    tickwidth=1.5,
                    tickcolor="black",
                    tickfont=dict(
                        family=tick_font_family, color="black", size=tick_font_size
                    ),
                    title=dict(
                        text=title_text, side="top"
                    ),  # カラーバーのタイトルを設定
                    titlefont=dict(
                        family=title_font_family,
                        color="black",
                        size=title_font_size,
                    ),
                ),
            ),
        )

        # レイアウトの設定
        layout = go.Layout(
            width=width,
            height=height,
            showlegend=False,
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
            ),
        )

        # 図の作成
        fig = go.Figure(data=[scatter_data], layout=layout)

        # 図の保存
        if save_fig:
            # 保存時の出力ディレクトリチェック
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。"
                )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            pyo.plot(fig, filename=output_path, auto_open=False)
            self.logger.info(f"Mapboxプロットを保存しました: {output_path}")
        # 図の表示
        if show_fig:
            pyo.iplot(fig)

    def plot_scatter_c2c1(
        self,
        hotspots: list[HotspotData],
        output_dir: str | Path | None = None,
        output_filename: str = "scatter_c2c1.png",
        dpi: int = 200,
        figsize: tuple[int, int] = (4, 4),
        fontsize: float = 12,
        save_fig: bool = True,
        show_fig: bool = True,
        ratio_labels: dict[float, tuple[float, float, str]] | None = None,
    ) -> None:
        """
        検出されたホットスポットのΔC2H6とΔCH4の散布図をプロットします。

        Parameters:
        ------
            hotspots : list[HotspotData]
                プロットするホットスポットのリスト
            output_dir : str | Path | None
                保存先のディレクトリパス
            output_filename : str
                保存するファイル名。デフォルトは"scatter_c2c1.png"。
            dpi : int
                解像度。デフォルトは200。
            figsize : tuple[int, int]
                図のサイズ。デフォルトは(4, 4)。
            fontsize : float
                フォントサイズ。デフォルトは12。
            save_fig : bool
                図の保存を許可するフラグ。デフォルトはTrue。
            show_fig : bool
                図の表示を許可するフラグ。デフォルトはTrue。
            ratio_labels : dict[float, tuple[float, float, str]] | None
                比率線とラベルの設定。
                キーは比率値、値は (x位置, y位置, ラベルテキスト) のタプル。
                Noneの場合はデフォルト設定を使用。デフォルト値:
                {
                    0.001: (1.25, 2, "0.001"),
                    0.005: (1.25, 8, "0.005"),
                    0.010: (1.25, 15, "0.01"),
                    0.020: (1.25, 30, "0.02"),
                    0.030: (1.0, 40, "0.03"),
                    0.076: (0.20, 42, "0.076 (Osaka)")
                }
        """
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

        # 色とラベルの定義
        colors: dict[HotspotType, str] = {"bio": "blue", "gas": "red", "comb": "green"}
        labels: dict[HotspotType, str] = {"bio": "bio", "gas": "gas", "comb": "comb"}

        # タイプごとにプロット（データが存在する場合のみ）
        for spot_type, data in type_data.items():
            if data:  # データが存在する場合のみプロット
                ch4_values, c2h6_values = zip(*data)
                plt.plot(
                    ch4_values,
                    c2h6_values,
                    "o",
                    c=colors[spot_type],
                    alpha=0.5,
                    ms=2,
                    label=labels[spot_type],
                )

        # デフォルトの比率とラベル設定
        default_ratio_labels = {
            0.001: (1.25, 2, "0.001"),
            0.005: (1.25, 8, "0.005"),
            0.010: (1.25, 15, "0.01"),
            0.020: (1.25, 30, "0.02"),
            0.030: (1.0, 40, "0.03"),
            0.076: (0.20, 42, "0.076 (Osaka)"),
        }

        ratio_labels = ratio_labels or default_ratio_labels

        # プロット後、軸の設定前に比率の線を追加
        x = np.array([0, 5])
        base_ch4 = 0.0
        base = 0.0

        # 各比率に対して線を引く
        for ratio, (x_pos, y_pos, label) in ratio_labels.items():
            y = (x - base_ch4) * 1000 * ratio + base
            plt.plot(x, y, "-", c="black", alpha=0.5)
            plt.text(x_pos, y_pos, label)

        plt.ylim(0, 50)
        plt.xlim(0, 2.0)
        plt.ylabel("Δ$\\mathregular{C_{2}H_{6}}$ (ppb)")
        plt.xlabel("Δ$\\mathregular{CH_{4}}$ (ppm)")
        plt.legend()

        # グラフの保存または表示
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            output_path: str = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            self.logger.info(f"散布図を保存しました: {output_path}")
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def plot_timeseries(
        self,
        dpi: int = 200,
        source_name: str | None = None,
        figsize: tuple[float, float] = (8, 4),
        output_dir: str | Path | None = None,
        output_filename: str = "timeseries.png",
        save_fig: bool = False,
        show_fig: bool = True,
        col_ch4: str = "ch4_ppm",
        col_c2h6: str = "c2h6_ppb",
        col_h2o: str = "h2o_ppm",
        ylim_ch4: tuple[float, float] | None = None,
        ylim_c2h6: tuple[float, float] | None = None,
        ylim_h2o: tuple[float, float] | None = None,
    ) -> None:
        """
        時系列データをプロットします。

        Parameters:
        ------
            dpi : int
                図の解像度を指定します。デフォルトは200です。
            source_name : str | None
                プロットするデータソースの名前。Noneの場合は最初のデータソースを使用します。
            figsize : tuple[float, float]
                図のサイズを指定します。デフォルトは(8, 4)です。
            output_dir : str | Path | None
                保存先のディレクトリを指定します。save_fig=Trueの場合は必須です。
            output_filename : str
                保存するファイル名を指定します。デフォルトは"time_series.png"です。
            save_fig : bool
                図を保存するかどうかを指定します。デフォルトはFalseです。
            show_fig : bool
                図を表示するかどうかを指定します。デフォルトはTrueです。
            col_ch4 : str
                CH4データのキーを指定します。デフォルトは"ch4_ppm"です。
            col_c2h6 : str
                C2H6データのキーを指定します。デフォルトは"c2h6_ppb"です。
            col_h2o : str
                H2Oデータのキーを指定します。デフォルトは"h2o_ppm"です。
            ylim_ch4 : tuple[float, float] | None
                CH4プロットのy軸範囲を指定します。デフォルトはNoneです。
            ylim_c2h6 : tuple[float, float] | None
                C2H6プロットのy軸範囲を指定します。デフォルトはNoneです。
            ylim_h2o : tuple[float, float] | None
                H2Oプロットのy軸範囲を指定します。デフォルトはNoneです。
        """
        dfs_dict: dict[str, pd.DataFrame] = self._data.copy()
        # データソースの選択
        if not dfs_dict:
            raise ValueError("データが読み込まれていません。")

        if source_name is None:
            source_name = list(dfs_dict.keys())[0]
        elif source_name not in dfs_dict:
            raise ValueError(
                f"指定されたデータソース '{source_name}' が見つかりません。"
            )

        df = dfs_dict[source_name]

        # プロットの作成
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # CH4プロット
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(df.index, df[col_ch4], c="red")
        if ylim_ch4:
            ax1.set_ylim(ylim_ch4)
        ax1.set_ylabel("$\\mathregular{CH_{4}}$ (ppm)")
        ax1.grid(True, alpha=0.3)

        # C2H6プロット
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(df.index, df[col_c2h6], c="red")
        if ylim_c2h6:
            ax2.set_ylim(ylim_c2h6)
        ax2.set_ylabel("$\\mathregular{C_{2}H_{6}}$ (ppb)")
        ax2.grid(True, alpha=0.3)

        # H2Oプロット
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(df.index, df[col_h2o], c="red")
        if ylim_h2o:
            ax3.set_ylim(ylim_h2o)
        ax3.set_ylabel("$\\mathregular{H_{2}O}$ (ppm)")
        ax3.grid(True, alpha=0.3)

        # x軸のフォーマット調整
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        plt.subplots_adjust(wspace=0.38, hspace=0.38)

        # 図の保存
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            self.logger.info(f"時系列プロットを保存しました: {output_path}")

        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

    def _detect_hotspots(
        self,
        df: pd.DataFrame,
        ch4_enhance_threshold: float,
    ) -> list[HotspotData]:
        """
        シンプル化したホットスポット検出

        Parameters:
        ------
            df : pd.DataFrame
                入力データフレーム
            ch4_enhance_threshold : float
                CH4増加の閾値

        Returns:
        ------
            list[HotspotData]
                検出されたホットスポットのリスト
        """
        hotspots: list[HotspotData] = []

        # CH4増加量が閾値を超えるデータポイントを抽出
        enhanced_mask = df["ch4_ppm_delta"] >= ch4_enhance_threshold

        if enhanced_mask.any():
            lat = df["latitude"][enhanced_mask]
            lon = df["longitude"][enhanced_mask]
            ratios = df["c2c1_ratio_delta"][enhanced_mask]
            delta_ch4 = df["ch4_ppm_delta"][enhanced_mask]
            delta_c2h6 = df["c2h6_ppb_delta"][enhanced_mask]

            # 各ポイントに対してホットスポットを作成
            for i in range(len(lat)):
                if pd.notna(ratios.iloc[i]):
                    current_lat = lat.iloc[i]
                    current_lon = lon.iloc[i]
                    correlation = df["ch4_c2h6_correlation"].iloc[i]

                    # 比率に基づいてタイプを決定
                    spot_type: HotspotType = "bio"
                    if ratios.iloc[i] >= 100:
                        spot_type = "comb"
                    elif ratios.iloc[i] >= 5:
                        spot_type = "gas"

                    angle: float = MobileSpatialAnalyzer._calculate_angle(
                        lat=current_lat,
                        lon=current_lon,
                        center_lat=self._center_lat,
                        center_lon=self._center_lon,
                    )
                    section: int = self._determine_section(angle)

                    hotspots.append(
                        HotspotData(
                            source=ratios.index[i].strftime("%Y-%m-%d %H:%M:%S"),
                            angle=angle,
                            avg_lat=current_lat,
                            avg_lon=current_lon,
                            delta_ch4=delta_ch4.iloc[i],
                            delta_c2h6=delta_c2h6.iloc[i],
                            correlation=max(-1, min(1, correlation)),
                            ratio=ratios.iloc[i],
                            section=section,
                            type=spot_type,
                        )
                    )

        return hotspots

    def _determine_section(self, angle: float) -> int:
        """
        角度に基づいて所属する区画を特定します。

        Parameters:
        ------
            angle : float
                計算された角度

        Returns:
        ------
            int
                区画番号（0-based-index）
        """
        for section_num, (start, end) in self._sections.items():
            if start <= angle < end:
                return section_num
        # -180度の場合は最後の区画に含める
        return self._num_sections - 1

    def _load_all_data(
        self, input_configs: list[MSAInputConfig]
    ) -> dict[str, pd.DataFrame]:
        """
        全入力ファイルのデータを読み込み、データフレームの辞書を返します。

        このメソッドは、指定された入力設定に基づいてすべてのデータファイルを読み込み、
        各ファイルのデータをデータフレームとして格納した辞書を生成します。

        Parameters:
        ------
            input_configs : list[MSAInputConfig]
                読み込むファイルの設定リスト。

        Returns:
        ------
            dict[str, pd.DataFrame]
                読み込まれたデータフレームの辞書。キーはファイル名、値はデータフレーム。
        """
        all_data: dict[str, pd.DataFrame] = {}
        for config in input_configs:
            df, source_name = self._load_data(config)
            all_data[source_name] = df
        return all_data

    def _load_data(self, config: MSAInputConfig) -> tuple[pd.DataFrame, str]:
        """
        測定データを読み込み、前処理を行うメソッド。

        Parameters:
        ------
            config : MSAInputConfig
                入力ファイルの設定を含むオブジェクト。

        Returns:
        ------
            tuple[pd.DataFrame, str]
                読み込まれたデータフレームとそのソース名を含むタプル。
        """
        source_name: str = Path(config.path).stem
        df: pd.DataFrame = pd.read_csv(config.path, na_values=["No Data", "nan"])

        # カラム名の標準化（測器に依存しない汎用的な名前に変更）
        df = df.rename(columns=self._column_mapping)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # インデックスを設定（元のtimestampカラムは保持）
        df = df.set_index("timestamp", drop=False)

        # 緯度経度のnanを削除
        df = df.dropna(subset=["latitude", "longitude"])

        if config.lag < 0:
            raise ValueError(
                f"Invalid lag value: {config.lag}. Must be a non-negative float."
            )

        # 遅れ時間の補正
        columns_to_shift: list[str] = ["ch4_ppm", "c2h6_ppb", "h2o_ppm"]
        # サンプリング周波数に応じてシフト量を調整
        shift_periods: float = -config.lag * config.fs  # fsを掛けて補正

        for col in columns_to_shift:
            df[col] = df[col].shift(shift_periods)

        df = df.dropna(subset=columns_to_shift)

        # 水蒸気干渉などの補正式を適用
        if config.correction_type is not None:
            df = CorrectingUtils.correct_df_by_type(df, config.correction_type)
        else:
            self.logger.warn(
                f"'correction_type' is None, so no correction functions will be applied. Source: {source_name}"
            )

        return df, source_name

    @staticmethod
    def _calculate_angle(
        lat: float, lon: float, center_lat: float, center_lon: float
    ) -> float:
        """
        中心からの角度を計算

        Parameters:
        ------
            lat : float
                対象地点の緯度
            lon : float
                対象地点の経度
            center_lat : float
                中心の緯度
            center_lon : float
                中心の経度

        Returns:
        ------
            float
                真北を0°として時計回りの角度（-180°から180°）
        """
        d_lat: float = lat - center_lat
        d_lon: float = lon - center_lon
        # arctanを使用して角度を計算（ラジアン）
        angle_rad: float = math.atan2(d_lon, d_lat)
        # ラジアンから度に変換（-180から180の範囲）
        angle_deg: float = math.degrees(angle_rad)
        return angle_deg

    @classmethod
    def _calculate_distance(
        cls, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        2点間の距離をメートル単位で計算（Haversine formula）

        Parameters:
        ------
            lat1 : float
                地点1の緯度
            lon1 : float
                地点1の経度
            lat2 : float
                地点2の緯度
            lon2 : float
                地点2の経度

        Returns:
        ------
            float
                2地点間の距離（メートル）
        """
        R = cls.EARTH_RADIUS_METERS

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

        return R * c  # メートル単位での距離

    @staticmethod
    def _calculate_hotspots_parameters(
        df: pd.DataFrame,
        window_size: int,
        col_ch4_ppm: str = "ch4_ppm",
        col_c2h6_ppb: str = "c2h6_ppb",
        ch4_threshold: float = 0.05,
        c2h6_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        ホットスポットのパラメータを計算します。
        このメソッドは、指定されたデータフレームに対して移動平均や相関を計算し、
        各種のデルタ値や比率を追加します。これにより、ホットスポットの分析に必要な
        パラメータを整形します。

        Parameters:
        ------
            df : pd.DataFrame
                入力データフレーム
            window_size : int
                移動窓のサイズ
            col_ch4_ppm : str
                CH4濃度を示すカラム名
            col_c2h6_ppb : str
                C2H6濃度を示すカラム名
            ch4_threshold : float
                CH4の閾値
            c2h6_threshold : float
                C2H6の閾値

        Returns:
        ------
            pd.DataFrame
                計算されたパラメータを含むデータフレーム
        """
        # 移動平均の計算
        df["ch4_ppm_mv"] = (
            df[col_ch4_ppm]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )
        df["c2h6_ppb_mv"] = (
            df[col_c2h6_ppb]
            .rolling(window=window_size, center=True, min_periods=1)
            .mean()
        )

        # 移動相関の計算
        df["ch4_c2h6_correlation"] = (
            df[col_ch4_ppm]
            .rolling(window=window_size, min_periods=1)
            .corr(df[col_c2h6_ppb])
        )

        # 移動平均からの偏差
        df["ch4_ppm_delta"] = df[col_ch4_ppm] - df["ch4_ppm_mv"]
        df["c2h6_ppb_delta"] = df[col_c2h6_ppb] - df["c2h6_ppb_mv"]

        # C2H6/CH4の比率計算
        df["c2c1_ratio"] = df[col_c2h6_ppb] / df[col_ch4_ppm]

        # デルタ値に基づく比の計算
        df["c2c1_ratio_delta"] = np.where(
            (df["ch4_ppm_delta"].abs() >= ch4_threshold)
            & (df["c2h6_ppb_delta"] >= c2h6_threshold),
            df["c2h6_ppb_delta"] / df["ch4_ppm_delta"],
            np.nan,
        )

        return df

    @staticmethod
    def _calculate_window_size(window_minutes: float) -> int:
        """
        時間窓からデータポイント数を計算

        Parameters:
        ------
            window_minutes : float
                時間窓の大きさ（分）

        Returns:
        ------
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

        Parameters:
        ------
            num_sections : int
                初期化する区画の数。
            section_size : float
                各区画の角度範囲のサイズ。

        Returns:
        ------
            dict[int, tuple[float, float]]
                区画番号（0-based-index）とその範囲の辞書。各区画は-180度から180度の範囲に分割されます。
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

        Parameters:
        ------
            current_lat : float
                判定する地点の緯度
            current_lon : float
                判定する地点の経度
            current_time : str
                判定する地点の時刻
            used_positions : list[tuple[float, float, str, float]]
                既存の地点情報のリスト (lat, lon, time, value)
            check_time_all : bool
                時間に関係なく重複チェックを行うかどうか
            min_time_threshold_seconds : float
                重複とみなす最小時間の閾値（秒）
            max_time_threshold_hours : float
                重複チェックを一時的に無視する最大時間の閾値（時間）
            hotspot_area_meter : float
                重複とみなす距離の閾値（m）

        Returns:
        ------
            bool
                重複している場合はTrue、そうでない場合はFalse
        """
        for used_lat, used_lon, used_time, _ in used_positions:
            # 距離チェック
            distance = MobileSpatialAnalyzer._calculate_distance(
                lat1=current_lat, lon1=current_lon, lat2=used_lat, lon2=used_lon
            )

            if distance < hotspot_area_meter:
                # 時間差の計算（秒単位）
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
    def _normalize_inputs(
        inputs: list[MSAInputConfig] | list[tuple[float, float, str | Path]],
    ) -> list[MSAInputConfig]:
        """
        入力設定を標準化

        Parameters:
        ------
            inputs : list[MSAInputConfig] | list[tuple[float, float, str | Path]]
                入力設定のリスト

        Returns:
        ------
            list[MSAInputConfig]
                標準化された入力設定のリスト
        """
        normalized: list[MSAInputConfig] = []
        for inp in inputs:
            if isinstance(inp, MSAInputConfig):
                normalized.append(inp)  # すでに検証済みのため、そのまま追加
            else:
                fs, lag, path = inp
                normalized.append(
                    MSAInputConfig.validate_and_create(fs=fs, lag=lag, path=path)
                )
        return normalized

    def remove_c2c1_ratio_duplicates(
        self,
        df: pd.DataFrame,
        min_time_threshold_seconds: float = 300,  # 5分以内は重複とみなす
        max_time_threshold_hours: float = 12.0,  # 12時間以上離れている場合は別のポイントとして扱う
        check_time_all: bool = True,  # 時間閾値を超えた場合の重複チェックを継続するかどうか
        hotspot_area_meter: float = 50.0,  # 重複とみなす距離の閾値（メートル）
        col_ch4_ppm: str = "ch4_ppm",
        col_ch4_ppm_mv: str = "ch4_ppm_mv",
        col_ch4_ppm_delta: str = "ch4_ppm_delta",
    ):
        """
        メタン濃度の増加が閾値を超えた地点から、重複を除外してユニークなホットスポットを抽出する関数。

        Parameters:
        ------
            df : pandas.DataFrame
                入力データフレーム。必須カラム:
                - ch4_ppm: メタン濃度（ppm）
                - ch4_ppm_mv: メタン濃度の移動平均（ppm）
                - ch4_ppm_delta: メタン濃度の増加量（ppm）
                - latitude: 緯度
                - longitude: 経度
            min_time_threshold_seconds : float, optional
                重複とみなす最小時間差（秒）。デフォルトは300秒（5分）。
            max_time_threshold_hours : float, optional
                別ポイントとして扱う最大時間差（時間）。デフォルトは12時間。
            check_time_all : bool, optional
                時間閾値を超えた場合の重複チェックを継続するかどうか。デフォルトはTrue。
            hotspot_area_meter : float, optional
                重複とみなす距離の閾値（メートル）。デフォルトは50メートル。

        Returns:
        ------
            pandas.DataFrame
                ユニークなホットスポットのデータフレーム。
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
                    # 時間差の計算（秒単位）
                    time_diff = pd.Timedelta(
                        spot.name - pd.to_datetime(used_time)
                    ).total_seconds()
                    time_diff_abs = abs(time_diff)

                    # 時間差に基づく判定
                    if check_time_all:
                        # 時間に関係なく、距離が近ければ重複とみなす
                        # ΔCH4が大きい方を残す（現在のスポットは必ず小さい）
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

        Parameters:
        ------
            hotspots : list[HotspotData]
                重複を除外する対象のホットスポットのリスト。
            check_time_all : bool
                時間に関係なく重複チェックを行うかどうか。
            min_time_threshold_seconds : float
                重複とみなす最小時間の閾値（秒）。
            max_time_threshold_hours : float
                重複チェックを一時的に無視する最大時間の閾値（時間）。
            hotspot_area_meter : float
                重複とみなす距離の閾値（メートル）。

        Returns:
        ------
            list[HotspotData]
                重複を除去したホットスポットのリスト。
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
            is_duplicate = MobileSpatialAnalyzer._is_duplicate_spot(
                current_lat=spot.avg_lat,
                current_lon=spot.avg_lon,
                current_time=spot.source,
                used_positions=used_positions_by_type[spot.type],
                check_time_all=check_time_all,
                min_time_threshold_seconds=min_time_threshold_seconds,
                max_time_threshold_hours=max_time_threshold_hours,
                hotspot_area_meter=hotspot_area_meter,
            )

            if not is_duplicate:
                unique_hotspots.append(spot)
                used_positions_by_type[spot.type].append(
                    (spot.avg_lat, spot.avg_lon, spot.source, spot.delta_ch4)
                )

        return unique_hotspots

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
    def calculate_emission_rates(
        hotspots: list[HotspotData],
        method: Literal["weller", "weitzel", "joo", "umezawa"] = "weller",
        print_summary: bool = True,
        custom_formulas: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[EmissionData], dict[str, dict[str, float]]]:
        """
        検出されたホットスポットのCH4漏出量を計算・解析し、統計情報を生成します。

        Parameters:
        ------
            hotspots : list[HotspotData]
                分析対象のホットスポットのリスト
            method : Literal["weller", "weitzel", "joo", "umezawa"]
                使用する計算式。デフォルトは"weller"。
            print_summary : bool
                統計情報を表示するかどうか。デフォルトはTrue。
            custom_formulas : dict[str, dict[str, float]] | None
                カスタム計算式の係数。
                例: {"custom_method": {"a": 1.0, "b": 1.0}}
                Noneの場合はデフォルトの計算式を使用。

        Returns:
        ------
            tuple[list[EmissionData], dict[str, dict[str, float]]]
                - 各ホットスポットの排出量データを含むリスト
                - タイプ別の統計情報を含む辞書
        """
        # デフォルトの経験式係数
        default_formulas = {
            "weller": {"a": 0.988, "b": 0.817},
            "weitzel": {"a": 0.521, "b": 0.795},
            "joo": {"a": 2.738, "b": 1.329},
            "umezawa": {"a": 2.716, "b": 0.741},
        }

        # カスタム計算式がある場合は追加
        emission_formulas = default_formulas.copy()
        if custom_formulas:
            emission_formulas.update(custom_formulas)

        if method not in emission_formulas:
            raise ValueError(f"Unknown method: {method}")

        # 係数の取得
        a = emission_formulas[method]["a"]
        b = emission_formulas[method]["b"]

        # 排出量の計算
        emission_data_list = []
        for spot in hotspots:
            # 漏出量の計算 (L/min)
            emission_rate = np.exp((np.log(spot.delta_ch4) + a) / b)
            # 日排出量 (L/day)
            daily_emission = emission_rate * 60 * 24
            # 年間排出量 (L/year)
            annual_emission = daily_emission * 365

            emission_data = EmissionData(
                source=spot.source,
                type=spot.type,
                section=spot.section,
                latitude=spot.avg_lat,
                longitude=spot.avg_lon,
                delta_ch4=spot.delta_ch4,
                delta_c2h6=spot.delta_c2h6,
                ratio=spot.ratio,
                emission_rate=emission_rate,
                daily_emission=daily_emission,
                annual_emission=annual_emission,
            )
            emission_data_list.append(emission_data)

        # 統計計算用にDataFrameを作成
        emission_df = pd.DataFrame([e.to_dict() for e in emission_data_list])

        # タイプ別の統計情報を計算
        stats = {}
        # emission_formulas の定義の後に、排出量カテゴリーの閾値を定義
        emission_categories = {
            "low": {"min": 0, "max": 6},  # < 6 L/min
            "medium": {"min": 6, "max": 40},  # 6-40 L/min
            "high": {"min": 40, "max": float("inf")},  # > 40 L/min
        }
        # get_args(HotspotType)を使用して型安全なリストを作成
        types = list(get_args(HotspotType))
        for spot_type in types:
            df_type = emission_df[emission_df["type"] == spot_type]
            if len(df_type) > 0:
                # 既存の統計情報を計算
                type_stats = {
                    "count": len(df_type),
                    "emission_rate_min": df_type["emission_rate"].min(),
                    "emission_rate_max": df_type["emission_rate"].max(),
                    "emission_rate_mean": df_type["emission_rate"].mean(),
                    "emission_rate_median": df_type["emission_rate"].median(),
                    "total_annual_emission": df_type["annual_emission"].sum(),
                    "mean_annual_emission": df_type["annual_emission"].mean(),
                }

                # 排出量カテゴリー別の統計を追加
                category_counts = {
                    "low": len(
                        df_type[
                            df_type["emission_rate"] < emission_categories["low"]["max"]
                        ]
                    ),
                    "medium": len(
                        df_type[
                            (
                                df_type["emission_rate"]
                                >= emission_categories["medium"]["min"]
                            )
                            & (
                                df_type["emission_rate"]
                                < emission_categories["medium"]["max"]
                            )
                        ]
                    ),
                    "high": len(
                        df_type[
                            df_type["emission_rate"]
                            >= emission_categories["high"]["min"]
                        ]
                    ),
                }
                type_stats["emission_categories"] = category_counts

                stats[spot_type] = type_stats

                if print_summary:
                    print(f"\n{spot_type}タイプの統計情報:")
                    print(f"  検出数: {type_stats['count']}")
                    print("  排出量 (L/min):")
                    print(f"    最小値: {type_stats['emission_rate_min']:.2f}")
                    print(f"    最大値: {type_stats['emission_rate_max']:.2f}")
                    print(f"    平均値: {type_stats['emission_rate_mean']:.2f}")
                    print(f"    中央値: {type_stats['emission_rate_median']:.2f}")
                    print("  排出量カテゴリー別の検出数:")
                    print(f"    低放出 (< 6 L/min): {category_counts['low']}")
                    print(f"    中放出 (6-40 L/min): {category_counts['medium']}")
                    print(f"    高放出 (> 40 L/min): {category_counts['high']}")
                    print("  年間排出量 (L/year):")
                    print(f"    合計: {type_stats['total_annual_emission']:.2f}")
                    print(f"    平均: {type_stats['mean_annual_emission']:.2f}")

        return emission_data_list, stats

    @staticmethod
    def plot_emission_analysis(
        emission_data_list: list[EmissionData],
        dpi: int = 300,
        output_dir: str | Path | None = None,
        output_filename: str = "emission_analysis.png",
        figsize: tuple[float, float] = (12, 5),
        add_legend: bool = True,
        hist_log_y: bool = False,
        hist_xlim: tuple[float, float] | None = None,
        hist_ylim: tuple[float, float] | None = None,
        scatter_xlim: tuple[float, float] | None = None,
        scatter_ylim: tuple[float, float] | None = None,
        hist_bin_width: float = 0.5,
        print_summary: bool = True,
        save_fig: bool = False,
        show_fig: bool = True,
        show_scatter: bool = True,  # 散布図の表示を制御するオプションを追加
    ) -> None:
        """
        排出量分析のプロットを作成する静的メソッド。

        Parameters:
        ------
            emission_data_list : list[EmissionData]
                EmissionDataオブジェクトのリスト。
            output_dir : str | Path | None
                出力先ディレクトリのパス。
            output_filename : str
                保存するファイル名。デフォルトは"emission_analysis.png"。
            dpi : int
                プロットの解像度。デフォルトは300。
            figsize : tuple[float, float]
                プロットのサイズ。デフォルトは(12, 5)。
            add_legend : bool
                凡例を追加するかどうか。デフォルトはTrue。
            hist_log_y : bool
                ヒストグラムのy軸を対数スケールにするかどうか。デフォルトはFalse。
            hist_xlim : tuple[float, float] | None
                ヒストグラムのx軸の範囲。デフォルトはNone。
            hist_ylim : tuple[float, float] | None
                ヒストグラムのy軸の範囲。デフォルトはNone。
            scatter_xlim : tuple[float, float] | None
                散布図のx軸の範囲。デフォルトはNone。
            scatter_ylim : tuple[float, float] | None
                散布図のy軸の範囲。デフォルトはNone。
            hist_bin_width : float
                ヒストグラムのビンの幅。デフォルトは0.5。
            print_summary : bool
                集計結果を表示するかどうか。デフォルトはFalse。
            save_fig : bool
                図をファイルに保存するかどうか。デフォルトはFalse。
            show_fig : bool
                図を表示するかどうか。デフォルトはTrue。
            show_scatter : bool
                散布図（右図）を表示するかどうか。デフォルトはTrue。
        """
        # データをDataFrameに変換
        df = pd.DataFrame([e.to_dict() for e in emission_data_list])

        # プロットの作成（散布図の有無に応じてサブプロット数を調整）
        if show_scatter:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            axes = [ax1, ax2]
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0] // 2, figsize[1]))
            axes = [ax1]

        # カラーマップの定義
        colors: dict[HotspotType, str] = {"bio": "blue", "gas": "red", "comb": "green"}

        # 存在するタイプを確認
        # HotspotTypeの定義順を基準にソート
        hotspot_types = list(get_args(HotspotType))
        existing_types = sorted(
            df["type"].unique(), key=lambda x: hotspot_types.index(x)
        )

        # 左側: ヒストグラム
        # ビンの範囲を設定
        start = 0  # 必ず0から開始
        if hist_xlim is not None:
            end = hist_xlim[1]
        else:
            end = np.ceil(df["emission_rate"].max() * 1.05)

        # ビン数を計算（end値をbin_widthで割り切れるように調整）
        n_bins = int(np.ceil(end / hist_bin_width))
        end = n_bins * hist_bin_width

        # ビンの生成（0から開始し、bin_widthの倍数で区切る）
        bins = np.linspace(start, end, n_bins + 1)

        # タイプごとにヒストグラムを積み上げ
        bottom = np.zeros(len(bins) - 1)
        for spot_type in existing_types:
            data = df[df["type"] == spot_type]["emission_rate"]
            if len(data) > 0:
                counts, _ = np.histogram(data, bins=bins)
                ax1.bar(
                    bins[:-1],
                    counts,
                    width=hist_bin_width,
                    bottom=bottom,
                    alpha=0.6,
                    label=spot_type,
                    color=colors[spot_type],
                )
                bottom += counts

        ax1.set_xlabel("CH$_4$ Emission (L min$^{-1}$)")
        ax1.set_ylabel("Frequency")
        if hist_log_y:
            # ax1.set_yscale("log")
            # 非線形スケールを設定（linthreshで線形から対数への遷移点を指定）
            ax1.set_yscale("symlog", linthresh=1.0)
        if hist_xlim is not None:
            ax1.set_xlim(hist_xlim)
        else:
            ax1.set_xlim(0, np.ceil(df["emission_rate"].max() * 1.05))

        if hist_ylim is not None:
            ax1.set_ylim(hist_ylim)
        else:
            ax1.set_ylim(0, ax1.get_ylim()[1])  # 下限を0に設定

        if show_scatter:
            # 右側: 散布図
            for spot_type in existing_types:
                mask = df["type"] == spot_type
                ax2.scatter(
                    df[mask]["emission_rate"],
                    df[mask]["delta_ch4"],
                    alpha=0.6,
                    label=spot_type,
                    color=colors[spot_type],
                )

            ax2.set_xlabel("Emission Rate (L min$^{-1}$)")
            ax2.set_ylabel("ΔCH$_4$ (ppm)")
            if scatter_xlim is not None:
                ax2.set_xlim(scatter_xlim)
            else:
                ax2.set_xlim(0, np.ceil(df["emission_rate"].max() * 1.05))

            if scatter_ylim is not None:
                ax2.set_ylim(scatter_ylim)
            else:
                ax2.set_ylim(0, np.ceil(df["delta_ch4"].max() * 1.05))

        # 凡例の表示
        if add_legend:
            for ax in axes:
                ax.legend(
                    bbox_to_anchor=(0.5, -0.30),
                    loc="upper center",
                    ncol=len(existing_types),
                )

        plt.tight_layout()

        # 図の保存
        if save_fig:
            if output_dir is None:
                raise ValueError(
                    "save_fig=Trueの場合、output_dirを指定する必要があります。有効なディレクトリパスを指定してください。"
                )
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
        # 図の表示
        if show_fig:
            plt.show()
        else:
            plt.close(fig=fig)

        if print_summary:
            # デバッグ用の出力
            print("\nビンごとの集計:")
            print(f"{'Range':>12} | {'bio':>8} | {'gas':>8} | {'total':>8}")
            print("-" * 50)

            for i in range(len(bins) - 1):
                bin_start = bins[i]
                bin_end = bins[i + 1]

                # 各タイプのカウントを計算
                counts_by_type: dict[HotspotType, int] = {"bio": 0, "gas": 0, "comb": 0}
                total = 0
                for spot_type in existing_types:
                    mask = (
                        (df["type"] == spot_type)
                        & (df["emission_rate"] >= bin_start)
                        & (df["emission_rate"] < bin_end)
                    )
                    count = len(df[mask])
                    counts_by_type[spot_type] = count
                    total += count

                # カウントが0の場合はスキップ
                if total > 0:
                    range_str = f"{bin_start:5.1f}-{bin_end:<5.1f}"
                    bio_count = counts_by_type.get("bio", 0)
                    gas_count = counts_by_type.get("gas", 0)
                    print(
                        f"{range_str:>12} | {bio_count:8d} | {gas_count:8d} | {total:8d}"
                    )
