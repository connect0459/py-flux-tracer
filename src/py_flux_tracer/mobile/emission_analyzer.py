import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import get_args, Literal
from .mobile_measurement_analyzer import HotspotData, HotspotType


@dataclass
class EmissionData:
    """
    ホットスポットの排出量データを格納するクラス。

    Parameters
    ----------
        timestamp : str
            タイムスタンプ
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
        delta_ratio : float
            C2H6/CH4比
        emission_rate : float
            排出量 (L/min)
        daily_emission : float
            日排出量 (L/day)
        annual_emission : float
            年間排出量 (L/year)
    """

    timestamp: str
    type: HotspotType
    section: str | int | float
    latitude: float
    longitude: float
    delta_ch4: float
    delta_c2h6: float
    delta_ratio: float
    emission_rate: float
    daily_emission: float
    annual_emission: float

    def __post_init__(self) -> None:
        """
        Initialize時のバリデーションを行います。

        Raises
        ----------
            ValueError: 入力値が不正な場合
        """
        # sourceのバリデーション
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("'timestamp' must be a non-empty string")

        # typeのバリデーションは型システムによって保証されるため削除
        # HotspotTypeはLiteral["bio", "gas", "comb"]として定義されているため、
        # 不正な値は型チェック時に検出されます

        # sectionのバリデーション（Noneは許可）
        if self.section is not None and not isinstance(self.section, (str, int, float)):
            raise ValueError("'section' must be a string, int, float, or None")

        # 緯度のバリデーション
        if (
            not isinstance(self.latitude, (int, float))
            or not -90 <= self.latitude <= 90
        ):
            raise ValueError("'latitude' must be a number between -90 and 90")

        # 経度のバリデーション
        if (
            not isinstance(self.longitude, (int, float))
            or not -180 <= self.longitude <= 180
        ):
            raise ValueError("'longitude' must be a number between -180 and 180")

        # delta_ch4のバリデーション
        if not isinstance(self.delta_ch4, (int, float)) or self.delta_ch4 < 0:
            raise ValueError("'delta_ch4' must be a non-negative number")

        # delta_c2h6のバリデーション
        if not isinstance(self.delta_c2h6, (int, float)):
            raise ValueError("'delta_c2h6' must be a int or float")

        # ratioのバリデーション
        if not isinstance(self.delta_ratio, (int, float)) or self.delta_ratio < 0:
            raise ValueError("'delta_ratio' must be a non-negative number")

        # emission_rateのバリデーション
        if not isinstance(self.emission_rate, (int, float)) or self.emission_rate < 0:
            raise ValueError("'emission_rate' must be a non-negative number")

        # daily_emissionのバリデーション
        expected_daily = self.emission_rate * 60 * 24
        if not math.isclose(self.daily_emission, expected_daily, rel_tol=1e-10):
            raise ValueError(
                f"'daily_emission' ({self.daily_emission}) does not match "
                f"calculated value from emission rate ({expected_daily})"
            )

        # annual_emissionのバリデーション
        expected_annual = self.daily_emission * 365
        if not math.isclose(self.annual_emission, expected_annual, rel_tol=1e-10):
            raise ValueError(
                f"'annual_emission' ({self.annual_emission}) does not match "
                f"calculated value from daily emission ({expected_annual})"
            )

        # NaN値のチェック
        numeric_fields = [
            self.latitude,
            self.longitude,
            self.delta_ch4,
            self.delta_c2h6,
            self.delta_ratio,
            self.emission_rate,
            self.daily_emission,
            self.annual_emission,
        ]
        if any(math.isnan(x) for x in numeric_fields):
            raise ValueError("Numeric fields cannot contain NaN values")

    def to_dict(self) -> dict:
        """
        データクラスの内容を辞書形式に変換します。

        Returns
        ----------
            dict: データクラスの属性と値を含む辞書
        """
        return {
            "timestamp": self.timestamp,
            "type": self.type,
            "section": self.section,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "delta_ch4": self.delta_ch4,
            "delta_c2h6": self.delta_c2h6,
            "delta_ratio": self.delta_ratio,
            "emission_rate": self.emission_rate,
            "daily_emission": self.daily_emission,
            "annual_emission": self.annual_emission,
        }


class EmissionAnalyzer:
    @staticmethod
    def calculate_emission_rates(
        hotspots: list[HotspotData],
        method: Literal["weller", "weitzel", "joo", "umezawa"] = "weller",
        print_summary: bool = False,
        custom_formulas: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[EmissionData], dict[str, dict[str, float]]]:
        """
        検出されたホットスポットのCH4漏出量を計算・解析し、統計情報を生成します。

        Parameters
        ----------
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

        Returns
        ----------
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
                timestamp=spot.timestamp,
                type=spot.type,
                section=spot.section,
                latitude=spot.avg_lat,
                longitude=spot.avg_lon,
                delta_ch4=spot.delta_ch4,
                delta_c2h6=spot.delta_c2h6,
                delta_ratio=spot.delta_ratio,
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
        hotspot_colors: dict[HotspotType, str] = {
            "bio": "blue",
            "gas": "red",
            "comb": "green",
        },
        add_legend: bool = True,
        hist_log_y: bool = False,
        hist_xlim: tuple[float, float] | None = None,
        hist_ylim: tuple[float, float] | None = None,
        scatter_xlim: tuple[float, float] | None = None,
        scatter_ylim: tuple[float, float] | None = None,
        hist_bin_width: float = 0.5,
        print_summary: bool = False,
        stack_bars: bool = True,  # 追加：積み上げ方式を選択するパラメータ
        save_fig: bool = False,
        show_fig: bool = True,
        show_scatter: bool = True,  # 散布図の表示を制御するオプションを追加
    ) -> None:
        """
        排出量分析のプロットを作成する静的メソッド。

        Parameters
        ----------
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
            hotspot_colors : dict[HotspotType, str]
                ホットスポットの色を定義する辞書。
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
        if stack_bars:
            # 積み上げ方式
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
                        color=hotspot_colors[spot_type],
                    )
                    bottom += counts
        else:
            # 重ね合わせ方式
            for spot_type in existing_types:
                data = df[df["type"] == spot_type]["emission_rate"]
                if len(data) > 0:
                    counts, _ = np.histogram(data, bins=bins)
                    ax1.bar(
                        bins[:-1],
                        counts,
                        width=hist_bin_width,
                        alpha=0.4,  # 透明度を上げて重なりを見やすく
                        label=spot_type,
                        color=hotspot_colors[spot_type],
                    )

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
                    color=hotspot_colors[spot_type],
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
