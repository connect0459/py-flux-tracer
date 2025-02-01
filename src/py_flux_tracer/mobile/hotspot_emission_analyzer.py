import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import get_args
from .mobile_measurement_analyzer import HotspotData, HotspotType


@dataclass
class EmissionData:
    """
    ホットスポットの排出量データを格納するクラス。

    Parameters
    ----------
        timestamp : str
            タイムスタンプ
        delta_ch4 : float
            CH4の増加量 (ppm)
        delta_c2h6 : float
            C2H6の増加量 (ppb)
        delta_ratio : float
            C2H6/CH4比
        emission_per_min : float
            排出量 (L/min)
        emission_per_day : float
            日排出量 (L/day)
        emission_per_year : float
            年間排出量 (L/year)
        latitude : float
            緯度
        longitude : float
            経度
        section : str | int | float
            セクション情報
        type : HotspotType
            ホットスポットの種類（`HotspotType`を参照）
    """

    timestamp: str
    delta_ch4: float
    delta_c2h6: float
    delta_ratio: float
    emission_per_min: float
    emission_per_day: float
    emission_per_year: float
    latitude: float
    longitude: float
    section: str | int | float
    type: HotspotType

    def __post_init__(self) -> None:
        """
        Initialize時のバリデーションを行います。

        Raises
        ----------
            ValueError: 入力値が不正な場合
        """
        # timestamp のバリデーション
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("'timestamp' must be a non-empty string")

        # typeのバリデーションは型システムによって保証されるため削除
        # HotspotTypeはLiteral["bio", "gas", "comb"]として定義されているため、
        # 不正な値は型チェック時に検出されます

        # section のバリデーション（Noneは許可）
        if self.section is not None and not isinstance(self.section, (str, int, float)):
            raise ValueError("'section' must be a string, int, float, or None")

        # latitude のバリデーション
        if (
            not isinstance(self.latitude, (int, float))
            or not -90 <= self.latitude <= 90
        ):
            raise ValueError("'latitude' must be a number between -90 and 90")

        # longitude のバリデーション
        if (
            not isinstance(self.longitude, (int, float))
            or not -180 <= self.longitude <= 180
        ):
            raise ValueError("'longitude' must be a number between -180 and 180")

        # delta_ch4 のバリデーション
        if not isinstance(self.delta_ch4, (int, float)) or self.delta_ch4 < 0:
            raise ValueError("'delta_ch4' must be a non-negative number")

        # delta_c2h6 のバリデーション
        if not isinstance(self.delta_c2h6, (int, float)):
            raise ValueError("'delta_c2h6' must be a int or float")

        # delta_ratio のバリデーション
        if not isinstance(self.delta_ratio, (int, float)):
            raise ValueError("'delta_ratio' must be a int or float")

        # emission_per_min のバリデーション
        if (
            not isinstance(self.emission_per_min, (int, float))
            or self.emission_per_min < 0
        ):
            raise ValueError("'emission_per_min' must be a non-negative number")

        # emission_per_day のバリデーション
        expected_daily = self.emission_per_min * 60 * 24
        if not math.isclose(self.emission_per_day, expected_daily, rel_tol=1e-10):
            raise ValueError(
                f"'emission_per_day' ({self.emission_per_day}) does not match "
                f"calculated value from emission rate ({expected_daily})"
            )

        # emission_per_year のバリデーション
        expected_annual = self.emission_per_day * 365
        if not math.isclose(self.emission_per_year, expected_annual, rel_tol=1e-10):
            raise ValueError(
                f"'emission_per_year' ({self.emission_per_year}) does not match "
                f"calculated value from daily emission ({expected_annual})"
            )

    def to_dict(self) -> dict:
        """
        データクラスの内容を辞書形式に変換します。

        Returns
        ----------
            dict: データクラスの属性と値を含む辞書
        """
        return {
            "timestamp": self.timestamp,
            "delta_ch4": self.delta_ch4,
            "delta_c2h6": self.delta_c2h6,
            "delta_ratio": self.delta_ratio,
            "emission_per_min": self.emission_per_min,
            "emission_per_day": self.emission_per_day,
            "emission_per_year": self.emission_per_year,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "section": self.section,
            "type": self.type,
        }


@dataclass
class EmissionFormula:
    """
    排出量計算式の係数セットを保持するデータクラス
    設定した`coef_a`と`coef_b`は以下のように使用される。

    ```py
    emission_per_min = np.exp((np.log(spot.delta_ch4) + coef_a) / coef_b)
    ```

    Parameters
    ----------
        name : str
            計算式の名前（例: "weller", "weitzel", "joo", "umezawa"など）
        coef_a : float
            計算式の係数a
        coef_b : float
            計算式の係数b

    Examples
    ----------
    >>> # Weller et al. (2022)の係数を使用する場合
    >>> formula = EmissionFormula(name="weller", coef_a=0.988, coef_b=0.817)
    >>>
    >>> # Weitzel et al. (2019)の係数を使用する場合
    >>> formula = EmissionFormula(name="weitzel", coef_a=0.521, coef_b=0.795)
    >>>
    >>> # カスタム係数を使用する場合
    >>> formula = EmissionFormula(name="custom", coef_a=1.0, coef_b=1.0)
    """

    name: str
    coef_a: float
    coef_b: float

    def __post_init__(self) -> None:
        """
        パラメータの検証を行います。
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("'name' must be a non-empty string")
        if not isinstance(self.coef_a, (int, float)):
            raise ValueError("'coef_a' must be a number")
        if not isinstance(self.coef_b, (int, float)):
            raise ValueError("'coef_b' must be a number")


@dataclass
class HotspotEmissionConfig:
    """
    排出量計算の設定を保持するデータクラス

    Parameters
    ----------
        formula : EmissionFormula
            使用する計算式の設定
        emission_categories : dict[str, dict[str, float]]
            排出量カテゴリーの閾値設定
            デフォルト値: {
                "low": {"min": 0, "max": 6},  # < 6 L/min
                "medium": {"min": 6, "max": 40},  # 6-40 L/min
                "high": {"min": 40, "max": float("inf")},  # > 40 L/min
            }

    Examples
    ----------
    >>> # Weller et al. (2022)の係数を使用する場合
    >>> config = HotspotEmissionConfig(
    ...     formula=EmissionFormula(name="weller", coef_a=0.988, coef_b=0.817),
    ...     emission_categories={
    ...         "low": {"min": 0, "max": 6},  # < 6 L/min
    ...         "medium": {"min": 6, "max": 40},  # 6-40 L/min
    ...         "high": {"min": 40, "max": float("inf")},  # > 40 L/min
    ...     }
    ... )
    >>> # 複数のconfigをリスト形式で定義する場合
    >>> emission_configs: list[HotspotEmissionConfig] = [
    ...     HotspotEmissionConfig(formula=EmissionFormula(name="weller", coef_a=0.988, coef_b=0.817)),
    ...     HotspotEmissionConfig(formula=EmissionFormula(name="weitzel", coef_a=0.521, coef_b=0.795)),
    ...     HotspotEmissionConfig(formula=EmissionFormula(name="joo", coef_a=2.738, coef_b=1.329)),
    ...     HotspotEmissionConfig(formula=EmissionFormula(name="umezawa", coef_a=2.716, coef_b=0.741)),
    ... ]
    """

    formula: EmissionFormula
    emission_categories: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "low": {"min": 0, "max": 6},  # < 6 L/min
            "medium": {"min": 6, "max": 40},  # 6-40 L/min
            "high": {"min": 40, "max": float("inf")},  # > 40 L/min
        }
    )

    def __post_init__(self) -> None:
        """
        パラメータの検証を行います。
        """
        # カテゴリーの閾値の整合性チェック
        for category, limits in self.emission_categories.items():
            if "min" not in limits or "max" not in limits:
                raise ValueError(
                    f"Category {category} must have 'min' and 'max' values"
                )
            if limits["min"] > limits["max"]:
                raise ValueError(f"Category {category} has invalid range: min > max")


class HotspotEmissionAnalyzer:
    @staticmethod
    def calculate_emission_rates(
        hotspots: list[HotspotData],
        config: HotspotEmissionConfig,
        print_summary: bool = False,
    ) -> list[EmissionData]:
        """
        検出されたホットスポットのCH4漏出量を計算・解析し、統計情報を生成します。

        Parameters
        ----------
            hotspots : list[HotspotData]
                分析対象のホットスポットのリスト
            config : HotspotEmissionConfig
                排出量計算の設定
            print_summary : bool
                統計情報を表示するかどうか。デフォルトはFalse。

        Returns
        ----------
            list[EmissionData]
                - 各ホットスポットの排出量データを含むリスト

        Examples
        ----------
        >>> # Weller et al. (2022)の係数を使用する例
        >>> config = HotspotEmissionConfig(
        ...     formula=EmissionFormula(name="weller", coef_a=0.988, coef_b=0.817),
        ...     emission_categories={
        ...         "low": {"min": 0, "max": 6},
        ...         "medium": {"min": 6, "max": 40},
        ...         "high": {"min": 40, "max": float("inf")},
        ...     }
        ... )
        >>> emissions_list = HotspotEmissionAnalyzer.calculate_emission_rates(
        ...     hotspots=hotspots,
        ...     config=config,
        ...     print_summary=True
        ... )
        """
        # 係数の取得
        coef_a: float = config.formula.coef_a
        coef_b: float = config.formula.coef_b

        # 排出量の計算
        emission_data_list = []
        for spot in hotspots:
            # 漏出量の計算 (L/min)
            emission_per_min = np.exp((np.log(spot.delta_ch4) + coef_a) / coef_b)
            # 日排出量 (L/day)
            emission_per_day = emission_per_min * 60 * 24
            # 年間排出量 (L/year)
            emission_per_year = emission_per_day * 365

            emission_data = EmissionData(
                timestamp=spot.timestamp,
                delta_ch4=spot.delta_ch4,
                delta_c2h6=spot.delta_c2h6,
                delta_ratio=spot.delta_ratio,
                emission_per_min=emission_per_min,
                emission_per_day=emission_per_day,
                emission_per_year=emission_per_year,
                latitude=spot.avg_lat,
                longitude=spot.avg_lon,
                section=spot.section,
                type=spot.type,
            )
            emission_data_list.append(emission_data)

        # 統計計算用にDataFrameを作成
        emission_df = pd.DataFrame([e.to_dict() for e in emission_data_list])

        # タイプ別の統計情報を計算
        # get_args(HotspotType)を使用して型安全なリストを作成
        types = list(get_args(HotspotType))
        for spot_type in types:
            df_type = emission_df[emission_df["type"] == spot_type]
            if len(df_type) > 0:
                # 既存の統計情報を計算
                type_stats = {
                    "count": len(df_type),
                    "emission_per_min_min": df_type["emission_per_min"].min(),
                    "emission_per_min_max": df_type["emission_per_min"].max(),
                    "emission_per_min_mean": df_type["emission_per_min"].mean(),
                    "emission_per_min_median": df_type["emission_per_min"].median(),
                    "total_annual_emission": df_type["emission_per_year"].sum(),
                    "mean_annual_emission": df_type["emission_per_year"].mean(),
                }

                # 排出量カテゴリー別の統計を追加
                category_counts = {}
                for category, limits in config.emission_categories.items():
                    mask = (df_type["emission_per_min"] >= limits["min"]) & (
                        df_type["emission_per_min"] < limits["max"]
                    )
                    category_counts[category] = len(df_type[mask])
                type_stats["emission_categories"] = category_counts

                if print_summary:
                    print(f"\n{spot_type}タイプの統計情報:")
                    print(f"  検出数: {type_stats['count']}")
                    print("  排出量 (L/min):")
                    print(f"    最小値: {type_stats['emission_per_min_min']:.2f}")
                    print(f"    最大値: {type_stats['emission_per_min_max']:.2f}")
                    print(f"    平均値: {type_stats['emission_per_min_mean']:.2f}")
                    print(f"    中央値: {type_stats['emission_per_min_median']:.2f}")
                    print("  排出量カテゴリー別の検出数:")
                    for category, count in category_counts.items():
                        print(f"    {category}: {count}")
                    print("  年間排出量 (L/year):")
                    print(f"    合計: {type_stats['total_annual_emission']:.2f}")
                    print(f"    平均: {type_stats['mean_annual_emission']:.2f}")

        return emission_data_list

    @staticmethod
    def plot_emission_analysis(
        emissions: list[EmissionData],
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
            emissions : list[EmissionData]
                calculate_emission_ratesで生成された分析結果
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
        df = pd.DataFrame([e.to_dict() for e in emissions])

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
            end = np.ceil(df["emission_per_min"].max() * 1.05)

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
                data = df[df["type"] == spot_type]["emission_per_min"]
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
                data = df[df["type"] == spot_type]["emission_per_min"]
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
            ax1.set_xlim(0, np.ceil(df["emission_per_min"].max() * 1.05))

        if hist_ylim is not None:
            ax1.set_ylim(hist_ylim)
        else:
            ax1.set_ylim(0, ax1.get_ylim()[1])  # 下限を0に設定

        if show_scatter:
            # 右側: 散布図
            for spot_type in existing_types:
                mask = df["type"] == spot_type
                ax2.scatter(
                    df[mask]["emission_per_min"],
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
                ax2.set_xlim(0, np.ceil(df["emission_per_min"].max() * 1.05))

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
                        & (df["emission_per_min"] >= bin_start)
                        & (df["emission_per_min"] < bin_end)
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
