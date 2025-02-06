import os
from pathlib import Path

import pandas as pd

from py_flux_tracer import (
    BiasRemovalConfig,
    FluxFootprintAnalyzer,
    H2OCorrectionConfig,
    HotspotData,
    MobileMeasurementAnalyzer,
    MobileMeasurementConfig,
    MonthlyConverter,
    setup_plot_params,
)

# picoデータの補正式に関するパラメータ
pico_h2o_correction = H2OCorrectionConfig(
    coef_b=1.0111e-06,  # 1次の係数
    coef_c=-1.8683e-10,  # 2次の係数
)
pico_bias_removal = BiasRemovalConfig(
    quantile_value=0.05,
)

# MSAInputConfigによる詳細指定
configs: list[MobileMeasurementConfig] = [
    MobileMeasurementConfig(
        lag=7,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.10.17/input/Pico100121_241017_092120+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.09/input/Pico100121_241109_103128.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.11/input/Pico100121_241111_091102+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.14/input/Pico100121_241114_093745+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.18/input/Pico100121_241118_092855+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.20/input/Pico100121_241120_092932+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.24/input/Pico100121_241124_092712+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.25/input/Pico100121_241125_090721+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.28/input/Pico100121_241128_090240+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.11.30/input/Pico100121_241130_092420+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
    MobileMeasurementConfig(
        lag=13,
        fs=1,
        path="/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/data/2024.12.02/input/Pico100121_241202_090316+.txt",
        h2o_correction=pico_h2o_correction,
        bias_removal=pico_bias_removal,
    ),
]

# フォントファイルを登録(必要な場合のみで可)
font_paths: list[str | Path] = [
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/msgothic.ttc",  # 日本語のデフォルト
]

# 変数定義
center_lan: float = 34.573904320329724  # 観測地点の緯度
center_lon: float = 135.4829511120712  # 観測地点の経度
num_sections: int = 4  # セクション数
plot_count: int = 100
# plot_count: int = 10000
# plot_count: int = 50000

# スケールチェック用の仮地点の要素(緯度、経度、ラベル)
check_points_for_scale_checker: list[tuple[float, float, str]] = [
    (34.55958388887034, 135.4461794468429, "石津水再生センター"),
    (34.601272994096846, 135.46248381802235, "三宝水再生センター"),
]
hotspot_custom_sizes: dict[str, tuple[tuple[float, float], float]] = {
    "small": ((0, 0.5), 50),
    "medium": ((0.5, 1.0), 150),
    "large": ((1.0, float("inf")), 250),
    # "small": ((0, 0.5), 100),
    # "medium": ((0.5, 1.0), 200),
    # "large": ((1.0, float("inf")), 300),
}

# ファイルおよびディレクトリのパス
output_dirpath: str = "/home/connect0459/labo/py-flux-tracer/workspace/footprint/private/outputs"  # 出力先のディレクトリ
dotenv_path: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/.env"  # .envファイル
)

start_end_dates_list: list[list[str]] = [
    ["2024-05-15", "2024-12-31"],
    ["2024-06-01", "2024-08-31"],
    ["2024-09-01", "2024-11-30"],
]
plot_ch4: bool = False
plot_c2h6: bool = False
plot_ratio: bool = True
plot_ratio_legend: bool = True
plot_ch4_gas: bool = False
plot_ch4_bio: bool = False
plot_mono: bool = False
plot_scale_checker: bool = False

if __name__ == "__main__":
    # 出力先ディレクトリを作成
    os.makedirs(output_dirpath, exist_ok=True)
    # 図のスタイルの指定
    setup_plot_params(font_family=["Arial", "MS Gothic"], font_paths=font_paths)

    # ホットスポットの検出
    msa = MobileMeasurementAnalyzer(
        center_lat=center_lan,
        center_lon=center_lon,
        configs=configs,
        num_sections=num_sections,
        hotspot_area_meter=50,
        window_minutes=5,
        logging_debug=False,
    )
    hotspots: list[HotspotData] = msa.analyze_hotspots(duplicate_check_mode="time_all")

    # インスタンスを作成
    ffa = FluxFootprintAnalyzer(
        z_m=111,
        # column_mapping={
        #     "datetime": "Date",  # 日時カラム
        #     "wind_direction": "Wind direction",  # 風向 [度]
        #     "wind_speed": "WS vector",  # 風速 [m/s]
        #     "friction_velocity": "u*",  # 摩擦速度 [m/s]
        #     "sigma_v": "sigmaV",  # 風速の標準偏差 [m/s]
        #     "sigma_v": "sigma_v",  # 風速の標準偏差 [m/s]
        #     "stability": "z/L",  # 安定度パラメータ [-]
        # },
        logging_debug=False,
    )

    # 航空写真の取得
    local_image_path: str = (
        "/home/connect0459/labo/py-flux-tracer/storage/assets/SAC_2025-zoom_13.png"
    )
    image = ffa.get_satellite_image_from_local(
        local_image_path=local_image_path
    )  # ローカル

    for i, start_end_date in enumerate(start_end_dates_list):
        start_date = start_end_date[0]
        end_date = start_end_date[1]
        date_tag: str = f"-{start_date}_{end_date}"
        ffa.logger.info(f"Start analyzing from {start_date} to {end_date}.")

        # with文でブロック終了時に__exit__を自動呼出し
        with MonthlyConverter(
            "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/monthly",
            file_pattern="SA.Ultra.*.xlsx",
        ) as converter:
            # 特定の期間のデータを読み込む
            df_month = converter.read_sheets(
                sheet_names=["Final"],
                start_date=start_date,
                end_date=end_date,
                include_end_date=True,
            )

        df: pd.DataFrame = ffa.combine_all_data(df_month, source_type="monthly")

        # CH4
        if plot_ch4:
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fch4_ultra",
                plot_count=plot_count,
            )
            ffa.plot_flux_footprint(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                cbar_label=r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename=f"footprint_ch4{date_tag}.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        # C2H6
        if plot_c2h6:
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fc2h6_ultra",
                plot_count=plot_count,
            )
            ffa.plot_flux_footprint(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=5,
                xy_max=5000,
                cbar_label=r"C$_2$H$_6$ flux (nmol m$^{-2}$ s$^{-1}$)",
                cbar_labelpad=35,
                output_dirpath=output_dirpath,
                output_filename=f"footprint_c2h6{date_tag}.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        # ratio
        df["Fratio"] = (df["Fc2h6_ultra"] / df["Fch4_ultra"]) / 0.076 * 100
        if plot_ratio:
            # カスタムのサイズ範囲とマーカーサイズを指定する場合
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fratio",
                plot_count=plot_count,
            )
            # フットプリントとホットスポットの可視化
            ffa.plot_flux_footprint_with_hotspots(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                hotspots=hotspots,
                hotspot_sizes=hotspot_custom_sizes,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                add_legend=False,
                cbar_label=r"Gas Ratio of CH$_4$ flux (%)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename=f"footprint_ratio{date_tag}.png",
                save_fig=True,
                show_fig=False,
            )

        if plot_ratio_legend:
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fratio",
                plot_count=100,
            )
            # フットプリントとホットスポットの可視化
            ffa.plot_flux_footprint_with_hotspots(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                hotspots=hotspots,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                add_legend=True,
                cbar_label=r"Gas Ratio of CH$_4$ flux (%)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename="footprint_ratio-legend.png",
                save_fig=True,
                show_fig=False,
            )
            ffa.plot_flux_footprint_with_hotspots(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                hotspots=hotspots,
                hotspot_labels={
                    "bio": "生物起源",
                    "gas": "都市ガス起源",
                    "comb": "燃焼起源",
                },
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                add_legend=True,
                cbar_label=r"Gas Ratio of CH$_4$ flux (%)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename="footprint_ratio-legend-ja.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        # 都市ガス起源のCH4フラックス
        if plot_ch4_gas:
            df["Fch4_gas"] = (df["Fratio"] / 100) * df["Fch4_ultra"]
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fch4_gas",
                plot_count=plot_count,
            )
            ffa.plot_flux_footprint(
                x_list=x_list,
                y_list=y_list,
                c_list=c_list,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=60,
                xy_max=5000,
                cbar_label=r"Gas CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename=f"footprint_ch4_gas{date_tag}.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        # 生物起源のCH4フラックス
        if plot_ch4_bio:
            df["Fch4_bio"] = (1 - (df["Fratio"] / 100)) * df["Fch4_ultra"]
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fch4_bio",
                plot_count=plot_count,
            )
            ffa.plot_flux_footprint(
                x_list=x_list,
                y_list=y_list,
                c_list=c_list,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=60,
                xy_max=5000,
                cbar_label=r"Bio CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename=f"footprint_ch4_bio{date_tag}.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        if plot_mono:
            image_for_mono = ffa.get_satellite_image_from_local(
                local_image_path=local_image_path,
                alpha=0.5,
                grayscale=True,
            )  # ローカル
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fch4_ultra",
                plot_count=100,
            )
            # フットプリントを描画しない
            ffa.plot_flux_footprint_with_hotspots(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=None,
                figsize=(8, 8),
                hotspots=hotspots,
                hotspots_alpha=0.5,
                hotspot_labels={
                    "bio": "生物起源",
                    "gas": "都市ガス起源",
                    "comb": "燃焼起源",
                },
                # hotspot_colors={"bio": "blue", "gas": "red", "comb": "green"},
                hotspot_colors={"bio": "gray", "gas": "gray", "comb": "green"},
                hotspot_markers={"bio": "^", "gas": "o", "comb": "s"},
                hotspot_sizes=hotspot_custom_sizes,
                legend_loc="upper right",
                legend_bbox_to_anchor=(0.95, 0.95),
                legend_ncol=1,
                center_lat=center_lan,
                center_lon=center_lon,
                satellite_image=image_for_mono,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                add_legend=False,
                add_cbar=False,
                cbar_label=r"Gas Ratio of CH$_4$ flux (%)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename="footprint_mono.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list

        if plot_scale_checker and i == 0:
            x_list, y_list, c_list = ffa.calculate_flux_footprint(
                df=df,
                col_flux="Fch4_ultra",
                plot_count=plot_count,
            )
            ffa.plot_flux_footprint_with_scale_checker(
                x_list=x_list,  # メートル単位のx座標
                y_list=y_list,  # メートル単位のy座標
                c_list=c_list,
                center_lat=center_lan,
                center_lon=center_lon,
                check_points=check_points_for_scale_checker,
                satellite_image=image,
                cmap="jet",
                vmin=0,
                vmax=100,
                xy_max=5000,
                cbar_label=r"CH$_4$ flux (nmol m$^{-2}$ s$^{-1}$)",
                cbar_labelpad=20,
                output_dirpath=output_dirpath,
                output_filename="footprint_ch4-scale_checker.png",
                save_fig=True,
                show_fig=False,
            )
            del x_list, y_list, c_list
