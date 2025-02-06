import os

from py_flux_tracer import (
    BiasRemovalConfig,
    EmissionData,
    EmissionFormula,
    H2OCorrectionConfig,
    HotspotData,
    HotspotEmissionAnalyzer,
    HotspotEmissionConfig,
    MobileMeasurementAnalyzer,
    MobileMeasurementConfig,
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

num_sections: int = 4  # セクション数
west_sections_list: list[int] = [
    0,
    1,
]  # 西側となるセクション番号(num_sectionsに応じて変更)
output_dirpath: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/mobile/private/outputs"
)
print_summary: bool = True
# print_summary: bool = False

if __name__ == "__main__":
    msa = MobileMeasurementAnalyzer(
        center_lat=34.573904320329724,
        center_lon=135.4829511120712,
        configs=configs,
        num_sections=num_sections,
        hotspot_area_meter=50,
        window_minutes=5,
        logging_debug=False,
    )

    # msa.calculate_measurement_stats()

    # ホットスポット検出
    # all_hotspots: list[HotspotData] = msa.analyze_hotspots()
    unique_hotspots: list[HotspotData] = msa.analyze_hotspots(
        # duplicate_check_mode="time_window",
        duplicate_check_mode="time_all",
    )

    # 結果の表示
    bio_spots = [h for h in unique_hotspots if h.type == "bio"]
    gas_spots = [h for h in unique_hotspots if h.type == "gas"]
    comb_spots = [h for h in unique_hotspots if h.type == "comb"]

    if print_summary:
        print("\nResults:")
        print(f"  Bio:{len(bio_spots)},Gas:{len(gas_spots)},Comb:{len(comb_spots)}")

        # 区画ごとの分析を追加
        # 各区画のホットスポット数をカウント
        section_counts = {
            i: {"bio": 0, "gas": 0, "comb": 0} for i in range(num_sections)
        }
        for spot in unique_hotspots:
            section_counts[spot.section][spot.type] += 1

        # 区画ごとの結果を表示
        print("\n区画ごとの分析結果:")
        section_size: float = msa.get_section_size()
        for section, counts in section_counts.items():
            start_angle = -180 + section * section_size
            end_angle = start_angle + section_size
            print(f"\n区画 {section} ({start_angle:.1f}° ~ {end_angle:.1f}°):")
            print(f"  Bio : {counts['bio']}")
            print(f"  Gas : {counts['gas']}")
            print(f"  Comb: {counts['comb']}")

    # sectionが0または1(西側)のホットスポットのみを残す
    # hotspots = [h for h in hotspots if h.section in west_sections_list]

    # 地図の作成と保存
    msa.create_hotspots_map(unique_hotspots, output_dirpath=output_dirpath)

    # ホットスポットを散布図で表示
    msa.plot_scatter_c2c1(
        unique_hotspots, output_dirpath=output_dirpath, show_fig=False
    )

    # ヒストグラムを作図
    msa.plot_ch4_delta_histogram(
        hotspots=unique_hotspots,
        output_dirpath=output_dirpath,
        figsize=(10, 6),
        xlim=(0, 1.4),
        yscale_log=False,
        show_fig=False,
        print_bins_analysis=True,
    )

    # 統計情報を表示
    msa.analyze_delta_ch4_stats(hotspots=unique_hotspots)
    # csvに出力
    msa.export_hotspots_to_csv(hotspots=unique_hotspots, output_dirpath=output_dirpath)

    # Emissionの分析
    emission_configs: list[HotspotEmissionConfig] = [
        HotspotEmissionConfig(
            formula=EmissionFormula(name="weller", coef_a=0.988, coef_b=0.817)
        ),
        HotspotEmissionConfig(
            formula=EmissionFormula(name="weitzel", coef_a=0.521, coef_b=0.795)
        ),
        HotspotEmissionConfig(
            formula=EmissionFormula(name="joo", coef_a=2.738, coef_b=1.329)
        ),
        HotspotEmissionConfig(
            formula=EmissionFormula(name="umezawa", coef_a=2.716, coef_b=0.741)
        ),
    ]
    for config in emission_configs:
        method: str = config.formula.name
        hea = HotspotEmissionAnalyzer()
        hea.logger.info(f"{method}のemission解析を開始します。")

        # 排出量の計算と基本統計
        emissions_list: list[EmissionData] = hea.calculate_emission_rates(
            hotspots=unique_hotspots, config=config, print_summary=True
        )

        # 分布の可視化
        hea.plot_emission_analysis(
            emissions=emissions_list,
            output_dirpath=output_dirpath,
            output_filename=f"emission_plots-{method}.png",
            hist_log_y=True,
            hist_xlim=(0, 60),
            scatter_xlim=(0, 60),
            hist_ylim=(0, 100),
            scatter_ylim=(0, 1.6),
            hist_bin_width=0.5,
            add_legend=False,
            save_fig=True,
            show_fig=False,
            show_scatter=False,
            # print_summary=True,
            print_summary=False,
        )

    # 各回の時系列データを可視化
    msa.plot_conc_timeseries(
        output_dirpath=os.path.join(output_dirpath, "timeseries"),
        output_filename="timeseries.png",
        save_fig=True,
        show_fig=False,
    )
