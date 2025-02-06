import os
from pathlib import Path

from py_flux_tracer import (
    EddyDataFiguresGenerator,
    SlopeLine,
    SpectralPlotConfig,
    setup_plot_params,
)

# フォントファイルを登録
font_paths: list[str | Path] = [
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/msgothic.ttc",  # 日本語のデフォルト
]
# フォント名を指定
font_array: list[str] = [
    "Arial",
    "MS Gothic",
]
setup_plot_params(
    font_family=font_array,
    font_paths=font_paths,
    # font_size=24,
    # legend_size=24,
    # tick_size=24,
)
output_dirpath = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/outputs"
)
terms_tags: list[str] = [
    # "example",
    "05_06",
    "07_08",
    "09_10",
    "11_12",
]

ch4_config = SpectralPlotConfig(
    # psd_ylabel=r"$fS_{\mathrm{CH_4}} / s_{\mathrm{CH_4}}^2$",
    psd_ylabel="無次元パワースペクトル",
    co_ylabel=r"$fC_{w\mathrm{CH_4}} / \overline{w'\mathrm{CH_4}'}$",
    color="red",
)
c2h6_config = SpectralPlotConfig(
    # psd_ylabel=r"$fS_{\mathrm{C_2H_6}} / s_{\mathrm{C_2H_6}}^2$",
    psd_ylabel="無次元パワースペクトル",
    co_ylabel=r"$fC_{w\mathrm{C_2H_6}} / \overline{w'\mathrm{C_2H_6}'}$",
    color="orange",
)

power_slope = SlopeLine(
    coordinates=((0.01, 10e-3), (10, 10e-6)), text="-2/3", text_pos=(1, 0.001)
)

if __name__ == "__main__":
    eda = EddyDataFiguresGenerator(fs=10)

    for term_tag in terms_tags:
        eda.logger.info(f"{term_tag}の処理を開始します。")
        input_dirpath: str = f"/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/data/eddy_csv-resampled-two-{term_tag}"

        # パワースペクトルのプロット
        eda.plot_c1c2_spectra(
            input_dirpath=input_dirpath,
            file_suffix=".csv",
            output_dirpath=(os.path.join(output_dirpath, "spectra")),
            output_filename_power=f"ps-{term_tag}.png",
            output_filename_co=f"cos-{term_tag}.png",
            scaling_power="spectrum",
            lag_second=None,
            plot_co=False,
            show_fig=False,
            ch4_config=ch4_config,
            c2h6_config=c2h6_config,
            ylim_power=(10e-7, 10e-3),
            power_slope=power_slope,
        )
        eda.logger.info("'spectra'を作成しました。")
