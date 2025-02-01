import os
from py_flux_tracer import (
    FigureUtils,
    MonthlyFiguresGenerator,
)

output_dir = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/outputs"
)
input_configs: list[tuple[str, float]] = [
    ("2024.08.06", 9.2),
    ("2024.09.10", 10.0),
    ("2024.10.07", 11.7),
    ("2024.11.01", 13.2),
    ("2024.12.04", 15.5),
]

# フラグ
plot_spectra: bool = True

if __name__ == "__main__":
    mfg = MonthlyFiguresGenerator()
    FigureUtils.setup_plot_params(
        font_size=22, tick_size=18, font_family=["Arial", "Dejavu Sans"]
    )

    for term_tag, lag_sec in input_configs:
        # monthを0埋めのMM形式に変換
        month_str = str(term_tag)
        mfg.logger.info(f"{month_str}の処理を開始します。")
        input_dir: str = f"/mnt/c/Users/nakao/workspace/sac/transfer_function/data/ultra/{term_tag}/eddy_csv-resampled"

        if plot_spectra:
            # パワースペクトルのプロット
            mfg.plot_spectra(
                input_dir="/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/data/eddy_csv-resampled-for_turb",
                output_dir=(os.path.join(output_dir, "spectra", "tests")),
                output_filename_power=f"power_spectrum-test-{term_tag}.png",
                output_filename_co=f"co_spectrum-test-{term_tag}.png",
                fs=10,
                lag_second=lag_sec,
                label_ch4=None,
                label_c2h6=None,
                show_fig=False,
            )
            mfg.logger.info("'spectra'を作成しました。")
