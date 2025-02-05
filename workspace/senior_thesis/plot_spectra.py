import os
import matplotlib.font_manager as fm
from py_flux_tracer import FigureUtils, MonthlyFiguresGenerator

# フォントファイルを登録
font_paths: list[str] = [
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/arial.ttf",  # 英語のデフォルト
    "/home/connect0459/labo/py-flux-tracer/workspace/private/fonts/msgothic.ttc",  # 日本語のデフォルト
]
for path in font_paths:
    fm.fontManager.addfont(path)
# フォント名を指定
font_array: list[str] = [
    "Arial",
    "MS Gothic",
]
FigureUtils.setup_plot_params(
    font_family=font_array,
    # font_size=24,
    # legend_size=24,
    # tick_size=24,
)
output_dirpath = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/outputs"
)
terms_tags: list[str] = [
    "05_06",
    # "07_08",
    # "09_10",
    # "11_12",
]

if __name__ == "__main__":
    mfg = MonthlyFiguresGenerator()

    for term_tag in terms_tags:
        # monthを0埋めのMM形式に変換
        month_str = str(term_tag)
        mfg.logger.info(f"{month_str}の処理を開始します。")
        input_dirpath: str = f"/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/data/eddy_csv-resampled-two-{term_tag}"

        # パワースペクトルのプロット
        mfg.plot_spectra(
            input_dirpath=input_dirpath,
            output_dirpath=(os.path.join(output_dirpath, "spectra")),
            output_filename_power=f"power_spectrum-{term_tag}.png",
            output_filename_co=f"co_spectrum-{term_tag}.png",
            fs=10,
            lag_second=0,
            label_ch4=None,
            label_c2h6=None,
            plot_co=False,
            show_fig=False,
        )
        mfg.logger.info("'spectra'を作成しました。")
