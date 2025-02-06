from pathlib import Path

from py_flux_tracer import (
    TfCurvesFromCsvConfig,
    TransferFunctionCalculator,
    setup_plot_params,
)

# # フォントファイルを登録
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

# 変数定義
tf_csv_path: str = (
    "/mnt/c/Users/nakao/workspace/sac/transfer_function/tf-a/TF_Ultra_a.csv"
)
output_dirpath: str = (
    "/mnt/c/Users/nakao/workspace/sac/transfer_function/outputs/all_curves"
)

# カスタムカラーの定義
custom_colors = [
    "#00ff00",
    "#3cb371",
    "#00ffff",
    "#00bfff",
    "#0000ff",
    "#9400d3",
    "#ff69b4",
    "#000000",
]

# ガスの設定
gas_configs = [
    TfCurvesFromCsvConfig(
        col_coef_a="a_ch4-used",
        label_gas="CH$_4$",
        base_color="red",
        gas_name="ch4",
    ),
    TfCurvesFromCsvConfig(
        col_coef_a="a_c2h6-used",
        label_gas="C$_2$H$_6$",
        base_color="orange",
        gas_name="c2h6",
    ),
]

if __name__ == "__main__":
    try:
        # 各ガスについて伝達関数曲線をプロット
        for config in gas_configs:
            TransferFunctionCalculator.create_plot_tf_curves_from_csv(
                filepath=tf_csv_path,
                config=config,
                figsize=(10, 6),
                output_dirpath=output_dirpath,
                output_filename=f"all_tf_curves-{config.gas_name}.png",
                line_colors=custom_colors,
                show_fig=False,
            )

    except KeyboardInterrupt:
        print("KeyboardInterrupt occurred. Abort processing.")
