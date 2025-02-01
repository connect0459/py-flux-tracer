import matplotlib.font_manager as fm
from py_flux_tracer import FigureUtils, TransferFunctionCalculator

# # フォントファイルを登録
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

FigureUtils.setup_plot_params(font_family=font_array)

# 変数定義
tf_csv_path: str = (
    "/mnt/c/Users/nakao/workspace/sac/transfer_function/tf-a/TF_Ultra_a.csv"
)
output_dir: str = (
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
    ("a_ch4-used", "CH$_4$", "red", "ch4"),
    ("a_c2h6-used", "C$_2$H$_6$", "orange", "c2h6"),
]

# メイン処理の例
if __name__ == "__main__":
    try:
        # 伝達関数曲線のプロット
        TransferFunctionCalculator.create_plot_tf_curves_from_csv(
            filepath=tf_csv_path,
            gas_configs=gas_configs,
            output_dir=output_dir,
            line_colors=custom_colors,
            show_fig=False,
        )

    except KeyboardInterrupt:
        print("KeyboardInterrupt occurred. Abort processing.")
