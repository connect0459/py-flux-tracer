import os
from dataclasses import dataclass
from pathlib import Path

from py_flux_tracer import TransferFunctionCalculator, setup_plot_params


@dataclass
class TFAnalysisConfig:
    """
    伝達関数の解析設定を管理するデータクラス

    Parameters
    ----------
        input_file: str | Path
            入力ファイルのパス
        output_dirpath: str | Path
            出力ディレクトリのパス
        suffix: str
            ファイル名の接尾辞(例: "", "-detrend")
        show_co_spectra: bool
            コスペクトルのプロットを表示するかどうか
        show_tf: bool
            伝達関数のプロットを表示するかどうか
    """

    input_file: str | Path
    output_dirpath: str | Path
    suffix: str
    show_co_spectra: bool = False
    show_tf: bool = False

    def __post_init__(self) -> None:
        """
        パスの検証を行います。
        """
        if not os.path.exists(self.input_file):
            raise ValueError(f"入力ファイルが存在しません: {self.input_file}")


""" ------ config start ------ """
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
    font_size=24,
    legend_size=24,
    tick_size=24,
)

# 変数定義
# UltraのFFTファイルで使用されるキー名(スペース込み)
col_freq: str = " f"
col_wt: str = "  f*cospec_wt/wt"
col_wch4: str = " f*cospec_wc/wc closed"
col_wc2h6: str = " f*cospec_wq/wq closed"
# 出力先のルートディレクトリ
output_dirpath: str = "/mnt/c/Users/nakao/workspace/sac/transfer_function/outputs"
# 日付のリスト
dates_list: list[str] = [
    "2024.06.21",
    "2024.08.06",
    "2024.09.01",
    "2024.10.07",
    "2024.11.01",
    "2024.12.04",
    "2024.12.23",
    "2025.01.10",
]
try:
    for date in dates_list:
        base_path = (
            f"/mnt/c/Users/nakao/workspace/sac/transfer_function/data/ultra/{date}"
        )

        # 解析設定の定義
        configs: list[TFAnalysisConfig] = [
            TFAnalysisConfig(
                input_file=os.path.join(base_path, f"TF_Ultra.{date}.csv"),
                output_dirpath=os.path.join(output_dirpath, "each-raw"),
                suffix="",  # トレンド除去なし
                show_co_spectra=False,
                show_tf=False,
            ),
            TFAnalysisConfig(
                input_file=os.path.join(base_path, f"TF_Ultra.{date}-detrend.csv"),
                output_dirpath=os.path.join(output_dirpath, "each-detrend"),
                suffix="-detrend",  # トレンド除去あり
                show_co_spectra=False,
                show_tf=False,
            ),
        ]
        """ ------ config end ------ """

        # メイン処理
        for config in configs:
            print(f"\n{os.path.basename(config.input_file)}の処理を開始...")

            tfc = TransferFunctionCalculator(
                filepath=config.input_file,
                col_freq=col_freq,
                cutoff_freq_low=0.01,
                cutoff_freq_high=1,
            )

            # コスペクトルのプロット
            tfc.create_plot_co_spectra(
                col1=col_wt,
                col2=col_wch4,
                label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
                label2=r"$fC_{wCH_{4}}$ / $\overline{w^\prime CH_{4}^\prime}$",
                color2="red",
                # subplot_label="(a)",
                show_fig=config.show_co_spectra,
                output_dirpath=config.output_dirpath,
                output_filename=f"co_ch4-{date}{config.suffix}.png",
                add_legend=False,
            )

            tfc.create_plot_co_spectra(
                col1=col_wt,
                col2=col_wc2h6,
                label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
                label2=r"$fC_{wC_{2}H_{6}}$ / $\overline{w^\prime C_{2}H_{6}^\prime}$",
                color2="orange",
                # subplot_label="(b)",
                show_fig=config.show_co_spectra,
                output_dirpath=config.output_dirpath,
                output_filename=f"co_c2h6-{date}{config.suffix}.png",
                add_legend=False,
            )

            print("伝達関数を分析中...")
            # 伝達関数の計算
            a_wch4, _, df_wch4 = tfc.calculate_transfer_function(
                col_reference=col_wt, col_target=col_wch4
            )
            a_wc2h6, _, df_wc2h6 = tfc.calculate_transfer_function(
                col_reference=col_wt, col_target=col_wc2h6
            )

            # カーブフィット図の作成
            tfc.create_plot_transfer_function(
                a=a_wch4,
                df_processed=df_wch4,
                reference_name="wTv",
                target_name=r"wCH$_4$",
                show_fig=config.show_tf,
                output_dirpath=config.output_dirpath,
                output_filename=f"tf_ch4-{date}{config.suffix}.png",
                label_target=r"CH$_4$",
            )
            tfc.create_plot_transfer_function(
                a=a_wc2h6,
                df_processed=df_wc2h6,
                reference_name="wTv",
                # target_name="wC2H6",
                target_name=r"wC$_2$H$_6$",
                show_fig=config.show_tf,
                output_dirpath=config.output_dirpath,
                output_filename=f"tf_c2h6-{date}{config.suffix}.png",
                label_target=r"C$_2$H$_6$",
            )

            print(f"wCH4の係数 a: {a_wch4}")
            print(f"wC2H6の係数 a: {a_wc2h6}")

except KeyboardInterrupt:
    print("KeyboardInterrupt occurred. Abort processing.")
