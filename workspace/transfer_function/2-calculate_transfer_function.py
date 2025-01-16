import os
from py_flux_tracer import TransferFunctionCalculator

# 変数定義
date: str = "2024.06.21"
base_path = f"C:\\Users\\nakao\\workspace\\sac\\transfer_function\\data\\ultra\\{date}"
# tf_file_name: str = f"TF_Ultra.{date}.csv"
tf_file_name: str = f"TF_Ultra.{date}-detrend.csv"

output_dir: str = "C:\\Users\\nakao\\workspace\\sac\\transfer_function\\output"

show_co_spectra_plot: bool = True
# show_co_spectra_plot: bool = False
show_tf_plot: bool = True
# show_tf_plot: bool = False

# UltraのFFTファイルで使用されるキー名(スペース込み)
col_wt: str = "  f*cospec_wt/wt"
col_wch4: str = " f*cospec_wc/wc closed"
col_wc2h6: str = " f*cospec_wq/wq closed"

# メイン処理
try:
    file_path: str = os.path.join(base_path, tf_file_name)
    tfc = TransferFunctionCalculator(file_path, " f", 0.01, 1)
    # TransferFunctionCalculator.setup_plot_params(font_family=["MS Gothic", "Arial"])

    # コスペクトルのプロット
    tfc.create_plot_co_spectra(
        col1=col_wt,
        col2=col_wch4,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wCH_{4}}$ / $\overline{w^\prime CH_{4}^\prime}$",
        color2="red",
        subplot_label="(a)",
        show_fig=show_co_spectra_plot,
        output_dir=output_dir,
        output_basename=f"co_ch4-{date}",
    )

    tfc.create_plot_co_spectra(
        col1=col_wt,
        col2=col_wc2h6,
        label1=r"$fC_{wTv}$ / $\overline{w^\prime Tv^\prime}$",
        label2=r"$fC_{wC_{2}H_{6}}$ / $\overline{w^\prime C_{2}H_{6}^\prime}$",
        color2="orange",
        subplot_label="(b)",
        show_fig=show_co_spectra_plot,
        output_dir=output_dir,
        output_basename=f"co_c2h6-{date}",
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
        target_name="wCH4",
        show_fig=show_tf_plot,
        output_dir=output_dir,
        output_basename=f"tf_ch4-{date}",
        label_gas=r"CH$_4$",
    )
    tfc.create_plot_transfer_function(
        a=a_wc2h6,
        df_processed=df_wc2h6,
        reference_name="wTv",
        target_name="wC2H6",
        show_fig=show_tf_plot,
        output_dir=output_dir,
        output_basename=f"tf_c2h6-{date}",
        label_gas=r"C$_2$H$_6$",
    )

    print(f"wCH4の係数 a: {a_wch4}")
    print(f"wC2H6の係数 a: {a_wc2h6}")
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
