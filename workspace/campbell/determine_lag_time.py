import os
from py_flux_tracer import EddyDataPreprocessor


if __name__ == "__main__":
    target_home: str = (
        "/mnt/c/Users/nakao/workspace/sac/ultra/data/2025.01.10/Ultra_Eddy"
    )
    input_dir: str = os.path.join(target_home, "eddy_csv-resampled")
    output_dir: str = os.path.join(target_home, "lag_time")

    # メイン処理
    edp = EddyDataPreprocessor(fs=10)
    # edp.get_generated_columns_names() # クラス内部で生成される列名を見る場合はコメントアウト
    edp.analyze_lag_times(
        input_dir=input_dir,
        input_files_suffix=".csv",
        use_resampling=False,
        col1="edp_wind_w",
        col2_list=["Tv", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
        output_dir=output_dir,
    )
