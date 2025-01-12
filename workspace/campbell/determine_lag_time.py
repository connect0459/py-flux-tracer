from py_flux_tracer import EddyDataPreprocessor


if __name__ == "__main__":
    target_home: str = (
        "/home/z23641k/labo/omu-eddy-covariance/workspace/campbell/private/data"
    )
    input_dir: str = f"{target_home}/eddy_csv"
    output_dir: str = f"{target_home}/output"

    # メイン処理
    edp = EddyDataPreprocessor(fs=10)
    edp.analyze_lag_times(
        input_dir=input_dir,
        input_files_suffix=".csv",
        use_resampling=False,
        key1="wind_w",
        key2_list=["Tv", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
        output_dir=output_dir,
    )
