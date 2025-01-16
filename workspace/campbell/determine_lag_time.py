from py_flux_tracer import EddyDataPreprocessor


if __name__ == "__main__":
    target_home: str = (
        "/home/connect0459/labo/py-flux-tracer/workspace/campbell/private/data/test-2025.01.10"
    )
    input_dir: str = f"{target_home}/eddy_csv"
    output_dir: str = f"{target_home}/output"

    # メイン処理
    edp = EddyDataPreprocessor(fs=10)
    edp.analyze_lag_times(
        input_dir=input_dir,
        input_files_suffix=".csv",
        use_resampling=False,
        col1="wind_w",
        col2_list=["Tv", "Ultra_CH4_ppm_C", "Ultra_C2H6_ppb"],
        output_dir=output_dir,
    )
