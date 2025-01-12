from py_flux_tracer import EddyDataPreprocessor


if __name__ == "__main__":
    root_path: str = (
        "/home/connect0459/labo/omu-eddy-covariance/workspace/campbell/private/data/test"
    )

    input_dir: str = f"{root_path}/eddy_csv"
    resampled_dir: str = f"{root_path}/eddy_csv-resampled"
    calc_py_dir: str = f"{root_path}/calc-py"

    try:
        edp = EddyDataPreprocessor(fs=10)
        edp.output_resampled_data(
            input_dir=input_dir,
            resampled_dir=resampled_dir,
            calc_py_dir=calc_py_dir,
            output_ratio=True,
            output_resampled=True,
        )
    except KeyboardInterrupt:
        # キーボード割り込みが発生した場合、処理を中止する
        print("KeyboardInterrupt occurred. Abort processing.")
