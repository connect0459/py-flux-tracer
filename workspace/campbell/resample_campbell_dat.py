import os

from py_flux_tracer import EddyDataPreprocessor

if __name__ == "__main__":
    root_path: str = "/mnt/c/Users/nakao/workspace/sac/ultra/data/2025.01.29/Ultra_Eddy"

    input_dirpath: str = os.path.join(root_path, "eddy_csv")
    resampled_dirpath: str = os.path.join(root_path, "eddy_csv-resampled")
    c2c1_ratio_dirpath: str = os.path.join(root_path, "calc-py")

    try:
        edp = EddyDataPreprocessor(fs=10)
        edp.output_resampled_data(
            input_dirpath=input_dirpath,
            c2c1_ratio_dirpath=c2c1_ratio_dirpath,
            resampled_dirpath=resampled_dirpath,
            output_c2c1_ratio=True,
            output_resampled=True,
        )
    except KeyboardInterrupt:
        # キーボード割り込みが発生した場合、処理を中止する
        print("KeyboardInterrupt occurred. Abort processing.")
