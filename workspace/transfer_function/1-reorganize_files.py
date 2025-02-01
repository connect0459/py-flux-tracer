import os
from py_flux_tracer import FftFileReorganizer

# 変数定義
base_path = "/mnt/c/Users/nakao/workspace/sac/transfer_function/data/ultra/2025.01.10"
flag_filename: str = "Flg-202412231100_202501101000.csv"
input_dir_names: list[str] = ["fft", "fft-detrend"]
output_dir_names: list[str] = ["sorted", "sorted-detrend"]

# メイン処理
try:
    flag_filepath: str = os.path.join(base_path, flag_filename)
    for input_dir_name, output_dir_name in zip(input_dir_names, output_dir_names):
        input_dir_path: str = os.path.join(base_path, input_dir_name)
        output_dir_path: str = os.path.join(base_path, output_dir_name)

        # インスタンスを作成
        reorganizer = FftFileReorganizer(
            input_dir=input_dir_path,
            output_dir=output_dir_path,
            flag_csv_path=flag_filepath,
            sort_by_rh=False,
        )
        reorganizer.logger.info(
            f"ファイルのコピーを開始します: {input_dir_name} -> {output_dir_name}"
        )
        reorganizer.reorganize()
        reorganizer.logger.info("ファイルのコピーが完了しました")
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
