import os
from py_flux_tracer import FftFileReorganizer

# 変数定義
base_path = "/home/connect0459/labo/py-flux-tracer/workspace/transfer_function/private/2024.08.06"
flag_file_name: str = "Flg-202406211500_202408061100.csv"
input_dir_names: list[str] = ["fft", "fft-detrend"]
output_dir_names: list[str] = ["sorted", "sorted-detrend"]

# メイン処理
try:
    flag_file_path: str = os.path.join(base_path, flag_file_name)
    for input_dir_name, output_dir_name in zip(input_dir_names, output_dir_names):
        input_dir_path: str = os.path.join(base_path, input_dir_name)
        output_dir_path: str = os.path.join(base_path, output_dir_name)

        # インスタンスを作成
        reorganizer = FftFileReorganizer(
            input_dir=input_dir_path,
            output_dir=output_dir_path,
            flag_csv_path=flag_file_path,
            sort_by_rh=False,
        )
        reorganizer.reorganize()
except KeyboardInterrupt:
    # キーボード割り込みが発生した場合、処理を中止する
    print("KeyboardInterrupt occurred. Abort processing.")
