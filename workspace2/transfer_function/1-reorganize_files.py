import os

from modules.fft_files_reorganizer import FftFileReorganizer

# 変数定義
base_path = "/mnt/c/Users/nakao/workspace/sac/transfer_function/data/ultra/2025.01.10"
flag_filename: str = "Flg-202412231100_202501101000.csv"
input_dirnames: list[str] = ["fft", "fft-detrend"]
output_dirnames: list[str] = ["sorted", "sorted-detrend"]

# メイン処理
if __name__ == "__main__":
    try:
        flag_filepath: str = os.path.join(base_path, flag_filename)
        for input_dirname, output_dirname in zip(
            input_dirnames, output_dirnames, strict=True
        ):
            input_dirpath_path: str = os.path.join(base_path, input_dirname)
            output_dirpath_path: str = os.path.join(base_path, output_dirname)

            # インスタンスを作成
            reorganizer = FftFileReorganizer(
                input_dirpath=input_dirpath_path,
                output_dirpath=output_dirpath_path,
                flag_csv_path=flag_filepath,
                sort_by_rh=False,
            )
            reorganizer.logger.info(
                f"ファイルのコピーを開始します: {input_dirname} -> {output_dirname}"
            )
            reorganizer.reorganize()
            reorganizer.logger.info("ファイルのコピーが完了しました")
    except KeyboardInterrupt:
        # キーボード割り込みが発生した場合、処理を中止する
        print("KeyboardInterrupt occurred. Abort processing.")
