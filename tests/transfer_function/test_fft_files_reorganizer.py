import csv
import os

import pytest

from py_flux_tracer import FftFileReorganizer


@pytest.fixture
def temp_dirs(tmp_path):
    """
    # テスト用の一時ディレクトリを作成し、テストデータを配置するためのフィクスチャ
    """
    input_dirpath = tmp_path / "input"
    output_dirpath = tmp_path / "output"
    input_dirpath.mkdir()
    return str(input_dirpath), str(output_dirpath)


@pytest.fixture
def flag_file(temp_dirs):
    """
    # テスト用のフラグファイルを作成するフィクスチャ
    """
    input_dirpath, _ = temp_dirs
    flag_path = os.path.join(input_dirpath, "flags.csv")

    with open(flag_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "Flg", "RH"])
        writer.writerow(["2024/01/01 12:00", "0", "50.0"])
        writer.writerow(["2024/01/01 12:30", "1", "75.0"])

    return flag_path


def test_get_rh_directory():
    """
    # RHディレクトリの命名規則が正しく機能することを確認
    """
    assert FftFileReorganizer.get_rh_directory(5.5) == "RH10"
    assert FftFileReorganizer.get_rh_directory(15.7) == "RH20"
    assert FftFileReorganizer.get_rh_directory(80.1) == "RH90"  # 80.1は90に切り上げ
    assert FftFileReorganizer.get_rh_directory(86.0) == "RH90"  # 86.0も90に切り上げ
    assert FftFileReorganizer.get_rh_directory(91.2) == "RH100"  # 91.2は100に切り上げ
    assert FftFileReorganizer.get_rh_directory(0.0) == "RH0"
    assert FftFileReorganizer.get_rh_directory(-1.0) == "bad_data"
    assert FftFileReorganizer.get_rh_directory(101.0) == "bad_data"


def test_file_reorganization(temp_dirs, flag_file):
    """
    # ファイルの再編成が正しく行われることを確認
    """
    input_dirpath, output_dirpath = temp_dirs

    # テスト用のFFTファイルを作成
    test_files = [
        "FFT_TOA5_1.SAC_Eddy_1_2024_01_01_1200.csv",
        "FFT_TOA5_1.SAC_Eddy_1_2024_01_01_1230.csv",
    ]

    for file in test_files:
        with open(os.path.join(input_dirpath, file), "w") as f:
            f.write("test data")

    # FftFileReorganizerのインスタンスを作成して実行
    reorganizer = FftFileReorganizer(
        input_dirpath=input_dirpath,
        output_dirpath=output_dirpath,
        flag_csv_path=flag_file,
        sort_by_rh=True,
    )
    reorganizer.reorganize()

    # 期待される出力を確認
    assert os.path.exists(os.path.join(output_dirpath, "good_data_all", test_files[0]))
    assert os.path.exists(os.path.join(output_dirpath, "bad_data", test_files[1]))
    assert os.path.exists(os.path.join(output_dirpath, "RH50", test_files[0]))


def test_invalid_filename(temp_dirs, flag_file):
    """
    # 無効なファイル名が適切に処理されることを確認
    """
    input_dirpath, output_dirpath = temp_dirs

    # 無効なファイル名でファイルを作成
    invalid_file = "invalid_filename.csv"
    with open(os.path.join(input_dirpath, invalid_file), "w") as f:
        f.write("test data")

    reorganizer = FftFileReorganizer(
        input_dirpath=input_dirpath, output_dirpath=output_dirpath, flag_csv_path=flag_file
    )
    reorganizer.reorganize()

    # 警告が生成されることを確認(実装方法によって確認方法は変更が必要かもしれません)
    assert not os.path.exists(os.path.join(output_dirpath, "good_data_all", invalid_file))
    assert not os.path.exists(os.path.join(output_dirpath, "bad_data", invalid_file))
