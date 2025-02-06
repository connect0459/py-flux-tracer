import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from py_flux_tracer import TransferFunctionCalculator


@pytest.fixture
def sample_data_file():
    """テスト用のサンプルデータファイルを作成するフィクスチャ"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        # ヘッダー行の作成
        f.write("freq,co1,co2\n")
        # テストデータの作成
        frequencies = np.logspace(-2, 1, 50)  # 0.01から10Hzまでの50点
        co1_values = 1.0 / (1 + frequencies)  # サンプルのコスペクトル1
        co2_values = 0.8 / (1 + frequencies)  # サンプルのコスペクトル2(減衰を含む)

        for freq, co1, co2 in zip(frequencies, co1_values, co2_values, strict=True):
            f.write(f"{freq:.6f},{co1:.6f},{co2:.6f}\n")

    yield f.name
    # テスト後にファイルを削除
    os.unlink(f.name)


@pytest.fixture
def calculator(sample_data_file):
    """TransferFunctionCalculatorのインスタンスを提供するフィクスチャ"""
    return TransferFunctionCalculator(
        filepath=sample_data_file,
        col_freq="freq",
        cutoff_freq_low=0.01,
        cutoff_freq_high=1.0,
    )


def test_initialization(calculator):
    """初期化のテスト"""
    assert isinstance(calculator._df, pd.DataFrame)
    assert calculator._col_freq == "freq"
    assert calculator._cutoff_freq_low == 0.01
    assert calculator._cutoff_freq_high == 1.0


def test_transfer_function():
    """伝達関数の計算テスト"""
    x = np.array([0.1, 0.5, 1.0])
    a = 0.5
    result = TransferFunctionCalculator.transfer_function(x, a)

    # 伝達関数の値が0から1の間に収まることを確認
    assert np.all(result >= 0)
    assert np.all(result <= 1)

    # x=aのときの値が期待値と一致することを確認
    assert (
        np.abs(
            TransferFunctionCalculator.transfer_function(np.array([a]), a)
            - np.exp(-np.log(np.sqrt(2)))
        )
        < 1e-10
    )


def test_process_data(calculator):
    """データ処理機能のテスト"""
    df_processed = calculator.process_data("co1", "co2")

    assert isinstance(df_processed, pd.DataFrame)
    assert "reference" in df_processed.columns
    assert "target" in df_processed.columns
    assert not df_processed.empty
    assert not df_processed.isnull().any().any()

    # 比率が1以下であることを確認
    assert np.all(df_processed["target"] / df_processed["reference"] <= 1)


def test_calculate_transfer_function(calculator):
    """伝達関数の係数計算テスト"""
    a, stderr, df_processed = calculator.calculate_transfer_function("co1", "co2")

    assert isinstance(a, float)
    assert isinstance(stderr, float)
    assert isinstance(df_processed, pd.DataFrame)
    assert a > 0  # 係数は正の値であるべき
    assert stderr > 0  # 標準誤差は正の値であるべき


def test_cutoff_df(calculator):
    """周波数カットオフ機能のテスト"""
    # テスト用のDataFrame作成
    test_df = pd.DataFrame(
        {"reference": [1.0] * 5, "target": [0.8] * 5},
        index=[0.001, 0.01, 0.1, 1.0, 10.0],
    )

    df_cutoff = calculator._cutoff_df(test_df)

    # カットオフ範囲内のデータのみが含まれていることを確認
    assert df_cutoff.index.min() >= calculator._cutoff_freq_low
    assert df_cutoff.index.max() <= calculator._cutoff_freq_high


def test_create_plot_co_spectra(calculator, tmp_path):
    """コスペクトルプロット作成のテスト"""
    output_dirpath = str(tmp_path)
    os.makedirs(output_dirpath, exist_ok=True)

    calculator.create_plot_co_spectra(
        "co1",
        "co2",
        output_dirpath=output_dirpath,
        show_fig=False,
        output_filename="co-co1_co2.png",  # 出力ファイル名を明示的に指定
        save_fig=True,  # 保存を明示的に指定
    )
    plt.close("all")

    assert os.path.exists(os.path.join(output_dirpath, "co-co1_co2.png"))


def test_create_plot_ratio(calculator, tmp_path):
    """比率プロット作成のテスト"""
    df_processed = calculator.process_data("co1", "co2")
    output_dirpath = str(tmp_path)
    os.makedirs(output_dirpath, exist_ok=True)

    calculator.create_plot_ratio(
        df_processed,
        "co1",
        "co2",
        output_dirpath=output_dirpath,
        show_fig=False,
        output_filename="ratio-co1_co2.png",  # 出力ファイル名を明示的に指定
        save_fig=True,  # 保存を明示的に指定
    )
    plt.close("all")

    assert os.path.exists(os.path.join(output_dirpath, "ratio-co1_co2.png"))


def test_create_plot_transfer_function(calculator, tmp_path):
    """伝達関数プロット作成のテスト"""
    a, _, df_processed = calculator.calculate_transfer_function("co1", "co2")
    output_dirpath = str(tmp_path)
    os.makedirs(output_dirpath, exist_ok=True)

    calculator.create_plot_transfer_function(
        a,
        df_processed,
        "co1",
        "co2",
        output_dirpath=output_dirpath,
        show_fig=False,
        output_filename="tf-co1_co2.png",  # 出力ファイル名を明示的に指定
        save_fig=True,  # 保存を明示的に指定
    )
    plt.close("all")

    assert os.path.exists(os.path.join(output_dirpath, "tf-co1_co2.png"))


def test_invalid_filepath():
    """無効なファイルパスのテスト"""
    with pytest.raises(FileNotFoundError):
        TransferFunctionCalculator(filepath="nonexistent.csv", col_freq="freq")


def test_invalid_data_processing(calculator):
    """無効なデータ処理のテスト"""
    with pytest.raises(KeyError):
        calculator.process_data("nonexistent1", "nonexistent2")
