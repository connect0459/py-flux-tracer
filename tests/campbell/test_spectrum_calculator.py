import pytest
import numpy as np
import pandas as pd
from py_flux_tracer import SpectrumCalculator


@pytest.fixture
def sample_data():
    """
    # テスト用の時系列データを生成するフィクスチャ
    # 正弦波とノイズを組み合わせてテストデータを作成
    """
    fs = 10.0  # サンプリング周波数
    t = np.linspace(0, 10, int(10 * fs))  # 10秒間のデータ

    # 基準となる信号（周波数1Hzの正弦波）
    signal1 = np.sin(2 * np.pi * 1 * t)
    # 遅れのある信号（位相差あり）
    signal2 = np.sin(2 * np.pi * 1 * t - np.pi / 4)

    # ノイズを追加
    noise = np.random.normal(0, 0.1, len(t))
    signal1 += noise
    signal2 += noise

    df = pd.DataFrame({"signal1": signal1, "signal2": signal2})

    return df, fs


def test_initialization(sample_data):
    """
    # クラスの初期化が正しく行われることを確認
    """
    df, fs = sample_data
    calculator = SpectrumCalculator(
        df=df, cols_apply_lag_time=["signal2"], lag_second=0.1, fs=fs
    )

    assert calculator.fs == fs
    assert calculator.cols_apply_lag_time == ["signal2"]
    assert calculator.lag_second == 0.1
    assert calculator.dimensionless
    assert calculator.plots == 30


def test_power_spectrum(sample_data):
    """
    # パワースペクトル計算が正しく機能することを確認
    """
    df, fs = sample_data
    calculator = SpectrumCalculator(df=df, cols_apply_lag_time=[], lag_second=0.0, fs=fs)

    freqs, power = calculator.calculate_power_spectrum(
        col="signal1", frequency_weighted=True, interpolate_points=False
    )

    # 基本的な検証
    assert len(freqs) > 0
    assert len(power) > 0
    assert len(freqs) == len(power)
    # 1Hz付近にピークがあることを確認
    peak_freq_idx = np.argmax(power[1:]) + 1  # 0Hz成分を除外
    assert 0.8 < freqs[peak_freq_idx] < 1.2


def test_cospectrum(sample_data):
    """
    # コスペクトル計算が正しく機能することを確認
    """
    df, fs = sample_data
    calculator = SpectrumCalculator(df=df, cols_apply_lag_time=[], lag_second=0.0, fs=fs)

    freqs, cospec, corr = calculator.calculate_co_spectrum(
        col1="signal1",
        col2="signal2",
        frequency_weighted=True,
        interpolate_points=False,
    )

    # 基本的な検証
    assert len(freqs) > 0
    assert len(cospec) > 0
    assert len(freqs) == len(cospec)
    assert -1 <= corr <= 1


def test_detrend():
    """
    # トレンド除去が正しく機能することを確認
    """
    # 線形トレンドを持つテストデータを生成
    t = np.linspace(0, 10, 100)
    trend = 2 * t + 1
    signal = np.sin(2 * np.pi * 1 * t) + trend

    detrended = SpectrumCalculator._detrend(signal, 10)

    # トレンドが除去されていることを確認
    assert np.abs(np.mean(detrended)) < 0.1
    assert np.abs(np.polyfit(t, detrended, 1)[0]) < 0.1


def test_window_function():
    """
    # 窓関数の生成が正しく機能することを確認
    """

    # 各種窓関数のテスト
    window_hamming = SpectrumCalculator._generate_window_function("hamming", 100)
    window_hanning = SpectrumCalculator._generate_window_function("hanning", 100)
    window_blackman = SpectrumCalculator._generate_window_function("blackman", 100)

    # 数値誤差を考慮した許容値を設定
    eps = 1e-15

    assert len(window_hamming) == 100
    assert len(window_hanning) == 100
    assert len(window_blackman) == 100
    assert -eps <= np.min(window_hamming) <= np.max(window_hamming) <= 1 + eps
    assert -eps <= np.min(window_hanning) <= np.max(window_hanning) <= 1 + eps
    assert -eps <= np.min(window_blackman) <= np.max(window_blackman) <= 1 + eps


def test_lag_correction():
    """
    # 遅れ時間補正が正しく機能することを確認
    """
    # 位相差のあるテストデータを生成
    t = np.linspace(0, 10, 100)
    signal1 = np.sin(2 * np.pi * 1 * t)
    signal2 = np.sin(2 * np.pi * 1 * (t - 0.1))  # 0.1秒の遅れ

    # 負の遅れ時間でエラーが発生することを確認
    with pytest.raises(ValueError):
        SpectrumCalculator._correct_lag_time(signal1, signal2, 10, -0.1)
