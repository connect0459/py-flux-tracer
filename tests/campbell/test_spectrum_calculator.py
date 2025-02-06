import numpy as np
import pandas as pd
import pytest

from py_flux_tracer import SpectrumCalculator, WindowFunctionType


@pytest.fixture
def sample_df():
    """基本的なテスト用DataFrameを提供するフィクスチャ"""
    # データ長を1024以上に調整
    t = np.linspace(0, 10, 2048)  # 1024より大きい値に変更
    signal1 = np.sin(2 * np.pi * 1.0 * t)  # 1Hz
    signal2 = np.sin(2 * np.pi * 2.0 * t)  # 2Hz

    return pd.DataFrame(
        {
            "signal1": signal1,
            "signal2": signal2,
            "noisy_signal": signal1 + np.random.normal(0, 0.1, len(t)),
        }
    )


@pytest.fixture
def calculator(sample_df):
    """SpectrumCalculatorのインスタンスを提供するフィクスチャ"""
    return SpectrumCalculator(df=sample_df, fs=100.0)


def test_init():
    """初期化のテスト"""
    df = pd.DataFrame({"test": [1, 2, 3]})

    # デフォルトパラメータでの初期化
    calc = SpectrumCalculator(df=df, fs=10.0)
    assert calc._fs == 10.0
    assert calc._apply_window
    assert calc._plots == 30
    assert calc._window_type == "hamming"

    # カスタムパラメータでの初期化
    calc = SpectrumCalculator(
        df=df, fs=20.0, apply_window=False, plots=50, window_type="hanning"
    )
    assert calc._fs == 20.0
    assert not calc._apply_window
    assert calc._plots == 50
    assert calc._window_type == "hanning"


def test_calculate_power_spectrum_normal(calculator):
    """パワースペクトル計算の正常系テスト"""
    # 基本的な計算
    freqs, power = calculator.calculate_power_spectrum(
        col="signal1", dimensionless=True, frequency_weighted=True
    )

    assert isinstance(freqs, np.ndarray)
    assert isinstance(power, np.ndarray)
    assert len(freqs) == len(power)
    assert np.all(freqs >= 0)  # 周波数は非負
    assert not np.any(np.isnan(power))  # NaNがないことを確認

    # スケーリングオプションのテスト
    freqs_density, power_density = calculator.calculate_power_spectrum(
        col="signal1", scaling="density"
    )
    assert len(freqs_density) == len(power_density)


def test_calculate_power_spectrum_error(calculator):
    """パワースペクトル計算の異常系テスト"""
    # 存在しない列名
    with pytest.raises(KeyError):
        calculator.calculate_power_spectrum(col="non_existent_column")

    # 無効なスケーリングオプション
    with pytest.raises(ValueError):
        calculator.calculate_power_spectrum(col="signal1", scaling="invalid_option")


def test_calculate_cross_spectrum_normal(calculator):
    """クロススペクトル計算の正常系テスト"""
    # 基本的な計算
    freqs, co_spec, quad_spec, corr = calculator.calculate_cross_spectrum(
        col1="signal1",
        col2="signal2",
        apply_lag_correction_to_col2=False,  # lag_secondが必要なため、Falseに変更
    )

    assert isinstance(freqs, np.ndarray)
    assert isinstance(co_spec, np.ndarray)
    assert isinstance(quad_spec, np.ndarray)
    assert isinstance(corr, float)
    assert len(freqs) == len(co_spec) == len(quad_spec)
    assert -1 <= corr <= 1  # 相関係数は-1から1の範囲


def test_calculate_cross_spectrum_error(calculator):
    """クロススペクトル計算の異常系テスト"""
    # 存在しない列名
    with pytest.raises(KeyError):
        calculator.calculate_cross_spectrum(col1="non_existent_column", col2="signal1")

    # 遅れ時間補正の設定エラー
    with pytest.raises(ValueError):
        calculator.calculate_cross_spectrum(
            col1="signal1",
            col2="signal2",
            apply_lag_correction_to_col2=True,
            lag_second=None,
        )


def test_calculate_co_spectrum_normal(calculator):
    """コスペクトル計算の正常系テスト"""
    freqs, co_spec, corr = calculator.calculate_co_spectrum(
        col1="signal1",
        col2="signal2",
        apply_lag_correction_to_col2=False,  # lag_secondが必要なため、Falseに変更
    )

    assert isinstance(freqs, np.ndarray)
    assert isinstance(co_spec, np.ndarray)
    assert isinstance(corr, float)
    assert len(freqs) == len(co_spec)
    assert -1 <= corr <= 1


def test_correct_lag_time():
    """遅れ時間補正のテスト"""
    # テストデータの作成
    data1 = np.array([1, 2, 3, 4, 5])
    data2 = np.array([1, 2, 3, 4, 5])
    fs = 1.0

    # 正常系
    corrected_data1, corrected_data2 = SpectrumCalculator._correct_lag_time(
        data1=data1, data2=data2, fs=fs, lag_second=2.0
    )
    assert len(corrected_data1) == len(corrected_data2)

    # 異常系:負の遅れ時間
    with pytest.raises(ValueError):
        SpectrumCalculator._correct_lag_time(
            data1=data1, data2=data2, fs=fs, lag_second=-1.0
        )


def test_detrend():
    """トレンド除去のテスト"""
    # テストデータの作成(線形トレンド + ノイズ)
    x = np.linspace(0, 10, 100)
    trend = 2 * x + 1
    noise = np.random.normal(0, 0.1, 100)
    data = trend + noise

    # 一次トレンド除去
    detrended_first = SpectrumCalculator._detrend(data, first=True, second=False)
    assert len(detrended_first) == len(data)

    # 二次トレンド除去
    detrended_second = SpectrumCalculator._detrend(data, first=False, second=True)
    assert len(detrended_second) == len(data)

    # エラーケース:両方False
    with pytest.raises(ValueError):
        SpectrumCalculator._detrend(data, first=False, second=False)


def test_generate_window_function():
    """窓関数生成のテスト"""
    data_length = 2000
    window_types: list[WindowFunctionType] = ["hanning", "hamming", "blackman"]

    # 各窓関数タイプのテスト
    for window_type in window_types:
        window = SpectrumCalculator._generate_window_function(window_type, data_length)
        assert isinstance(window, np.ndarray)
        assert len(window) == data_length
        assert np.all(window > -1e-10)  # 数値誤差を考慮した閾値を設定
        assert np.all(window <= 1)


def test_smooth_spectrum():
    """スペクトル平滑化のテスト"""
    # テストデータの作成
    spectrum = np.random.normal(0, 1, 100)
    frequencies = np.linspace(0, 5, 100)

    smoothed = SpectrumCalculator._smooth_spectrum(
        spectrum=spectrum, frequencies=frequencies, freq_threshold=0.1
    )

    assert isinstance(smoothed, np.ndarray)
    assert len(smoothed) == len(spectrum)
    assert not np.any(np.isnan(smoothed))  # NaNがないことを確認


def test_lag_time_correction_with_timestamp():
    """
    タイムスタンプを含むデータの遅れ時間補正テスト

    これはすごく大事で、遅れ時間の補正ができていないと
    クロススペクトル計算に支障が出る。
    """
    # テストデータの作成
    n_samples = 2000  # 十分な長さのデータ
    fs = 10.0  # サンプリング周波数
    lag_seconds = 10.0  # 遅れ時間

    # タイムスタンプの生成
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = [base_time + pd.Timedelta(seconds=i / fs) for i in range(n_samples)]

    # テストデータの生成
    base_data = np.sin(2 * np.pi * 0.1 * np.arange(n_samples) / fs)  # 基準データ
    target_data = np.roll(base_data, int(lag_seconds * fs))  # 遅れを持つデータ

    # DataFrameの作成
    df = pd.DataFrame(
        {"TIMESTAMP": timestamps, "col_base": base_data, "col_target": target_data}
    )

    # SpectrumCalculatorのインスタンス化
    calculator = SpectrumCalculator(df=df, fs=fs)

    # クロススペクトル計算(遅れ時間補正あり)
    freqs, co_spec, quad_spec, corr = calculator.calculate_cross_spectrum(
        col1="col_base",
        col2="col_target",
        apply_lag_correction_to_col2=True,
        lag_second=lag_seconds,
    )

    # 補正後のデータを取得
    corrected_data1, corrected_data2 = SpectrumCalculator._correct_lag_time(
        data1=df["col_base"].to_numpy(),
        data2=df["col_target"].to_numpy(),
        fs=fs,
        lag_second=lag_seconds,
    )

    # 検証
    # 1. データ長が同じであることを確認
    assert len(corrected_data1) == len(corrected_data2)

    # 2. 遅れ時間補正後、対応するデータポイントが一致することを確認
    # 最初の数ポイントをチェック(数値誤差を考慮)
    np.testing.assert_allclose(
        corrected_data1[:10], corrected_data2[:10], rtol=1e-10, atol=1e-10
    )

    # 3. 相関係数が高いことを確認(正しく補正されていれば高い相関を示すはず)
    assert corr > 0.99
