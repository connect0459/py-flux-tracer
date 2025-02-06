from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal

WindowFunctionType = Literal["hanning", "hamming", "blackman"]


class SpectrumCalculator:
    def __init__(
        self,
        df: pd.DataFrame,
        fs: float,
        apply_window: bool = True,
        plots: int = 30,
        window_type: WindowFunctionType = "hamming",
    ):
        """
        データロガーから取得したデータファイルを用いて計算を行うクラス。

        Parameters
        ----------
            df: pd.DataFrame
                pandasのデータフレーム。解析対象のデータを含む。
            fs: float
                サンプリング周波数(Hz)。データのサンプリングレートを指定。
            apply_window: bool, optional
                窓関数を適用するフラグ。デフォルトはTrue。
            plots: int, optional
                プロットする点の数。可視化のためのデータポイント数。
            window_type: WindowFunctionType, optional
                窓関数の種類。デフォルトは'hamming'。
        """
        self._df: pd.DataFrame = df
        self._fs: float = fs
        self._apply_window: bool = apply_window
        self._plots: int = plots
        self._window_type: WindowFunctionType = window_type

    def calculate_co_spectrum(
        self,
        col1: str,
        col2: str,
        dimensionless: bool = True,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        scaling: str = "spectrum",
        detrend_1st: bool = True,
        detrend_2nd: bool = False,
        apply_lag_correction_to_col2: bool = True,
        lag_second: float | None = None,
    ) -> tuple:
        """指定されたcol1とcol2のコスペクトルをDataFrameから計算するためのメソッド。

        Parameters
        ----------
            col1: str
                データの列名1
            col2: str  
                データの列名2
            dimensionless: bool, optional
                分散で割って無次元化を行うかのフラグ。デフォルトはTrueで無次元化を行う
            frequency_weighted: bool, optional
                周波数の重みづけを適用するかのフラグ。デフォルトはTrueで重みづけを行う
            interpolate_points: bool, optional
                対数軸上で等間隔なデータ点を生成するかのフラグ。デフォルトはTrueで等間隔点を生成する
            scaling: str, optional
                スペクトルのスケーリング方法。"density"でスペクトル密度、"spectrum"でスペクトル。デフォルトは"spectrum"
            detrend_1st: bool, optional
                1次トレンドを除去するかのフラグ。デフォルトはTrueで除去を行う
            detrend_2nd: bool, optional
                2次トレンドを除去するかのフラグ。デフォルトはFalseで除去を行わない
            apply_lag_correction_to_col2: bool, optional
                col2に遅れ時間補正を適用するかのフラグ。デフォルトはTrueで補正を行う
            lag_second: float | None, optional
                col1からcol2が遅れている時間(秒)。apply_lag_correction_to_col2がTrueの場合に必要。デフォルトはNone

        Returns
        ----------
            tuple
                以下の3つの要素を含むタプル:
                - freqs: np.ndarray
                    周波数軸(対数スケールの場合は対数変換済み)
                - co_spectrum: np.ndarray  
                    コスペクトル(対数スケールの場合は対数変換済み)
                - corr_coef: float
                    変数の相関係数

        Examples
        --------
        >>> sc = SpectrumCalculator(df=data, fs=10)
        >>> freqs, co_spec, corr = sc.calculate_co_spectrum(
        ...     col1="Uz",
        ...     col2="Ultra_CH4_ppm_C",
        ...     lag_second=0.1
        ... )
        """
        freqs, co_spectrum, _, corr_coef = self.calculate_cross_spectrum(
            col1=col1,
            col2=col2,
            dimensionless=dimensionless,
            frequency_weighted=frequency_weighted,
            interpolate_points=interpolate_points,
            scaling=scaling,
            detrend_1st=detrend_1st,
            detrend_2nd=detrend_2nd,
            apply_lag_correction_to_col2=apply_lag_correction_to_col2,
            lag_second=lag_second,
        )
        return freqs, co_spectrum, corr_coef

    def calculate_cross_spectrum(
        self,
        col1: str,
        col2: str,
        dimensionless: bool = True,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        scaling: str = "spectrum",
        detrend_1st: bool = True,
        detrend_2nd: bool = False,
        apply_lag_correction_to_col2: bool = True,
        lag_second: float | None = None,
    ) -> tuple:
        """
        指定されたcol1とcol2のクロススペクトルをDataFrameから計算するためのメソッド。

        Parameters
        ----------
            col1: str
                データの列名1
            col2: str
                データの列名2
            dimensionless: bool, optional
                分散で割って無次元化を行うかのフラグ。デフォルトはTrueで無次元化を行う
            frequency_weighted: bool, optional
                周波数の重みづけを適用するかのフラグ。デフォルトはTrueで重みづけを行う
            interpolate_points: bool, optional
                対数軸上で等間隔なデータ点を生成するかのフラグ。デフォルトはTrueで等間隔点を生成する
            scaling: str, optional
                スペクトルのスケーリング方法。"density"でスペクトル密度、"spectrum"でスペクトル。デフォルトは"spectrum"
            detrend_1st: bool, optional
                1次トレンドを除去するかのフラグ。デフォルトはTrueで除去を行う
            detrend_2nd: bool, optional
                2次トレンドを除去するかのフラグ。デフォルトはFalseで除去を行わない
            apply_lag_correction_to_col2: bool, optional
                col2に遅れ時間補正を適用するかのフラグ。デフォルトはTrueで補正を行う
            lag_second: float | None, optional
                col1からcol2が遅れている時間(秒)。apply_lag_correction_to_col2がTrueの場合に必要。デフォルトはNone

        Returns
        ----------
            tuple[np.ndarray, np.ndarray, np.ndarray, float]
                以下の4つの要素を含むタプル:
                - freqs: 周波数軸(対数スケールの場合は対数変換済み)
                - co_spectrum: コスペクトル(対数スケールの場合は対数変換済み)
                - quad_spectrum: クアドラチャスペクトル
                - corr_coef: 変数の相関係数

        Examples
        --------
        >>> sc = SpectrumCalculator(df=data, fs=10)
        >>> freqs, co_spec, quad_spec, corr = sc.calculate_cross_spectrum(
        ...     col1="Uz",
        ...     col2="Ultra_CH4_ppm_C",
        ...     lag_second=0.1
        ... )
        """
        # バリデーション
        valid_scaling_options = ["density", "spectrum"]
        if scaling not in valid_scaling_options:
            raise ValueError(
                f"'scaling'は次のパラメータから選択してください: {valid_scaling_options}"
            )

        fs: float = self._fs
        df_internal: pd.DataFrame = self._df.copy()
        # データ取得と前処理
        data1: np.ndarray = np.array(df_internal[col1].values)
        data2: np.ndarray = np.array(df_internal[col2].values)

        # 遅れ時間の補正
        if apply_lag_correction_to_col2:
            if lag_second is None:
                raise ValueError(
                    "apply_lag_correction_to_col2=True の場合は lag_second に有効な遅れ時間(秒)を指定してください。"
                )
            data1, data2 = SpectrumCalculator._correct_lag_time(
                data1=data1, data2=data2, fs=fs, lag_second=lag_second
            )

        # トレンド除去
        if detrend_1st or detrend_2nd:
            data1 = SpectrumCalculator._detrend(
                data=data1, first=detrend_1st, second=detrend_2nd
            )
            data2 = SpectrumCalculator._detrend(
                data=data2, first=detrend_1st, second=detrend_2nd
            )

        # 相関係数の計算
        corr_coef: float = np.corrcoef(data1, data2)[0, 1]

        # クロススペクトル計算
        freqs, p_xy = signal.csd(
            data1,
            data2,
            fs=self._fs,
            window=self._window_type,
            nperseg=1024,
            scaling=scaling,
        )

        # コスペクトルとクアドラチャスペクトルの抽出
        co_spectrum = np.real(p_xy)
        quad_spectrum = np.imag(p_xy)

        # 周波数の重みづけ
        if frequency_weighted:
            co_spectrum[1:] *= freqs[1:]
            quad_spectrum[1:] *= freqs[1:]

        # 無次元化
        if dimensionless:
            cov_matrix: np.ndarray = np.cov(data1, data2)
            covariance: float = cov_matrix[0, 1]
            co_spectrum /= covariance
            quad_spectrum /= covariance

        if interpolate_points:
            # 補間処理
            log_freq_min = np.log10(0.001)
            log_freq_max = np.log10(freqs[-1])
            log_freq_resampled = np.logspace(log_freq_min, log_freq_max, self._plots)

            # スペクトルの補間
            co_resampled = np.interp(
                log_freq_resampled, freqs, co_spectrum, left=np.nan, right=np.nan
            )
            quad_resampled = np.interp(
                log_freq_resampled, freqs, quad_spectrum, left=np.nan, right=np.nan
            )

            # NaNを除外
            valid_mask = ~np.isnan(co_resampled)
            freqs = log_freq_resampled[valid_mask]
            co_spectrum = co_resampled[valid_mask]
            quad_spectrum = quad_resampled[valid_mask]

        # 0Hz成分を除外
        nonzero_mask = freqs != 0
        freqs = freqs[nonzero_mask]
        co_spectrum = co_spectrum[nonzero_mask]
        quad_spectrum = quad_spectrum[nonzero_mask]

        return freqs, co_spectrum, quad_spectrum, corr_coef

    def calculate_power_spectrum(
        self,
        col: str,
        dimensionless: bool = True,
        frequency_weighted: bool = True,
        interpolate_points: bool = True,
        scaling: str = "spectrum",
        detrend_1st: bool = True,
        detrend_2nd: bool = False,
    ) -> tuple:
        """指定されたcolに基づいてDataFrameからパワースペクトルと周波数軸を計算します。
        scipy.signal.welchを使用してパワースペクトルを計算します。

        Parameters
        ----------
            col: str
                パワースペクトルを計算するデータの列名
            dimensionless: bool, optional
                分散で割って無次元化を行うかどうか。デフォルトはTrueで無次元化を行います。
            frequency_weighted: bool, optional
                周波数の重みづけを適用するかどうか。デフォルトはTrueで重みづけを行います。
            interpolate_points: bool, optional
                対数軸上で等間隔なデータ点を生成するかどうか。デフォルトはTrueで等間隔化を行います。
            scaling: str, optional
                スペクトルの計算方法。"density"でスペクトル密度、"spectrum"でスペクトル。デフォルトは"spectrum"です。
            detrend_1st: bool, optional
                1次トレンドを除去するかどうか。デフォルトはTrueで除去を行います。
            detrend_2nd: bool, optional
                2次トレンドを除去するかどうか。デフォルトはFalseで除去を行いません。

        Returns
        ----------
            tuple
                以下の要素を含むタプル:
                - freqs (np.ndarray): 周波数軸(対数スケールの場合は対数変換済み)
                - power_spectrum (np.ndarray): パワースペクトル(対数スケールの場合は対数変換済み)

        Examples
        --------
        >>> sc = SpectrumCalculator(df=data_frame, fs=10)
        >>> freqs, power = sc.calculate_power_spectrum(
        ...     col="Uz",
        ...     dimensionless=True,
        ...     frequency_weighted=True
        ... )
        """
        # バリデーション
        valid_scaling_options = ["density", "spectrum"]
        if scaling not in valid_scaling_options:
            raise ValueError(
                f"'scaling'は次のパラメータから選択してください: {valid_scaling_options}"
            )

        # データの取得とトレンド除去
        df_internal: pd.DataFrame = self._df.copy()
        data: np.ndarray = np.array(df_internal[col].values)
        # どちらか一方でもTrueの場合は適用
        if detrend_1st or detrend_2nd:
            data = SpectrumCalculator._detrend(
                data=data, first=detrend_1st, second=detrend_2nd
            )

        # welchメソッドでパワースペクトル計算
        freqs, power_spectrum = signal.welch(
            data, fs=self._fs, window=self._window_type, nperseg=1024, scaling=scaling
        )

        # 周波数の重みづけ(0Hz除外の前に実施)
        if frequency_weighted:
            power_spectrum = freqs * power_spectrum

        # 無次元化(0Hz除外の前に実施)
        if dimensionless:
            variance = np.var(data)
            power_spectrum /= variance

        if interpolate_points:
            # 補間処理(0Hz除外の前に実施)
            log_freq_min = np.log10(0.001)
            log_freq_max = np.log10(freqs[-1])
            log_freq_resampled = np.logspace(log_freq_min, log_freq_max, self._plots)

            power_spectrum_resampled = np.interp(
                log_freq_resampled, freqs, power_spectrum, left=np.nan, right=np.nan
            )

            # NaNを除外
            valid_mask = ~np.isnan(power_spectrum_resampled)
            freqs = log_freq_resampled[valid_mask]
            power_spectrum = power_spectrum_resampled[valid_mask]

        # 0Hz成分を最後に除外
        nonzero_mask = freqs != 0
        freqs = freqs[nonzero_mask]
        power_spectrum = power_spectrum[nonzero_mask]

        return freqs, power_spectrum

    @staticmethod
    def _correct_lag_time(
        data1: np.ndarray,
        data2: np.ndarray,
        fs: float,
        lag_second: float,
    ) -> tuple:
        """
        相互相関関数を用いて遅れ時間を補正する。クロススペクトルの計算に使用。

        Parameters
        ----------
            data1: np.ndarray
                基準データ
            data2: np.ndarray
                遅れているデータ
            fs: float
                サンプリング周波数
            lag_second: float
                data1からdata2が遅れている時間(秒)。負の値は許可されない。

        Returns
        ----------
            tuple
                - data1: np.ndarray
                    基準データ(シフトなし)
                - data2: np.ndarray
                    補正された遅れているデータ
        """
        if lag_second < 0:
            raise ValueError("lag_second must be non-negative.")

        # lag_secondをサンプリング周波数でスケーリングしてインデックスに変換
        lag_index: int = int(lag_second * fs)

        # データの長さを取得
        data_length = len(data1)

        # data2のみをシフト(NaNで初期化)
        shifted_data2 = np.full(data_length, np.nan)
        shifted_data2[:-lag_index] = data2[lag_index:] if lag_index > 0 else data2

        # NaNを含まない部分のみを抽出
        valid_mask = ~np.isnan(shifted_data2)
        data1 = data1[valid_mask]
        data2 = shifted_data2[valid_mask]

        return data1, data2

    @staticmethod
    def _detrend(
        data: np.ndarray, first: bool = True, second: bool = False
    ) -> np.ndarray:
        """
        データから一次トレンドおよび二次トレンドを除去します。

        Parameters
        ----------
            data: np.ndarray
                入力データ
            first: bool, optional
                一次トレンドを除去するかどうか。デフォルトはTrue。
            second: bool, optional
                二次トレンドを除去するかどうか。デフォルトはFalse。

        Returns
        ----------
            np.ndarray
                トレンド除去後のデータ

        Raises
        ----------
            ValueError
                first と second の両方がFalseの場合
        """
        if not (first or second):
            raise ValueError("少なくとも一次または二次トレンドの除去を指定してください")

        detrended_data: np.ndarray = data.copy()

        # 一次トレンドの除去
        if first:
            detrended_data = signal.detrend(detrended_data)

        # 二次トレンドの除去
        if second:
            # 二次トレンドを除去するために、まず一次トレンドを除去
            detrended_data = signal.detrend(detrended_data, type="linear")
            # 二次トレンドを除去するために、二次多項式フィッティングを行う
            coeffs_second = np.polyfit(
                np.arange(len(detrended_data)), detrended_data, 2
            )
            trend_second = np.polyval(coeffs_second, np.arange(len(detrended_data)))
            detrended_data = detrended_data - trend_second

        return detrended_data

    @staticmethod
    def _generate_window_function(
        type: WindowFunctionType, data_length: int
    ) -> np.ndarray:
        """
        指定された種類の窓関数を適用する

        Parameters
        ----------
            type: WindowFunctionType
                窓関数の種類
            data_length: int
                データ長

        Returns
        ----------
            np.ndarray
                適用された窓関数

        Notes
        ----------
            - 指定された種類の窓関数を適用し、numpy配列として返す
            - 無効な種類が指定された場合、hanning窓を使用する
        """
        if type == "hamming":
            return np.hamming(data_length)
        elif type == "blackman":
            return np.blackman(data_length)
        return np.hanning(data_length)

    @staticmethod
    def _smooth_spectrum(
        spectrum: np.ndarray, frequencies: np.ndarray, freq_threshold: float = 0.1
    ) -> np.ndarray:
        """
        高周波数領域に対して3点移動平均を適用する処理を行う。
        この処理により、高周波数成分のノイズを低減し、スペクトルの滑らかさを向上させる。

        Parameters
        ----------
            spectrum: np.ndarray
                スペクトルデータ
            frequencies: np.ndarray
                対応する周波数データ
            freq_threshold: float, optional
                高周波数の閾値。デフォルトは0.1。

        Returns
        ----------
            np.ndarray
                スムーズ化されたスペクトルデータ
        """
        smoothed = spectrum.copy()  # オリジナルデータのコピーを作成

        # 周波数閾値以上の部分のインデックスを取得
        high_freq_mask = frequencies >= freq_threshold

        # 高周波数領域のみを処理
        high_freq_indices = np.where(high_freq_mask)[0]
        if len(high_freq_indices) > 2:  # 最低3点必要
            for i in high_freq_indices[1:-1]:  # 端点を除く
                smoothed[i] = (
                    0.25 * spectrum[i - 1] + 0.5 * spectrum[i] + 0.25 * spectrum[i + 1]
                )

            # 高周波領域の端点の処理
            first_idx = high_freq_indices[0]
            last_idx = high_freq_indices[-1]
            smoothed[first_idx] = 0.5 * (spectrum[first_idx] + spectrum[first_idx + 1])
            smoothed[last_idx] = 0.5 * (spectrum[last_idx - 1] + spectrum[last_idx])

        return smoothed
