from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class H2OCorrectionConfig:
    """水蒸気補正の設定を保持するデータクラス

    Parameters
    ----------
        coef_b: float | None, optional
            補正曲線の1次係数。デフォルト値はNone。
        coef_c: float | None, optional
            補正曲線の2次係数。デフォルト値はNone。
        h2o_ppm_threshold: float | None, optional
            水蒸気濃度の下限値(この値未満のデータは除外)。デフォルト値は2000。
        target_h2o_ppm: float, optional
            換算先の水蒸気濃度。デフォルト値は10000 ppm。

    Examples
    --------
    >>> config = H2OCorrectionConfig(
    ...     coef_b=0.001,
    ...     coef_c=0.0001,
    ...     h2o_ppm_threshold=2000,
    ...     target_h2o_ppm=10000
    ... )
    """

    coef_b: float | None = None
    coef_c: float | None = None
    h2o_ppm_threshold: float | None = 2000
    target_h2o_ppm: float = 10000


@dataclass
class BiasRemovalConfig:
    """バイアス除去の設定を保持するデータクラス

    Parameters
    ----------
        quantile_value: float, optional
            バイアス除去に使用するクォンタイル値。デフォルト値は0.05。
        base_ch4_ppm: float, optional
            補正前の値から最小値を引いた後に足すCH4濃度の基準値。デフォルト値は2.0。
        base_c2h6_ppb: float, optional
            補正前の値から最小値を引いた後に足すC2H6濃度の基準値。デフォルト値は0。

    Examples
    --------
    >>> config = BiasRemovalConfig(
    ...     quantile_value=0.05,
    ...     base_ch4_ppm=2.0,
    ...     base_c2h6_ppb=0
    ... )
    """

    quantile_value: float = 0.05
    base_ch4_ppm: float = 2.0
    base_c2h6_ppb: float = 0

    def __post_init__(self) -> None:
        """パラメータの検証を行います。

        Raises
        ----------
            ValueError: quantile_valueが0以上1未満でない場合
        """
        if not 0 <= self.quantile_value < 1:
            raise ValueError(
                f"quantile_value must be between 0 and 1, got {self.quantile_value}"
            )


class CorrectingUtils:
    """
    車載濃度観測で得たファイルのDataFrameを補正する関数をクラス化したものです。
    """

    @staticmethod
    def correct_h2o_interference(
        df: pd.DataFrame,
        coef_b: float,
        coef_c: float,
        col_ch4_ppm: str = "ch4_ppm",
        col_h2o_ppm: str = "h2o_ppm",
        h2o_ppm_threshold: float | None = 2000,
        target_h2o_ppm: float = 10000,
    ) -> pd.DataFrame:
        """
        水蒸気干渉を補正するためのメソッドです。
        CH4濃度を特定の水蒸気濃度下での値に換算します。

        Parameters
        ----------
            df: pd.DataFrame
                補正対象のデータフレーム
            coef_b: float
                補正曲線の1次係数
            coef_c: float
                補正曲線の2次係数
            col_ch4_ppm: str, optional
                CH4濃度を示すカラム名。デフォルト値は"ch4_ppm"
            col_h2o_ppm: str, optional
                水蒸気濃度を示すカラム名。デフォルト値は"h2o_ppm"
            h2o_ppm_threshold: float | None, optional
                水蒸気濃度の下限値(この値未満のデータは除外)。デフォルト値は2000 ppm
            target_h2o_ppm: float, optional
                換算先の水蒸気濃度。デフォルト値は10000 ppm

        Returns
        ----------
            pd.DataFrame
                水蒸気干渉が補正されたデータフレーム

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> # サンプルデータの作成
        >>> df = pd.DataFrame({
        ...     'ch4_ppm': [2.0, 2.1, 2.2],
        ...     'h2o_ppm': [5000, 6000, 7000]
        ... })
        >>> # 水蒸気干渉の補正
        >>> df_corrected = CorrectingUtils.correct_h2o_interference(
        ...     df=df,
        ...     coef_b=0.0001,
        ...     coef_c=0.00001
        ... )
        """
        # 元のデータを保護するためコピーを作成
        df_h2o_corrected: pd.DataFrame = df.copy()
        # 水蒸気濃度の配列を取得
        h2o: np.ndarray = np.array(df_h2o_corrected[col_h2o_ppm])

        # 現在の水蒸気濃度での補正項
        term_ch4 = coef_b * h2o + coef_c * np.power(h2o, 2)

        # 目標水蒸気濃度での補正項
        target_term = coef_b * target_h2o_ppm + coef_c * np.power(target_h2o_ppm, 2)

        # CH4濃度の補正(現在の補正項を引いて、目標水蒸気濃度での補正項を足す)
        df_h2o_corrected[col_ch4_ppm] = (
            df_h2o_corrected[col_ch4_ppm] - term_ch4 + target_term
        )

        # 極端に低い水蒸気濃度のデータは信頼性が低いため除外
        if h2o_ppm_threshold is not None:
            df_h2o_corrected.loc[df[col_h2o_ppm] < h2o_ppm_threshold, col_ch4_ppm] = (
                np.nan
            )
            df_h2o_corrected = df_h2o_corrected.dropna(subset=[col_ch4_ppm])

        return df_h2o_corrected

    @staticmethod
    def remove_bias(
        df: pd.DataFrame,
        col_ch4_ppm: str = "ch4_ppm",
        col_c2h6_ppb: str = "c2h6_ppb",
        base_ch4_ppm: float = 2.0,
        base_c2h6_ppb: float = 0,
        quantile_value: float = 0.05,
    ) -> pd.DataFrame:
        """
        データフレームからバイアスを除去します。

        Parameters
        ----------
            df: pd.DataFrame
                バイアスを除去する対象のデータフレーム
            col_ch4_ppm: str, optional
                CH4濃度を示すカラム名。デフォルト値は"ch4_ppm"
            col_c2h6_ppb: str, optional
                C2H6濃度を示すカラム名。デフォルト値は"c2h6_ppb"
            base_ch4_ppm: float, optional
                補正前の値から最小値を引いた後に足すCH4濃度。デフォルト値は2.0
            base_c2h6_ppb: float, optional
                補正前の値から最小値を引いた後に足すC2H6濃度。デフォルト値は0
            quantile_value: float, optional
                下位何クォンタイルの値を最小値として補正を行うか。デフォルト値は0.05

        Returns
        ----------
            pd.DataFrame
                バイアスが除去されたデータフレーム

        Examples
        --------
        >>> import pandas as pd
        >>> # サンプルデータの作成
        >>> df = pd.DataFrame({
        ...     'ch4_ppm': [2.1, 2.2, 2.3],
        ...     'c2h6_ppb': [1.1, 1.2, 1.3]
        ... })
        >>> # バイアスの除去
        >>> df_unbiased = CorrectingUtils.remove_bias(
        ...     df=df,
        ...     base_ch4_ppm=2.0,
        ...     base_c2h6_ppb=0
        ... )
        """
        df_internal: pd.DataFrame = df.copy()
        # CH4
        ch4_min: float = df_internal[col_ch4_ppm].quantile(quantile_value)
        df_internal[col_ch4_ppm] = df_internal[col_ch4_ppm] - ch4_min + base_ch4_ppm
        # C2H6
        c2h6_min: float = df_internal[col_c2h6_ppb].quantile(quantile_value)
        df_internal[col_c2h6_ppb] = df_internal[col_c2h6_ppb] - c2h6_min + base_c2h6_ppb
        return df_internal
