import numpy as np
import pandas as pd

"""
CORRECTION_TYPES_PATTERN (list[str]): 補正式の種類を定義するリスト。correct_df_by_typeで使用する。
"""
CORRECTION_TYPES_PATTERN: list[str] = ["pico_1"]


class CorrectingUtils:
    @staticmethod
    def correct_df_by_type(df: pd.DataFrame, correction_type: str) -> pd.DataFrame:
        """
        指定された補正式に基づいてデータフレームを補正します。

        Parameters:
        ------
            df : pd.DataFrame
                補正対象のデータフレーム。
            correction_type : str
                適用する補正式の種類。CORRECTION_TYPES_PATTERNから選択する。

        Returns:
        ------
            pd.DataFrame
                補正後のデータフレーム。

        Raises:
        ------
            ValueError
                無効な補正式が指定された場合。
        """
        if correction_type == "pico_1":
            coef_a: float = 2.0631  # 切片
            coef_b: float = 1.0111e-06  # 1次の係数
            coef_c: float = -1.8683e-10  # 2次の係数
            # 水蒸気補正
            df_corrected: pd.DataFrame = CorrectingUtils._correct_h2o_interference(
                df=df,
                coef_a=coef_a,
                coef_b=coef_b,
                coef_c=coef_c,
                col_ch4="ch4_ppm",
                col_h2o="h2o_ppm",
                h2o_threshold=2000,
            )
            # 負の値のエタン濃度の補正など
            df_corrected = CorrectingUtils._remove_bias(
                df=df_corrected, col_ch4_ppm="ch4_ppm", col_c2h6_ppb="c2h6_ppb"
            )
            return df_corrected
        else:
            raise ValueError(f"invalid correction_type: {correction_type}.")

    @staticmethod
    def _correct_h2o_interference(
        df: pd.DataFrame,
        coef_a: float,
        coef_b: float,
        coef_c: float,
        col_ch4: str = "ch4_ppm",
        col_h2o: str = "h2o_ppm",
        h2o_threshold: float | None = 2000,
    ) -> pd.DataFrame:
        """
        水蒸気干渉を補正するためのメソッドです。
        CH4濃度に対する水蒸気の影響を2次関数を用いて補正します。

        References:
        ------
            - Commane et al. (2023): Intercomparison of commercial analyzers for atmospheric ethane and methane observations
                https://amt.copernicus.org/articles/16/1431/2023/,
                https://amt.copernicus.org/articles/16/1431/2023/amt-16-1431-2023.pdf

        Parameters:
        ------
            df : pd.DataFrame
                補正対象のデータフレーム
            coef_a : float
                補正曲線の切片
            coef_b : float
                補正曲線の1次係数
            coef_c : float
                補正曲線の2次係数
            col_ch4 : str
                CH4濃度を示すカラム名
            col_h2o : str
                水蒸気濃度を示すカラム名
            h2o_threshold : float | None
                水蒸気濃度の下限値（この値未満のデータは除外）

        Returns:
        ------
            pd.DataFrame
                水蒸気干渉が補正されたデータフレーム
        """
        # 元のデータを保護するためコピーを作成
        df = df.copy()
        # 水蒸気濃度の配列を取得
        h2o = np.array(df[col_h2o])

        # 補正項の計算
        correction_curve = coef_a + coef_b * h2o + coef_c * pow(h2o, 2)
        max_correction = np.max(correction_curve)
        correction_term = -(correction_curve - max_correction)

        # CH4濃度の補正
        df[col_ch4] = df[col_ch4] + correction_term

        # 極端に低い水蒸気濃度のデータは信頼性が低いため除外
        if h2o_threshold is not None:
            df.loc[df[col_h2o] < h2o_threshold, col_ch4] = np.nan
            df = df.dropna(subset=[col_ch4])

        return df

    @staticmethod
    def _remove_bias(
        df: pd.DataFrame,
        col_ch4_ppm: str = "ch4_ppm",
        col_c2h6_ppb: str = "c2h6_ppb",
    ) -> pd.DataFrame:
        """
        データフレームからバイアスを除去します。

        Parameters:
        ------
            df : pd.DataFrame
                バイアスを除去する対象のデータフレーム。
            col_ch4_ppm : str
                CH4濃度を示すカラム名。デフォルトは"ch4_ppm"。
            col_c2h6_ppb : str
                C2H6濃度を示すカラム名。デフォルトは"c2h6_ppb"。

        Returns:
        ------
            pd.DataFrame
                バイアスが除去されたデータフレーム。
        """
        df_processed: pd.DataFrame = df.copy()
        c2h6_min = np.percentile(df_processed[col_c2h6_ppb], 5)
        df_processed[col_c2h6_ppb] = df_processed[col_c2h6_ppb] - c2h6_min
        ch4_min = np.percentile(df_processed[col_ch4_ppm], 5)
        df_processed[col_ch4_ppm] = df_processed[col_ch4_ppm] - ch4_min + 2.0
        return df_processed
