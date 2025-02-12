from .campbell.eddy_data_figures_generator import (
    EddyDataFiguresGenerator,
    SlopeLine,
    SpectralPlotConfig,
)
from .campbell.eddy_data_preprocessor import EddyDataPreprocessor, MeasuredWindKeyType
from .campbell.spectrum_calculator import SpectrumCalculator, WindowFunctionType
from .commons.utilities import setup_logger, setup_plot_params
from .footprint.flux_footprint_analyzer import FluxFootprintAnalyzer
from .mobile.correcting_utils import (
    BiasRemovalConfig,
    CorrectingUtils,
    H2OCorrectionConfig,
)
from .mobile.hotspot_emission_analyzer import (
    EmissionData,
    EmissionFormula,
    HotspotEmissionAnalyzer,
    HotspotEmissionConfig,
)
from .mobile.mobile_measurement_analyzer import (
    HotspotData,
    HotspotParams,
    HotspotType,
    KMLGeneratorConfig,
    MobileMeasurementAnalyzer,
    MobileMeasurementConfig,
)
from .monthly.monthly_converter import MonthlyConverter
from .monthly.monthly_figures_generator import MonthlyFiguresGenerator
from .transfer_function.fft_files_reorganizer import FftFileReorganizer
from .transfer_function.transfer_function_calculator import (
    TfCurvesFromCsvConfig,
    TransferFunctionCalculator,
)

"""
versionを動的に設定する。
`./_version.py`がない場合はsetuptools_scmを用いてGitからバージョン取得を試行
それも失敗した場合にデフォルトバージョン(0.0.0)を設定
"""
try:
    from ._version import __version__  # type:ignore
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(root="..", relative_to=__file__)
    except Exception:
        __version__ = "0.0.0"

"""
py-flux-tracer: 大気中のメタン・エタン濃度の解析ツール

このパッケージは、大気中のメタン(CH4)とエタン(C2H6)の濃度データを解析するための
包括的なツールセットを提供します。主な機能は以下の通りです:

主な機能:
- 渦相関法による乱流データの解析
- フラックスフットプリントの計算
- 車載濃度観測データの解析とホットスポット検出
  - KMLファイルの生成による空間分布の可視化
- 月間データの集計と可視化
- スペクトル解析と伝達関数の計算(Flux Calculatorの出力ファイルに依存)

サブモジュール:
- campbell: 渦相関法による乱流データの解析
- commons: 共通のユーティリティ関数
- footprint: フラックスフットプリントの計算
- mobile: モバイル測定データの解析とホットスポット検出
- monthly: 月間データの集計と可視化
- transfer_function: スペクトル解析とトランスファー関数の計算

詳細な使用方法については、各モジュールのドキュメントを参照してください。
"""

# モジュールを __all__ にセット
__all__ = [
    "BiasRemovalConfig",
    "CorrectingUtils",
    "EddyDataFiguresGenerator",
    "EddyDataPreprocessor",
    "EmissionData",
    "EmissionFormula",
    "FftFileReorganizer",
    "FluxFootprintAnalyzer",
    "H2OCorrectionConfig",
    "HotspotData",
    "HotspotEmissionAnalyzer",
    "HotspotEmissionConfig",
    "HotspotParams",
    "HotspotType",
    "KMLGeneratorConfig",
    "MeasuredWindKeyType",
    "MobileMeasurementAnalyzer",
    "MobileMeasurementConfig",
    "MonthlyConverter",
    "MonthlyFiguresGenerator",
    "SlopeLine",
    "SpectralPlotConfig",
    "SpectrumCalculator",
    "TfCurvesFromCsvConfig",
    "TransferFunctionCalculator",
    "WindowFunctionType",
    "__version__",
    "setup_logger",
    "setup_plot_params",
]
