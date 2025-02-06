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

__version__ = __version__
"""
@private
このモジュールはバージョン情報の管理に使用され、ドキュメントには含めません。
private属性を適用するために再宣言してdocstringを記述しています。
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
