from .campbell.eddy_data_preprocessor import EddyDataPreprocessor
from .campbell.spectrum_calculator import SpectrumCalculator
from .commons.hotspot_data import HotspotData, HotspotType
from .footprint.flux_footprint_analyzer import FluxFootprintAnalyzer
from .mobile.correcting_utils import CorrectingUtils, CORRECTION_TYPES_PATTERN
from .mobile.mobile_spatial_analyzer import (
    EmissionData,
    MobileSpatialAnalyzer,
    MSAInputConfig,
)
from .monthly.monthly_converter import MonthlyConverter
from .monthly.monthly_figures_generator import MonthlyFiguresGenerator
from .transfer_function.fft_files_reorganizer import FftFileReorganizer
from .transfer_function.transfer_function_calculator import TransferFunctionCalculator

"""
versionを動的に設定する。
`./_version.py`がない場合はsetuptools_scmを用いてGitからバージョン取得を試行
それも失敗した場合にデフォルトバージョン（0.0.0）を設定
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
    "__version__",
    "EddyDataPreprocessor",
    "SpectrumCalculator",
    "HotspotData",
    "HotspotType",
    "FluxFootprintAnalyzer",
    "CorrectingUtils",
    "CORRECTION_TYPES_PATTERN",
    "EmissionData",
    "MobileSpatialAnalyzer",
    "MSAInputConfig",
    "MonthlyConverter",
    "MonthlyFiguresGenerator",
    "FftFileReorganizer",
    "TransferFunctionCalculator",
]
