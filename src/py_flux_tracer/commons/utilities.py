import os
from logging import INFO, Formatter, Logger, StreamHandler, getLogger
from pathlib import Path
from typing import Any

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
    """
    ロガーを設定します。

    このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
    ログメッセージには、日付、ログレベル、メッセージが含まれます。

    渡されたロガーが None または不正な場合は、新たにロガーを作成し、標準出力に
    ログメッセージが表示されるように StreamHandler を追加します。ロガーのレベルは
    引数で指定された log_level に基づいて設定されます。

    Parameters
    ----------
        logger: Logger | None
            使用するロガー。Noneの場合は新しいロガーを作成します。
        log_level: int, optional
            ロガーのログレベル。デフォルトはINFOで、ログの詳細度を制御します。

    Returns
    ----------
        Logger
            設定されたロガーオブジェクト。

    Examples
    --------
    >>> from logging import getLogger, INFO
    >>> logger = getLogger("my_logger")
    >>> configured_logger = setup_logger(logger, log_level=INFO)
    >>> configured_logger.info("ログメッセージ")
    2024-01-01 00:00:00,000 - INFO - ログメッセージ
    """
    if logger is not None and isinstance(logger, Logger):
        return logger
    # 渡されたロガーがNoneまたは正しいものでない場合は独自に設定
    new_logger: Logger = getLogger()
    # 既存のハンドラーをすべて削除
    for handler in new_logger.handlers[:]:
        new_logger.removeHandler(handler)
    new_logger.setLevel(log_level)  # ロガーのレベルを設定
    ch = StreamHandler()
    ch_formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)  # フォーマッターをハンドラーに設定
    new_logger.addHandler(ch)  # StreamHandlerの追加
    return new_logger


def setup_plot_params(
    font_family: list[str] | None = None,
    font_paths: list[str | Path] | None = None,
    font_size: float = 20,
    legend_size: float = 20,
    tick_size: float = 20,
    title_size: float = 20,
    plot_params: dict[str, Any] | None = None,
) -> None:
    """
    matplotlibのプロットパラメータを設定します。

    Parameters
    ----------
    font_family: list[str] | None, optional
        使用するフォントファミリーのリスト。デフォルト値は["Arial", "MS Gothic", "sans-serif"]です。
    font_paths: list[str | Path] | None, optional
        フォントファイルのパスのリスト。デフォルト値はNoneです。指定された場合、fontManagerでフォントを登録します。
    font_size: float, optional
        軸ラベルのフォントサイズ。デフォルト値は20です。
    legend_size: float, optional
        凡例のフォントサイズ。デフォルト値は20です。
    tick_size: float, optional
        軸目盛りのフォントサイズ。デフォルト値は20です。
    title_size: float, optional
        タイトルのフォントサイズ。デフォルト値は20です。
    plot_params: dict[str, Any] | None, optional
        matplotlibのプロットパラメータの辞書。デフォルト値はNoneです。指定された場合、デフォルトのパラメータに上書きされます。

    Returns
    -------
    None
        戻り値はありません。

    Examples
    --------
    >>> # デフォルト設定でプロットパラメータを設定
    >>> setup_plot_params()
    
    >>> # カスタムフォントとサイズを指定
    >>> setup_plot_params(
    ...     font_family=["Helvetica", "sans-serif"],
    ...     font_size=16,
    ...     legend_size=14
    ... )
    
    >>> # カスタムプロットパラメータを追加
    >>> custom_params = {"figure.figsize": (10, 6), "lines.linewidth": 2}
    >>> setup_plot_params(plot_params=custom_params)
    """
    # フォントファイルの登録
    if font_paths:
        for path in font_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The font file at {path} does not exist.")
            fm.fontManager.addfont(path)

    # デフォルト値の設定
    if font_family is None:
        font_family = ["Arial", "MS Gothic", "sans-serif"]

    # デフォルトのプロットパラメータ
    default_params = {
        "axes.linewidth": 1.0,
        "axes.titlesize": title_size,  # タイトル
        "grid.color": "gray",
        "grid.linewidth": 1.0,
        "font.family": font_family,
        "font.size": font_size,  # 軸ラベル
        "legend.fontsize": legend_size,  # 凡例
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.labelsize": tick_size,  # 軸目盛
        "ytick.labelsize": tick_size,  # 軸目盛
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "ytick.direction": "out",
        "ytick.major.width": 1.0,
    }

    # plot_paramsが定義されている場合、デフォルトに追記
    if plot_params:
        default_params.update(plot_params)

    plt.rcParams.update(default_params)  # プロットパラメータを更新
