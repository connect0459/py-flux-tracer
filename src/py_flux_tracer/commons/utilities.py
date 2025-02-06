import matplotlib.pyplot as plt
from typing import Any
from logging import getLogger, Formatter, Logger, StreamHandler, INFO


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
        logger : Logger | None
            使用するロガー。Noneの場合は新しいロガーを作成します。
        log_level : int
            ロガーのログレベル。デフォルトはINFO。

    Returns
    ----------
        Logger
            設定されたロガーオブジェクト。
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
    font_family: list[str] = ["Arial", "MS Gothic", "Dejavu Sans"],
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
        font_family : list[str]
            使用するフォントファミリーのリスト。
        font_size : float
            軸ラベルのフォントサイズ。
        legend_size : float
            凡例のフォントサイズ。
        tick_size : float
            軸目盛りのフォントサイズ。
        title_size : float
            タイトルのフォントサイズ。
        plot_params : dict[str, Any] | None
            matplotlibのプロットパラメータの辞書。
    """
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
