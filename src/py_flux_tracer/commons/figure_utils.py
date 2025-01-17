import matplotlib.pyplot as plt


class FigureUtils:
    @staticmethod
    def setup_plot_params(
        font_family: list[str] = ["Arial", "MS Gothic", "Dejavu Sans"],
        font_size: float = 20,
        legend_size: float = 20,
        tick_size: float = 20,
        title_size: float = 20,
        plot_params: dict[str, any] | None = None,
    ) -> None:
        """
        matplotlibのプロットパラメータを設定します。

        Parameters:
        ------
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
            plot_params : dict[str, any] | None
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
