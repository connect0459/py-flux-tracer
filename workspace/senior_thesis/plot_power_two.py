import os
from py_flux_tracer import (
    MonthlyConverter,
    MonthlyFiguresGenerator,
)

include_end_date: bool = True
start_date, end_date = "2024-05-15", "2024-11-30"  # yyyy-MM-ddで指定
months: list[str] = [
    "05_06",
    "07_08",
    "09_10",
    "11_12",
]
subplot_labels: list[list[str]] = [
    ["(a1)", "(a2)"],
    ["(b1)", "(b2)"],
    ["(c1)", "(c2)"],
    ["(d1)", "(d2)"],
]
lags_list: list[int] = [9.2, 10.0, 10.0, 10.0, 11.7, 13.2, 15.5]
output_dir = (
    "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/outputs"
)

# フラグ
plot_spectra: bool = True

if __name__ == "__main__":
    # Ultra
    with MonthlyConverter(
        "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Ultra.*.xlsx",
    ) as converter:
        df_ultra = converter.read_sheets(
            # sheet_names=["Final", "Final.SA"],
            # columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4"],
            sheet_names=["Final"],
            columns=["Fch4_ultra", "Fc2h6_ultra", "Fch4_open"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )

    # Picarro
    with MonthlyConverter(
        "/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/monthly",
        file_pattern="SA.Picaro.*.xlsx",
    ) as converter:
        df_picarro = converter.read_sheets(
            sheet_names=["Final"],
            columns=["Fch4_picaro"],
            start_date=start_date,
            end_date=end_date,
            include_end_date=include_end_date,
        )
        # print(df_picarro.head(10))

    # 両方を結合したDataFrameを明示的に作成
    df_combined = MonthlyConverter.merge_dataframes(df1=df_ultra, df2=df_picarro)

    # print("------")
    # print(df_combined.head(10))  # DataFrameの先頭10行を表示

    mfg = MonthlyFiguresGenerator()
    MonthlyFiguresGenerator.setup_plot_params(font_size=22, tick_size=18)

    for month, lag_sec, subplot_label in zip(months, lags_list, subplot_labels):
        # monthを0埋めのMM形式に変換
        month_str = str(month)
        mfg.logger.info(f"{month_str}の処理を開始します。")

        if plot_spectra:
            # パワースペクトルのプロット
            mfg.plot_spectra(
                input_dir=f"/home/connect0459/labo/py-flux-tracer/workspace/senior_thesis/private/data/eddy_csv-resampled-two-{month_str}",
                output_dir=(os.path.join(output_dir, "spectra", "two")),
                output_basename=f"spectrum-two-{month}",
                fs=10,
                lag_second=lag_sec,
                label_ch4=None,
                label_c2h6=None,
            )
            mfg.logger.info("'spectra'を作成しました。")
