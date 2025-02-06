import warnings
from datetime import datetime, timedelta
from importlib.metadata import version
from logging import DEBUG, INFO, Logger
from pathlib import Path

import pandas as pd

from ..commons.utilities import setup_logger


class MonthlyConverter:
    """
    Monthlyシート(Excel)を一括で読み込み、DataFrameに変換するクラス。
    デフォルトは'SA.Ultra.*.xlsx'に対応していますが、コンストラクタのfile_patternを
    変更すると別のシートにも対応可能です(例: 'SA.Picaro.*.xlsx')。
    """

    FILE_DATE_FORMAT = "%Y.%m"  # ファイル名用
    PERIOD_DATE_FORMAT = "%Y-%m-%d"  # 期間指定用

    def __init__(
        self,
        directory: str | Path,
        file_pattern: str = "SA.Ultra.*.xlsx",
        na_values: list[str] | None = None,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        MonthlyConverterクラスのコンストラクタ

        Parameters
        ----------
            directory: str | Path
                Excelファイルが格納されているディレクトリのパス
            file_pattern: str, optional
                ファイル名のパターン。デフォルト値は'SA.Ultra.*.xlsx'です。
            na_values: list[str] | None, optional
                NaNと判定する値のパターン。デフォルト値はNoneで、その場合は以下の値が使用されます:
                [
                    "#DIV/0!",
                    "#VALUE!",
                    "#REF!",
                    "#N/A",
                    "#NAME?",
                    "NAN",
                    "nan",
                ]
            logger: Logger | None, optional
                使用するロガー。デフォルト値はNoneで、その場合は新しいロガーが作成されます。
            logging_debug: bool, optional
                ログレベルを"DEBUG"に設定するかどうか。デフォルト値はFalseで、その場合はINFO以上のレベルのメッセージが出力されます。

        Examples
        --------
        >>> converter = MonthlyConverter("path/to/excel/files")
        >>> converter = MonthlyConverter(
        ...     "path/to/excel/files",
        ...     file_pattern="SA.Picaro.*.xlsx",
        ...     logging_debug=True
        ... )
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

        if na_values is None:
            na_values = ["#DIV/0!", "#VALUE!", "#REF!", "#N/A", "#NAME?", "NAN", "nan"]
        self._na_values: list[str] = na_values
        self._directory = Path(directory)
        if not self._directory.exists():
            raise NotADirectoryError(f"Directory not found: {self._directory}")

        # Excelファイルのパスを保持
        self._excel_files: dict[str, pd.ExcelFile] = {}
        self._file_pattern: str = file_pattern

    def close(self) -> None:
        """
        すべてのExcelファイルをクローズする
        """
        for excel_file in self._excel_files.values():
            excel_file.close()
        self._excel_files.clear()

    def get_available_dates(self) -> list[str]:
        """
        利用可能なファイルの日付一覧を返却します。

        Returns
        ----------
            list[str]
                'yyyy.MM'形式の日付リスト

        Examples
        --------
        >>> converter = MonthlyConverter("path/to/excel/files")
        >>> dates = converter.get_available_dates()
        >>> print(dates)
        ['2023.01', '2023.02', '2023.03']
        """
        dates = []
        for filename in self._directory.glob(self._file_pattern):
            try:
                date = self._extract_date(filename.name)
                dates.append(date.strftime(self.FILE_DATE_FORMAT))
            except ValueError:
                continue
        return sorted(dates)

    def get_sheet_names(self, filename: str) -> list[str]:
        """
        指定されたファイルで利用可能なシート名の一覧を返却します。

        Parameters
        ----------
            filename: str
                対象のExcelファイル名を指定します。ファイル名のみを指定し、パスは含めません。

        Returns
        ----------
            list[str]
                シート名のリスト

        Examples
        --------
        >>> converter = MonthlyConverter("path/to/excel/files")
        >>> sheets = converter.get_sheet_names("2023.01.xlsx")
        >>> print(sheets)
        ['Sheet1', 'Sheet2', 'Sheet3']
        """
        if filename not in self._excel_files:
            filepath = self._directory / filename
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            self._excel_files[filename] = pd.ExcelFile(filepath)
        return [str(name) for name in self._excel_files[filename].sheet_names]

    def read_sheets(
        self,
        sheet_names: str | list[str],
        columns: list[str] | None = None,
        col_datetime: str = "Date",
        header: int = 0,
        skiprows: int | list[int] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        include_end_date: bool = True,
        sort_by_date: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたシートを読み込み、DataFrameとして返却します。
        デフォルトでは2行目(単位の行)はスキップされます。
        重複するカラム名がある場合は、より先に指定されたシートに存在するカラムの値を保持します。

        Parameters
        ----------
            sheet_names: str | list[str]
                読み込むシート名を指定します。文字列または文字列のリストを指定できます。
            columns: list[str] | None, optional
                残すカラム名のリストを指定します。Noneの場合は全てのカラムを保持します。
            col_datetime: str, optional
                日付と時刻の情報が含まれるカラム名を指定します。デフォルト値は'Date'です。
            header: int, optional
                データのヘッダー行を指定します。デフォルト値は0です。
            skiprows: int | list[int] | None, optional
                スキップする行数を指定します。Noneの場合のデフォルトでは2行目をスキップします。
            start_date: str | None, optional
                開始日を'yyyy-MM-dd'形式で指定します。この日付の'00:00:00'のデータが開始行となります。
            end_date: str | None, optional
                終了日を'yyyy-MM-dd'形式で指定します。この日付をデータに含めるかはinclude_end_dateフラグによって変わります。
            include_end_date: bool, optional
                終了日を含めるかどうかを指定します。デフォルト値はTrueです。
            sort_by_date: bool, optional
                ファイルの日付でソートするかどうかを指定します。デフォルト値はTrueです。

        Returns
        ----------
            pd.DataFrame
                読み込まれたデータを結合したDataFrameを返します。

        Examples
        --------
        >>> converter = MonthlyConverter("path/to/excel/files")
        >>> # 単一シートの読み込み
        >>> df = converter.read_sheets("Sheet1")
        >>> # 複数シートの読み込み
        >>> df = converter.read_sheets(["Sheet1", "Sheet2"])
        >>> # 特定の期間のデータ読み込み
        >>> df = converter.read_sheets(
        ...     "Sheet1",
        ...     start_date="2023-01-01",
        ...     end_date="2023-12-31"
        ... )
        """
        if skiprows is None:
            skiprows = [1]
        if isinstance(sheet_names, str):
            sheet_names = [sheet_names]

        self._load_excel_files(start_date, end_date)

        if not self._excel_files:
            raise ValueError("No Excel files found matching the criteria")

        # ファイルを日付順にソート
        sorted_files = (
            sorted(self._excel_files.items(), key=lambda x: self._extract_date(x[0]))
            if sort_by_date
            else self._excel_files.items()
        )

        # 各シートのデータを格納するリスト
        sheet_dfs = {sheet_name: [] for sheet_name in sheet_names}

        # 各ファイルからデータを読み込む
        for filename, excel_file in sorted_files:
            file_date = self._extract_date(filename)

            for sheet_name in sheet_names:
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=header,
                        skiprows=skiprows,
                        na_values=self._na_values,
                    )
                    # 年と月を追加
                    df["year"] = file_date.year
                    df["month"] = file_date.month
                    sheet_dfs[sheet_name].append(df)

        if not any(sheet_dfs.values()):
            raise ValueError(f"No sheets found matching: {sheet_names}")

        # 各シートのデータを結合
        combined_sheets = {}
        for sheet_name, dfs in sheet_dfs.items():
            if dfs:  # シートにデータがある場合のみ結合
                combined_sheets[sheet_name] = pd.concat(dfs, ignore_index=True)

        # 最初のシートをベースにする
        base_df = combined_sheets[sheet_names[0]]

        # 2つ目以降のシートを結合
        for sheet_name in sheet_names[1:]:
            if sheet_name in combined_sheets:
                base_df = self.merge_dataframes(
                    base_df, combined_sheets[sheet_name], date_column=col_datetime
                )

        # 日付でフィルタリング
        if start_date:
            start_dt = pd.to_datetime(start_date)
            base_df = base_df[base_df[col_datetime] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            if include_end_date:
                end_dt += pd.Timedelta(days=1)
            base_df = base_df[base_df[col_datetime] < end_dt]

        # カラムの選択
        if columns is not None:
            required_columns = [col_datetime, "year", "month"]
            available_columns = base_df.columns.tolist()  # 利用可能なカラムを取得
            if not all(col in available_columns for col in columns):
                raise ValueError(
                    f"指定されたカラムが見つかりません: {columns}. 利用可能なカラム: {available_columns}"
                )
            selected_columns = list(set(columns + required_columns))
            base_df = base_df[selected_columns]

        return base_df

    def __enter__(self) -> "MonthlyConverter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def get_last_day_of_month(year: int, month: int) -> int:
        """
        指定された年月の最終日を返します。

        Parameters
        ----------
            year: int
                年を指定します。
            month: int
                月を指定します。1から12の整数を指定してください。

        Returns
        ----------
            int
                指定された年月の最終日の日付。1から31の整数で返されます。

        Examples
        ----------
        >>> MonthlyConverter.get_last_day_of_month(2023, 2)
        28
        >>> MonthlyConverter.get_last_day_of_month(2024, 2)  # 閏年の場合
        29
        """
        next_month = (
            datetime(year, month % 12 + 1, 1)
            if month < 12
            else datetime(year + 1, 1, 1)
        )
        last_day = (next_month - timedelta(days=1)).day
        return last_day

    @staticmethod
    def extract_period_data(
        df: pd.DataFrame,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        include_end_date: bool = True,
        datetime_column: str = "Date",
    ) -> pd.DataFrame:
        """
        指定された期間のデータを抽出します。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム。
            start_date: str | pd.Timestamp
                開始日。YYYY-MM-DD形式の文字列またはTimestampで指定します。
            end_date: str | pd.Timestamp
                終了日。YYYY-MM-DD形式の文字列またはTimestampで指定します。
            include_end_date: bool, optional
                終了日を含めるかどうかを指定します。デフォルトはTrueです。
            datetime_column: str, optional
                日付を含む列の名前を指定します。デフォルトは"Date"です。

        Returns
        ----------
            pd.DataFrame
                指定された期間のデータのみを含むデータフレーム。

        Examples
        ----------
        >>> df = pd.DataFrame({
        ...     'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        ...     'Value': [1, 2, 3]
        ... })
        >>> MonthlyConverter.extract_period_data(
        ...     df, 
        ...     '2023-01-01', 
        ...     '2023-01-02'
        ... )
           Date  Value
        0  2023-01-01  1
        1  2023-01-02  2
        """
        # データフレームのコピーを作成
        df_internal = df.copy()
        df_internal[datetime_column] = pd.to_datetime(df_internal[datetime_column])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 開始日と終了日の順序チェック
        if start_dt > end_dt:
            raise ValueError("start_date は end_date より前である必要があります")

        # 期間でフィルタリング
        period_data = df_internal[
            (df_internal[datetime_column] >= start_dt)
            & (
                df_internal[datetime_column]
                < (end_dt + pd.Timedelta(days=1) if include_end_date else end_dt)
            )
        ]

        return period_data

    @staticmethod
    def merge_dataframes(
        df1: pd.DataFrame, df2: pd.DataFrame, date_column: str = "Date"
    ) -> pd.DataFrame:
        """
        2つのDataFrameを結合します。重複するカラムは元の名前とサフィックス付きの両方を保持します。

        Parameters
        ----------
            df1: pd.DataFrame
                ベースとなるデータフレームを指定します。
            df2: pd.DataFrame
                結合するデータフレームを指定します。
            date_column: str, optional
                日付カラムの名前を指定します。デフォルトは"Date"です。

        Returns
        ----------
            pd.DataFrame
                結合されたデータフレームを返します。重複するカラムには_xと_yのサフィックスが付与されます。

        Examples
        ----------
        >>> df1 = pd.DataFrame({
        ...     'Date': ['2023-01-01', '2023-01-02'],
        ...     'Value': [1, 2]
        ... })
        >>> df2 = pd.DataFrame({
        ...     'Date': ['2023-01-01', '2023-01-02'],
        ...     'Value': [10, 20]
        ... })
        >>> MonthlyConverter.merge_dataframes(df1, df2)
               Date  Value  Value_x  Value_y
        0  2023-01-01   1       1       10
        1  2023-01-02   2       2       20
        """
        # インデックスをリセット
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        # 日付カラムを統一
        df2[date_column] = df1[date_column]

        # 重複しないカラムと重複するカラムを分離
        duplicate_cols = [date_column, "year", "month"]  # 常に除外するカラム
        overlapping_cols = [
            col
            for col in df2.columns
            if col in df1.columns and col not in duplicate_cols
        ]
        unique_cols = [
            col
            for col in df2.columns
            if col not in df1.columns and col not in duplicate_cols
        ]

        # 結果のDataFrameを作成
        result = df1.copy()

        # 重複しないカラムを追加
        for col in unique_cols:
            result[col] = df2[col]

        # 重複するカラムを処理
        for col in overlapping_cols:
            # 元のカラムはdf1の値を保持(既に result に含まれている)
            # _x サフィックスでdf1の値を追加
            result[f"{col}_x"] = df1[col]
            # _y サフィックスでdf2の値を追加
            result[f"{col}_y"] = df2[col]

        return result

    def _extract_date(self, filename: str) -> datetime:
        """
        ファイル名から日付を抽出する

        Parameters
        ----------
            filename: str
                "SA.Ultra.yyyy.MM.xlsx"または"SA.Picaro.yyyy.MM.xlsx"形式のファイル名

        Returns
        ----------
            datetime
                抽出された日付
        """
        # ファイル名から日付部分を抽出
        date_str = ".".join(filename.split(".")[-3:-1])  # "yyyy.MM"の部分を取得
        return datetime.strptime(date_str, self.FILE_DATE_FORMAT)

    def _load_excel_files(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> None:
        """
        指定された日付範囲のExcelファイルを読み込む

        Parameters
        ----------
            start_date: str | None
                開始日 ('yyyy-MM-dd'形式)
            end_date: str | None
                終了日 ('yyyy-MM-dd'形式)
        """
        # 期間指定がある場合は、yyyy-MM-dd形式から年月のみを抽出
        start_dt = None
        end_dt = None
        if start_date:
            temp_dt = datetime.strptime(start_date, self.PERIOD_DATE_FORMAT)
            start_dt = datetime(temp_dt.year, temp_dt.month, 1)
        if end_date:
            temp_dt = datetime.strptime(end_date, self.PERIOD_DATE_FORMAT)
            end_dt = datetime(temp_dt.year, temp_dt.month, 1)

        # 既存のファイルをクリア
        self.close()

        for excel_path in self._directory.glob(self._file_pattern):
            try:
                file_date = self._extract_date(excel_path.name)

                # 日付範囲チェック
                if start_dt and file_date < start_dt:
                    continue
                if end_dt and file_date > end_dt:
                    continue

                if excel_path.name not in self._excel_files:
                    self._excel_files[excel_path.name] = pd.ExcelFile(excel_path)

            except ValueError as e:
                self.logger.warning(
                    f"Could not parse date from file {excel_path.name}: {e}"
                )

    @staticmethod
    def extract_monthly_data(
        df: pd.DataFrame,
        target_months: list[int],
        start_day: int | None = None,
        end_day: int | None = None,
        datetime_column: str = "Date",
    ) -> pd.DataFrame:
        """
        指定された月と期間のデータを抽出します。

        Parameters
        ----------
            df: pd.DataFrame
                入力データフレーム。
            target_months: list[int]
                抽出したい月のリスト(1から12の整数)。
            start_day: int | None
                開始日(1から31の整数)。Noneの場合は月初め。
            end_day: int | None
                終了日(1から31の整数)。Noneの場合は月末。
            datetime_column: str, optional
                日付を含む列の名前。デフォルトは"Date"。

        Returns
        ----------
            pd.DataFrame
                指定された期間のデータのみを含むデータフレーム。

        .. warning::
            このメソッドは非推奨です。代わりに `extract_period_data` を使用してください。
            v1.0.0 で削除される予定です。
        """
        try:
            ver = version("py_flux_tracer")
            # print(ver)
            if ver.startswith("0."):
                warnings.warn(
                    "`extract_monthly_data` is deprecated. Please use `extract_period_data` instead. This method will be removed in v1.0.0.",
                    FutureWarning,
                    stacklevel=2,  # 警告メッセージでライブラリの内部実装ではなく、非推奨のメソッドを使用しているユーザーのコードの行を指し示すことができる
                )
        except Exception:
            pass

        # 入力チェック
        if not all(1 <= month <= 12 for month in target_months):
            raise ValueError("target_monthsは1から12の間である必要があります")

        if start_day is not None and not 1 <= start_day <= 31:
            raise ValueError("start_dayは1から31の間である必要があります")

        if end_day is not None and not 1 <= end_day <= 31:
            raise ValueError("end_dayは1から31の間である必要があります")

        if start_day is not None and end_day is not None and start_day > end_day:
            raise ValueError("start_dayはend_day以下である必要があります")

        # datetime_column をDatetime型に変換
        df_internal = df.copy()
        df_internal[datetime_column] = pd.to_datetime(df_internal[datetime_column])

        # 月でフィルタリング
        monthly_data = df_internal[
            df_internal[datetime_column].dt.month.isin(target_months)
        ]

        # 日付範囲でフィルタリング
        if start_day is not None:
            monthly_data = monthly_data[
                monthly_data[datetime_column].dt.day >= start_day
            ]
        if end_day is not None:
            monthly_data = monthly_data[monthly_data[datetime_column].dt.day <= end_day]

        return monthly_data
