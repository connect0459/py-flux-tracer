import pandas as pd
from pathlib import Path
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class MonthlyConverter:
    """
    Monthlyシート（Excel）を一括で読み込み、DataFrameに変換するクラス。
    デフォルトは'SA.Ultra.*.xlsx'に対応していますが、コンストラクタのfile_patternを
    変更すると別のシートにも対応可能です（例: 'SA.Picaro.*.xlsx'）。
    """

    FILE_DATE_FORMAT = "%Y.%m"  # ファイル名用
    PERIOD_DATE_FORMAT = "%Y-%m-%d"  # 期間指定用

    def __init__(
        self,
        directory: str | Path,
        file_pattern: str = "SA.Ultra.*.xlsx",
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        MonthlyConverterクラスのコンストラクタ

        Parameters:
        ------
            directory : str | Path
                Excelファイルが格納されているディレクトリのパス
            file_pattern : str
                ファイル名のパターン。デフォルトは'SA.Ultra.*.xlsx'。
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            logging_debug : bool
                ログレベルを"DEBUG"に設定するかどうか。デフォルトはFalseで、Falseの場合はINFO以上のレベルのメッセージが出力されます。
        """
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = MonthlyConverter.setup_logger(logger, log_level)

        self._directory = Path(directory)
        if not self._directory.exists():
            raise NotADirectoryError(f"Directory not found: {self._directory}")

        # Excelファイルのパスを保持
        self._excel_files: dict[str, pd.ExcelFile] = {}
        self._file_pattern: str = file_pattern

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
        """
        ロガーを設定します。

        このメソッドは、ロギングの設定を行い、ログメッセージのフォーマットを指定します。
        ログメッセージには、日付、ログレベル、メッセージが含まれます。

        渡されたロガーがNoneまたは不正な場合は、新たにロガーを作成し、標準出力に
        ログメッセージが表示されるようにStreamHandlerを追加します。ロガーのレベルは
        引数で指定されたlog_levelに基づいて設定されます。

        Parameters:
        ------
            logger : Logger | None
                使用するロガー。Noneの場合は新しいロガーを作成します。
            log_level : int
                ロガーのログレベル。デフォルトはINFO。

        Returns:
        ------
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

        Returns:
        ------
            list[str]
                'yyyy.MM'形式の日付リスト
        """
        dates = []
        for file_name in self._directory.glob(self._file_pattern):
            try:
                date = self._extract_date(file_name.name)
                dates.append(date.strftime(self.FILE_DATE_FORMAT))
            except ValueError:
                continue
        return sorted(dates)

    def get_sheet_names(self, file_name: str) -> list[str]:
        """
        指定されたファイルで利用可能なシート名の一覧を返却する

        Parameters:
        ------
            file_name : str
                Excelファイル名

        Returns:
        ------
            list[str]
                シート名のリスト
        """
        if file_name not in self._excel_files:
            file_path = self._directory / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            self._excel_files[file_name] = pd.ExcelFile(file_path)
        return self._excel_files[file_name].sheet_names

    def read_sheets(
        self,
        sheet_names: str | list[str],
        columns: list[str] | None = None,  # 新しいパラメータを追加
        col_datetime: str = "Date",
        header: int = 0,
        skiprows: int | list[int] = [1],
        start_date: str | None = None,
        end_date: str | None = None,
        include_end_date: bool = True,
        sort_by_date: bool = True,
    ) -> pd.DataFrame:
        """
        指定されたシートを読み込み、DataFrameとして返却します。
        デフォルトでは2行目（単位の行）はスキップされます。
        重複するカラム名がある場合は、より先に指定されたシートに存在するカラムの値を保持します。

        Parameters:
        ------
            sheet_names : str | list[str]
                読み込むシート名。文字列または文字列のリストを指定できます。
            columns : list[str] | None
                残すカラム名のリスト。Noneの場合は全てのカラムを保持します。
            col_datetime : str
                日付と時刻の情報が含まれるカラム名。デフォルトは'Date'。
            header : int
                データのヘッダー行を指定します。デフォルトは0。
            skiprows : int | list[int]
                スキップする行数。デフォルトでは1行目をスキップします。
            start_date : str | None
                開始日 ('yyyy-MM-dd')。この日付の'00:00:00'のデータが開始行となります。
            end_date : str | None
                終了日 ('yyyy-MM-dd')。この日付をデータに含めるかはinclude_end_dateフラグによって変わります。
            include_end_date : bool
                終了日を含めるかどうか。デフォルトはTrueです。
            sort_by_date : bool
                ファイルの日付でソートするかどうか。デフォルトはTrueです。

        Returns:
        ------
            pd.DataFrame
                読み込まれたデータを結合したDataFrameを返します。
        """
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
        for file_name, excel_file in sorted_files:
            file_date = self._extract_date(file_name)

            for sheet_name in sheet_names:
                if sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(
                        excel_file,
                        sheet_name=sheet_name,
                        header=header,
                        skiprows=skiprows,
                        na_values=[
                            "#DIV/0!",
                            "#VALUE!",
                            "#REF!",
                            "#N/A",
                            "#NAME?",
                            "NAN",
                        ],
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

    def _extract_date(self, file_name: str) -> datetime:
        """
        ファイル名から日付を抽出する

        Parameters:
        ------
            file_name : str
                "SA.Ultra.yyyy.MM.xlsx"または"SA.Picaro.yyyy.MM.xlsx"形式のファイル名

        Returns:
        ------
            datetime
                抽出された日付
        """
        # ファイル名から日付部分を抽出
        date_str = ".".join(file_name.split(".")[-3:-1])  # "yyyy.MM"の部分を取得
        return datetime.strptime(date_str, self.FILE_DATE_FORMAT)

    def _load_excel_files(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> None:
        """
        指定された日付範囲のExcelファイルを読み込む

        Parameters:
        ------
            start_date : str | None
                開始日 ('yyyy-MM-dd'形式)
            end_date : str | None
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

        Parameters:
        ------
            df : pd.DataFrame
                入力データフレーム。
            target_months : list[int]
                抽出したい月のリスト（1から12の整数）。
            start_day : int | None
                開始日（1から31の整数）。Noneの場合は月初め。
            end_day : int | None
                終了日（1から31の整数）。Noneの場合は月末。
            datetime_column : str, optional
                日付を含む列の名前。デフォルトは"Date"。

        Returns:
        ------
            pd.DataFrame
                指定された期間のデータのみを含むデータフレーム。
        """
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
        df = df.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])

        # 月でフィルタリング
        monthly_data = df[df[datetime_column].dt.month.isin(target_months)]

        # 日付範囲でフィルタリング
        if start_day is not None:
            monthly_data = monthly_data[
                monthly_data[datetime_column].dt.day >= start_day
            ]
        if end_day is not None:
            monthly_data = monthly_data[monthly_data[datetime_column].dt.day <= end_day]

        return monthly_data

    @staticmethod
    def merge_dataframes(
        df1: pd.DataFrame, df2: pd.DataFrame, date_column: str = "Date"
    ) -> pd.DataFrame:
        """
        2つのDataFrameを結合します。重複するカラムは元の名前とサフィックス付きの両方を保持します。

        Parameters:
        ------
            df1 : pd.DataFrame
                ベースとなるDataFrame
            df2 : pd.DataFrame
                結合するDataFrame
            date_column : str
                日付カラムの名前。デフォルトは"Date"

        Returns:
        ------
            pd.DataFrame
                結合されたDataFrame
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
            # 元のカラムはdf1の値を保持（既に result に含まれている）
            # _x サフィックスでdf1の値を追加
            result[f"{col}_x"] = df1[col]
            # _y サフィックスでdf2の値を追加
            result[f"{col}_y"] = df2[col]

        return result
