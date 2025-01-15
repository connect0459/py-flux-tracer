import os
import re
import csv
import shutil
from tqdm import tqdm
from datetime import datetime
from logging import getLogger, Formatter, Logger, StreamHandler, DEBUG, INFO


class FftFileReorganizer:
    """
    FFTファイルを再編成するためのクラス。

    入力ディレクトリからファイルを読み取り、フラグファイルに基づいて
    出力ディレクトリに再編成します。時間の完全一致を要求し、
    一致しないファイルはスキップして警告を出します。
    オプションで相対湿度（RH）に基づいたサブディレクトリへの分類も可能です。
    """

    # クラス定数の定義
    DEFAULT_FILENAME_PATTERNS: list[str] = [
        r"FFT_TOA5_\d+\.SAC_Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?\.csv",
        r"FFT_TOA5_\d+\.SAC_Ultra\.Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?(?:-resampled)?\.csv",
    ]  # デフォルトのファイル名のパターン（正規表現）
    DEFAULT_OUTPUT_DIRS = {
        "GOOD_DATA": "good_data_all",
        "BAD_DATA": "bad_data",
    }  # 出力ディレクトリの構造に関する定数

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        flag_csv_path: str,
        filename_patterns: list[str] | None = None,
        output_dirs: dict[str, str] | None = None,
        sort_by_rh: bool = True,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        FftFileReorganizerクラスを初期化します。

        Parameters:
        ------
            input_dir : str
                入力ファイルが格納されているディレクトリのパス
            output_dir : str
                出力ファイルを格納するディレクトリのパス
            flag_csv_path : str
                フラグ情報が記載されているCSVファイルのパス
            filename_patterns : list[str] | None
                ファイル名のパターン（正規表現）のリスト
            output_dirs : dict[str, str] | None
                出力ディレクトリの構造を定義する辞書
            sort_by_rh : bool
                RHに基づいてサブディレクトリにファイルを分類するかどうか
            logger : Logger | None
                使用するロガー
            logging_debug : bool
                ログレベルをDEBUGに設定するかどうか
        """
        self._fft_path: str = input_dir
        self._sorted_path: str = output_dir
        self._output_dirs = output_dirs or self.DEFAULT_OUTPUT_DIRS
        self._good_data_path: str = os.path.join(
            output_dir, self._output_dirs["GOOD_DATA"]
        )
        self._bad_data_path: str = os.path.join(
            output_dir, self._output_dirs["BAD_DATA"]
        )
        self._filename_patterns: list[str] = (
            self.DEFAULT_FILENAME_PATTERNS.copy()
            if filename_patterns is None
            else filename_patterns
        )
        self._flag_file_path: str = flag_csv_path
        self._sort_by_rh: bool = sort_by_rh
        self._flags = {}
        self._warnings = []
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = FftFileReorganizer.setup_logger(logger, log_level)

    def reorganize(self):
        """
        ファイルの再編成プロセス全体を実行します。
        ディレクトリの準備、フラグファイルの読み込み、
        有効なファイルの取得、ファイルのコピーを順に行います。
        処理後、警告メッセージがあれば出力します。
        """
        self._prepare_directories()
        self._read_flag_file()
        valid_files = self._get_valid_files()
        self._copy_files(valid_files)
        self.logger.info("ファイルのコピーが完了しました。")

        if self._warnings:
            self.logger.warning("Warnings:")
            for warning in self._warnings:
                self.logger.warning(warning)

    def _copy_files(self, valid_files):
        """
        有効なファイルを適切な出力ディレクトリにコピーします。
        フラグファイルの時間と完全に一致するファイルのみを処理します。

        Parameters:
        ------
            valid_files : list
                コピーする有効なファイル名のリスト
        """
        with tqdm(total=len(valid_files)) as pbar:
            for filename in valid_files:
                src_file = os.path.join(self._fft_path, filename)
                file_time = self._parse_datetime(filename)

                if file_time in self._flags:
                    flag = self._flags[file_time]["Flg"]
                    rh = self._flags[file_time]["RH"]
                    if flag == 0:
                        # Copy to self._good_data_path
                        dst_file_good = os.path.join(self._good_data_path, filename)
                        shutil.copy2(src_file, dst_file_good)

                        if self._sort_by_rh:
                            # Copy to RH directory
                            rh_dir = FftFileReorganizer.get_rh_directory(rh)
                            dst_file_rh = os.path.join(
                                self._sorted_path, rh_dir, filename
                            )
                            shutil.copy2(src_file, dst_file_rh)
                    else:
                        dst_file = os.path.join(self._bad_data_path, filename)
                        shutil.copy2(src_file, dst_file)
                else:
                    self._warnings.append(
                        f"{filename} に対応するフラグが見つかりません。スキップします。"
                    )

                pbar.update(1)

    def _get_valid_files(self):
        """
        入力ディレクトリから有効なファイルのリストを取得します。

        Parameters:
        ------
        なし

        Returns:
        ------
            valid_files : list
                日時でソートされた有効なファイル名のリスト
        """
        fft_files = os.listdir(self._fft_path)
        valid_files = []
        for file in fft_files:
            try:
                self._parse_datetime(file)
                valid_files.append(file)
            except ValueError as e:
                self._warnings.append(f"{file} をスキップします: {str(e)}")
        return sorted(valid_files, key=self._parse_datetime)

    def _parse_datetime(self, filename: str) -> datetime:
        """
        ファイル名から日時情報を抽出します。

        Parameters:
        ------
            filename : str
                解析対象のファイル名

        Returns:
        ------
            datetime : datetime
                抽出された日時情報

        Raises:
        ------
            ValueError
                ファイル名から日時情報を抽出できない場合
        """
        for pattern in self._filename_patterns:
            match = re.match(pattern, filename)
            if match:
                year, month, day, time = match.groups()
                datetime_str: str = f"{year}{month}{day}{time}"
                return datetime.strptime(datetime_str, "%Y%m%d%H%M")

        raise ValueError(f"Could not parse datetime from filename: {filename}")

    def _prepare_directories(self):
        """
        出力ディレクトリを準備します。
        既存のディレクトリがある場合は削除し、新しく作成します。
        """
        for path in [self._sorted_path, self._good_data_path, self._bad_data_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

        if self._sort_by_rh:
            for i in range(10, 101, 10):
                rh_path = os.path.join(self._sorted_path, f"RH{i}")
                os.makedirs(rh_path, exist_ok=True)

    def _read_flag_file(self):
        """
        フラグファイルを読み込み、self._flagsディクショナリに格納します。
        """
        with open(self._flag_file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                time = datetime.strptime(row["time"], "%Y/%m/%d %H:%M")
                try:
                    rh = float(row["RH"])
                except ValueError:  # RHが#N/Aなどの数値に変換できない値の場合
                    self.logger.debug(f"Invalid RH value at {time}: {row['RH']}")
                    rh = -1  # 不正な値として扱うため、負の値を設定

                self._flags[time] = {"Flg": int(row["Flg"]), "RH": rh}

    @staticmethod
    def get_rh_directory(rh: float):
        """
        すべての値を10刻みで切り上げる（例: 80.1 → RH90, 86.0 → RH90, 91.2 → RH100）
        """
        if rh < 0 or rh > 100:  # 相対湿度として不正な値を除外
            return "bad_data"
        elif rh == 0:  # 0の場合はRH0に入れる
            return "RH0"
        else:  # 10刻みで切り上げ
            return f"RH{min(int((rh + 9.99) // 10 * 10), 100)}"

    @staticmethod
    def setup_logger(logger: Logger | None, log_level: int = INFO) -> Logger:
        """
        ロガーを設定します。

        ロギングの設定を行い、ログメッセージのフォーマットを指定します。
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
