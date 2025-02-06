import csv
import os
import re
import shutil
from datetime import datetime
from logging import DEBUG, INFO, Logger
from typing import ClassVar

from tqdm import tqdm

from ..commons.utilities import setup_logger


class FftFileReorganizer:
    """
    FFTファイルを再編成するためのクラス。

    入力ディレクトリからファイルを読み取り、フラグファイルに基づいて
    出力ディレクトリに再編成します。時間の完全一致を要求し、
    一致しないファイルはスキップして警告を出します。
    オプションで相対湿度(RH)に基づいたサブディレクトリへの分類も可能です。
    """

    # クラス定数の定義
    DEFAULT_FILENAME_PATTERNS: ClassVar[list[str]] = [
        r"FFT_TOA5_\d+\.SAC_Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?\.csv",
        r"FFT_TOA5_\d+\.SAC_Ultra\.Eddy_\d+_(\d{4})_(\d{2})_(\d{2})_(\d{4})(?:\+)?(?:-resampled)?\.csv",
    ]  # デフォルトのファイル名のパターン(正規表現)
    DEFAULT_OUTPUT_DIRS: ClassVar[dict[str, str]] = {
        "GOOD_DATA": "good_data_all",
        "BAD_DATA": "bad_data",
    }  # 出力ディレクトリの構造に関する定数

    def __init__(
        self,
        input_dirpath: str,
        output_dirpath: str,
        flag_csv_path: str,
        filename_patterns: list[str] | None = None,
        output_dirpaths_struct: dict[str, str] | None = None,
        sort_by_rh: bool = True,
        logger: Logger | None = None,
        logging_debug: bool = False,
    ):
        """
        FftFileReorganizerクラスを初期化します。

        Parameters
        ----------
            input_dirpath: str
                入力ファイルが格納されているディレクトリのパス
            output_dirpath: str
                出力ファイルを格納するディレクトリのパス
            flag_csv_path: str
                フラグ情報が記載されているCSVファイルのパス
            filename_patterns: list[str] | None, optional
                ファイル名のパターン(正規表現)のリスト。指定しない場合はデフォルトのパターンが使用されます。
            output_dirpaths_struct: dict[str, str] | None, optional
                出力ディレクトリの構造を定義する辞書。指定しない場合はデフォルトの構造が使用されます。
            sort_by_rh: bool, optional
                RHに基づいてサブディレクトリにファイルを分類するかどうか。デフォルトはTrueです。
            logger: Logger | None, optional
                使用するロガー。指定しない場合は新規のロガーが作成されます。
            logging_debug: bool, optional
                ログレベルをDEBUGに設定するかどうか。デフォルトはFalseです。

        Examples
        -------
        >>> reorganizer = FftFileReorganizer(
        ...     input_dirpath="./raw_data",
        ...     output_dirpath="./processed_data",
        ...     flag_csv_path="./flags.csv"
        ... )
        >>> reorganizer.reorganize()  # ファイルの再編成を実行
        """
        self._fft_path: str = input_dirpath
        self._sorted_path: str = output_dirpath
        self._output_dirpaths_struct = (
            output_dirpaths_struct or self.DEFAULT_OUTPUT_DIRS
        )
        self._good_data_path: str = os.path.join(
            output_dirpath, self._output_dirpaths_struct["GOOD_DATA"]
        )
        self._bad_data_path: str = os.path.join(
            output_dirpath, self._output_dirpaths_struct["BAD_DATA"]
        )
        self._filename_patterns: list[str] = (
            self.DEFAULT_FILENAME_PATTERNS.copy()
            if filename_patterns is None
            else filename_patterns
        )
        self._flag_filepath: str = flag_csv_path
        self._sort_by_rh: bool = sort_by_rh
        self._flags = {}
        self._warnings = []
        # ロガー
        log_level: int = INFO
        if logging_debug:
            log_level = DEBUG
        self.logger: Logger = setup_logger(logger=logger, log_level=log_level)

    def reorganize(self):
        """
        ファイルの再編成プロセス全体を実行します。
        
        ディレクトリの準備、フラグファイルの読み込み、有効なファイルの取得、ファイルのコピーを順に行います。
        処理後、警告メッセージがあれば出力します。

        Returns
        ----------
            None
                戻り値はありません。

        Examples
        -------
        >>> reorganizer = FftFileReorganizer(
        ...     input_dirpath="./raw_data",
        ...     output_dirpath="./processed_data", 
        ...     flag_csv_path="./flags.csv"
        ... )
        >>> reorganizer.reorganize()  # ファイルの再編成を実行
        """
        self._prepare_directories()
        self._read_flag_file()
        valid_files = self._get_valid_files()
        self._copy_files(valid_files)

        if self._warnings:
            self.logger.warning("Warnings:")
            for warning in self._warnings:
                self.logger.warning(warning)

    def _copy_files(self, valid_files):
        """
        有効なファイルを適切な出力ディレクトリにコピーします。
        フラグファイルの時間と完全に一致するファイルのみを処理します。

        Parameters
        ----------
            valid_files: list
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

        Parameters
        ----------
        なし

        Returns
        ----------
            valid_files: list
                日時でソートされた有効なファイル名のリスト
        """
        fft_files = os.listdir(self._fft_path)
        valid_files = []
        for file in fft_files:
            try:
                self._parse_datetime(file)
                valid_files.append(file)
            except ValueError as e:
                self._warnings.append(f"{file} をスキップします: {e!s}")
        return sorted(valid_files, key=self._parse_datetime)

    def _parse_datetime(self, filename: str) -> datetime:
        """
        ファイル名から日時情報を抽出します。

        Parameters
        ----------
            filename: str
                解析対象のファイル名

        Returns
        ----------
            datetime: datetime
                抽出された日時情報

        Raises
        ----------
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
        with open(self._flag_filepath) as f:
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
        相対湿度の値に基づいて、保存先のディレクトリ名を決定します。
        値は10刻みで切り上げられます。

        Parameters
        ----------
            rh: float
                相対湿度の値(0-100の範囲)

        Returns
        ----------
            str
                ディレクトリ名(例: "RH90")。不正な値の場合は"bad_data"

        Examples
        ----------
        >>> FFTFilesReorganizer.get_rh_directory(80.1)
        'RH90'
        >>> FFTFilesReorganizer.get_rh_directory(86.0) 
        'RH90'
        >>> FFTFilesReorganizer.get_rh_directory(91.2)
        'RH100'
        >>> FFTFilesReorganizer.get_rh_directory(-1)
        'bad_data'
        """
        if rh < 0 or rh > 100:  # 相対湿度として不正な値を除外
            return "bad_data"
        elif rh == 0:  # 0の場合はRH0に入れる
            return "RH0"
        else:  # 10刻みで切り上げ
            return f"RH{min(int((rh + 9.99) // 10 * 10), 100)}"
