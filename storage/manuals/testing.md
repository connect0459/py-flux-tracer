# テスト

uv環境でテストを作成して実行する方法について記述しています。

## 必要なパッケージのインストール

テストを実行するために必要なpytestをインストールします。

```bash
uv add --dev pytest
```

## テストファイルを作成

テストファイルは`tests`ディレクトリ配下に作成します。テストファイル名は`test_`で始めるか、`_test`で終わる必要があります。

今回は`test_`で始める形式に統一します。以下の記述を`pyproject.toml`に追加してください。

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

プロジェクト構造の例:

```txt
project/
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
└── tests/
    ├── __init__.py
    └── test_module.py
```

テストファイルの例:

```python
# tests/test_module.py
import pytest
from src.my_package.module import my_function

def test_my_function():
    assert my_function(2, 3) == 5
```

## テストファイルを実行

特定のファイル(例: `tests/commons/test_dataclasses.py`)を実行:

```bash
uv run pytest tests/commons/test_dataclasses.py -v
```

特定のディレクトリ(例: `tests/commons`)を実行:

```bash
uv run pytest tests/commons/ -v
```

`tests`ディレクトリ全体を実行:

```bash
uv run pytest tests/ -v
```

## オプション設定

`pyproject.toml`にテストの設定を追加することで、テストの実行をカスタマイズできます:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
addopts = "-ra -q"
pythonpath = ["src"]
```

## よく使用するオプション

- `-v`: 詳細な実行結果を表示
- `-s`: print文の出力を表示
- `-k "test_name"`: 特定のテスト名にマッチするテストのみ実行
- `--pdb`: エラー発生時にデバッガを起動
- `-x`: 最初のエラーで実行を停止
