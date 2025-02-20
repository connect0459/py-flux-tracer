# プロジェクトの初期設定

このドキュメントでは、パッケージの既存のクラスを修正したり、新たなクラスを追加して新バージョンのパッケージを公開するために必要な初期設定について記述しています。

## 前提

- pip または uv がローカルPCにインストールされている
- GitHubアカウントを所有しており、SSH接続のセットアップが完了している

## ローカル環境で動作させる

### リポジトリをクローン

まず、GitHubからリポジトリをクローンします。

```bash
git clone git@github.com:connect0459/py-flux-tracer.git
```

クローンできたら、カレントディレクトリを切り替えます。

```bash
cd py-flux-tracer
```

### 依存関係のインストール

プロジェクトルートに設置された `pyproject.toml` を読み込んで使用するパッケージをダウンロードします。

#### pip の場合

```bash
pip install -e ".[dev]"
```

#### uv の場合

```bash
uv sync
```
