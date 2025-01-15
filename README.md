# py-flux-tracer

このパッケージは、大気観測で得られたデータファイルを解析するPythonパッケージです。渦相関法を主とする大気観測データを対象としています。

## インストール

### 必要条件

- Python >= 3.11
- pip または uv

### インストール方法

```bash
pip install py-flux-tracer
```

または

```bash
uv add py-flux-tracer
```

## ライセンス

本ソフトウェアは、 [LICENSE](https://github.com/connect0459/py_flux_tracer/blob/main/LICENSE) に基づいて使用してください。

## ドキュメント

開発者に向けてドキュメントを作成しています。`storage/manuals`配下に格納しています。

- [リファレンス](./storage/manuals/references.md)
- パッケージの開発
  - [1. プロジェクトの初期設定](./storage/manuals/development/1-init-project.md)
  - [2. Gitを用いた開発の概要](./storage/manuals/development/2-overview-git.md)
  - [3. 新機能の追加](./storage/manuals/development/3-add-features.md)
- [パッケージのデプロイ](./storage/manuals/deployment.md)
- [コマンド集](./storage/manuals/cmd.md)

クラスやメソッドのdocstringはNumPy/SciPy Styleで記述しています。 [pdoc](https://github.com/pdoc3/pdoc) や [Sphinx](https://github.com/sphinx-doc/sphinx) などのNumPy/SciPy Styleをサポートするドキュメント生成ツールでHTMLドキュメントを生成可能です。

## コントリビュータ

- [connect0459](https://github.com/connect0459)
