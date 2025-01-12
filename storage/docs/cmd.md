# コマンド集

よく使うコマンドをまとめたドキュメントです。

## 仮想環境

有効化

```bash
. .venv/bin/activate
```

無効化

```bash
deactivate
```

## ファイル操作

特定の文字列をファイル名に含むファイルを再帰的に削除するコマンド。以下は":Zone.Identifier"をファイル名に含むファイルを削除するコマンド。

```bash
find . -type f -name '*:Zone.Identifier' -exec rm {} +
```
