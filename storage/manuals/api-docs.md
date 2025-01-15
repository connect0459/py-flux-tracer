# APIドキュメント

python環境またはuvを用いたコマンドを紹介しています。ここでは [pdoc](https://github.com/pdoc3/pdoc) でドキュメント出力を行います。

## pdocのインストール

```bash
pip install pdoc
```

または

```bash
uv pip install pdoc
```

## 生成

```bash
pdoc -o storage/api-docs/v0 py_flux_tracer
```

または

```bash
uv run pdoc -o storage/api-docs/v0 py_flux_tracer
```

## ブラウザで表示

```bash
pdoc -h localhost -p 8080 -t storage/api-docs/v0 py_flux_tracer
```

または

```bash
uv run pdoc -h localhost -p 8080 -t storage/api-docs/v0 py_flux_tracer
```

<http://localhost:8080> にアクセスするとドキュメントが表示されます。
