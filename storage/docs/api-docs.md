# APIドキュメント

python環境またはuvを用いたコマンドを紹介しています。ここでは [pdoc](https://github.com/pdoc3/pdoc) でドキュメント出力を行います。

## 生成

```bash
pdoc -o storage/api-docs py_flux_tracer
```

または

```bash
uv run pdoc -o storage/api-docs py_flux_tracer
```

## ブラウザで表示

```bash
pdoc py_flux_tracer -h localhost -p 8080
```

または

```bash
uv run pdoc py_flux_tracer -h localhost -p 8080
```

<http://localhost:8080> にアクセスするとドキュメントが表示される。
