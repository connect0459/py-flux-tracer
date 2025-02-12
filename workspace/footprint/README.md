# footprint

このディレクトリはMonthlyシートに記録された対象気体のフラックスについて、フットプリントを計算して衛星画像上にマッピングする解析を実行するプロジェクトです。

## `fetch_satellite_image.py`

衛星画像を取得するためのスクリプト。設定は次の手順で行う。

### 1. `.env`にGoogle Maps Static APIのAPIキーを追加

Google Maps Platformで取得したAPIキーを追加してください。以下は入力例です。

```env
# Google Maps Platform で取得できるAPIキー
# footprint の衛星画像を取得する際に使用
GOOGLE_MAPS_STATIC_API_KEY=AIzaSyBr_xxx
```

### 2. `fetch_satellite_image.py`

以下のようにして、configに必要な情報を入力します。画像取得に必要な情報は以下の通りです。

- `dotenv_path` : `.env`のファイルパス。
- `sites_configs` : 観測サイトのタグ(`name`)、緯度経度(`center_lat`、`center_lon`)。
- `target_site_name` : 今回の実行で取得するサイト名(`sites_configs`のいずれかの`name`に一致する必要がある)。
- `zoom` : ズームレベル。
- `local_image_dir` : 取得した画像を保存するディレクトリ。

```py
""" ------ config start ------ """

# 変数定義
dotenv_path: str = (
    "/home/connect0459/labo/py-flux-tracer/workspace/footprint/.env"  # .envファイル
)

SiteConfigKeys = Literal["name", "center_lat", "center_lon"]

sites_configs: list[dict[SiteConfigKeys, str | float]] = [
    {
        "name": "SAC",
        "center_lat": 34.573904320329724,
        "center_lon": 135.4829511120712,
    },
    {
        "name": "YYG",
        "center_lat": 35.6644926,
        "center_lon": 139.6842876,
    },
]

# 画像の設定
target_site_name: str = "SAC"
# target_site_name: str = "YYG"
zoom: float = 13
local_image_dir: str = "/home/connect0459/labo/py-flux-tracer/storage/assets"

""" ------ config end ------ """
```

情報が入力できたら、コードを実行する。取得された画像は、`local_image_dir`に保存される。

## `plot_footprint.py`
