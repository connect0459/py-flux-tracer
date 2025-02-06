import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv

from py_flux_tracer import (
    FluxFootprintAnalyzer,
)

# 変数定義
dotenv_path = "/home/connect0459/labo/py-flux-tracer/workspace/.env"  # .envファイル

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


if __name__ == "__main__":
    # サイト情報の取得
    site_info = next(
        (site for site in sites_configs if site["name"] == target_site_name), None
    )
    if site_info is None:
        raise ValueError(f"Site '{target_site_name}' not found in sites_config")

    # 環境変数の読み込み
    load_dotenv(dotenv_path)

    # APIキーの取得
    gms_api_key: str | None = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not gms_api_key:
        raise ValueError("GOOGLE_MAPS_STATIC_API_KEY is not set in .env file")

    current_year = datetime.now().year  # 現在の年を取得

    # インスタンスを作成
    ffa = FluxFootprintAnalyzer(z_m=111, logging_debug=False)

    # 航空写真の取得
    local_image_path: str = os.path.join(
        local_image_dir, f"{target_site_name}_{current_year}-zoom_{zoom}.png"
    )
    image = ffa.get_satellite_image_from_api(
        api_key=gms_api_key,
        center_lat=float(site_info["center_lat"]),  # float型にキャスト
        center_lon=float(site_info["center_lon"]),  # float型にキャスト
        output_filepath=local_image_path,
        zoom=zoom,
    )  # API
