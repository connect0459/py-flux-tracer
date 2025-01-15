import os
from dotenv import load_dotenv
from py_flux_tracer import (
    FluxFootprintAnalyzer,
)


# 変数定義
# center_lan: float = 34.573904320329724  # SACの緯度
# center_lon: float = 135.4829511120712  # SACの経度
center_lan: float = 35.6644926  # YYGの緯度
center_lon: float = 139.6842876  # YYGの経度
dotenv_path = "/home/connect0459/labo/py-flux-tracer/workspace/.env"  # .envファイル

# 画像の設定
# site_name:str="SAC"
site_name: str = "YYG"
zoom: float = 13
local_image_dir: str = "/home/connect0459/labo/py-flux-tracer/storage/assets"


if __name__ == "__main__":
    # 環境変数の読み込み
    load_dotenv(dotenv_path)

    # APIキーの取得
    gms_api_key: str | None = os.getenv("GOOGLE_MAPS_STATIC_API_KEY")
    if not gms_api_key:
        raise ValueError("GOOGLE_MAPS_STATIC_API_KEY is not set in .env file")

    # インスタンスを作成
    ffa = FluxFootprintAnalyzer(z_m=111, logging_debug=False)

    # 航空写真の取得
    local_image_path: str = os.path.join(
        local_image_dir, f"{site_name}-zoom_{zoom}.png"
    )
    image = ffa.get_satellite_image_from_api(
        api_key=gms_api_key,
        center_lat=center_lan,
        center_lon=center_lon,
        output_path=local_image_path,
        zoom=zoom,
    )  # API
