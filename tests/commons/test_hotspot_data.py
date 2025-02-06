from py_flux_tracer import HotspotData


def test_dataclasses_initialization():
    """HotspotDataが正しく初期化されることを確認するテスト"""
    hotspot = HotspotData(
        angle=45.0,
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        delta_ch4=0.1,
        delta_c2h6=0.5,
        delta_ratio=0.05,
        section=1,
        timestamp="satellite",
        type="gas",
    )

    assert hotspot.angle == 45.0
    assert hotspot.avg_lat == 35.6895
    assert hotspot.avg_lon == 139.6917
    assert hotspot.correlation == 0.85
    assert hotspot.delta_ratio == 0.05
    assert hotspot.section == 1
    assert hotspot.timestamp == "satellite"
    assert hotspot.type == "gas"
