from py_flux_tracer import HotspotData


def test_dataclasses_initialization():
    """HotspotDataが正しく初期化されることを確認するテスト"""
    hotspot = HotspotData(
        angle=45.0,
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        ratio=0.05,
        section=1,
        source="satellite",
        type="gas",
    )

    assert hotspot.angle == 45.0
    assert hotspot.avg_lat == 35.6895
    assert hotspot.avg_lon == 139.6917
    assert hotspot.correlation == 0.85
    assert hotspot.ratio == 0.05
    assert hotspot.section == 1
    assert hotspot.source == "satellite"
    assert hotspot.type == "gas"


def test_dataclasses_equality():
    """同じ値を持つHotspotDataインスタンスが等しいと判定されることを確認するテスト"""
    hotspot1 = HotspotData(
        angle=45.0,
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        ratio=0.05,
        section=1,
        source="satellite",
        type="gas",
    )

    hotspot2 = HotspotData(
        angle=45.0,
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        ratio=0.05,
        section=1,
        source="satellite",
        type="gas",
    )

    assert hotspot1 == hotspot2


def test_dataclasses_inequality():
    """異なる値を持つHotspotDataインスタンスが等しくないと判定されることを確認するテスト"""
    hotspot1 = HotspotData(
        angle=45.0,
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        ratio=0.05,
        section=1,
        source="satellite",
        type="gas",
    )

    hotspot2 = HotspotData(
        angle=30.0,  # 異なる角度
        avg_lat=35.6895,
        avg_lon=139.6917,
        correlation=0.85,
        ratio=0.05,
        section=1,
        source="satellite",
        type="gas",
    )

    assert hotspot1 != hotspot2


def test_dataclasses_type_validation():
    """typeフィールドが有効な値のみを受け入れることを確認するテスト"""
    valid_types = ["bio", "gas", "comb"]

    for valid_type in valid_types:
        hotspot = HotspotData(
            angle=45.0,
            avg_lat=35.6895,
            avg_lon=139.6917,
            correlation=0.85,
            ratio=0.05,
            section=1,
            source="satellite",
            type=valid_type,
        )
        assert hotspot.type == valid_type
