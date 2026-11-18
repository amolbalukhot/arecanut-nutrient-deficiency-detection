import pytest
import config
from services import recommendation_service as rs


def test_all_classes_have_recommendations():
    assert set(rs.RECOMMENDATIONS) == set(config.CLASS_NAMES)
    for c in config.CLASS_NAMES:
        assert "soil" in rs.get_recommendation(c)["note"].lower()


def test_unknown_class():
    with pytest.raises(KeyError):
        rs.get_recommendation("Iron_Deficiency")


def test_class_order_is_fixed():
    assert config.CLASS_NAMES == ["Healthy", "Nitrogen_Deficiency", "Potassium_Deficiency", "Magnesium_Deficiency"]
