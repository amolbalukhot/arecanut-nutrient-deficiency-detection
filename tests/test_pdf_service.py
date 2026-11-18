from pathlib import Path
from PIL import Image
from services import pdf_service


def test_pdf_with_and_without_heatmap(tmp_path):
    up, hm = tmp_path / "up", tmp_path / "hm"; up.mkdir(); hm.mkdir()
    Image.new("RGB", (50, 40), "green").save(up / "a.png")
    rec = dict(id=1, created_at="2026-01-01 10:00:00", image_path="a.png", verification_class="Arecanut",
               verification_confidence=0.9, verification_accepted=1, nutrient_class="Healthy",
               nutrient_confidence=0.8, heatmap_path="missing.png", recommendation="ok")
    p = pdf_service.generate_report(rec, tmp_path / "r", up, hm)  # missing heatmap must not crash
    assert p.read_bytes().startswith(b"%PDF")
    rec.update(id=2, verification_accepted=0, nutrient_class=None, nutrient_confidence=None, heatmap_path=None)
    assert pdf_service.generate_report(rec, tmp_path / "r", up, hm).is_file()
