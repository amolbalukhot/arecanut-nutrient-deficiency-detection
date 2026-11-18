"""ReportLab PDF reports generated from stored prediction records."""
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import config

GREEN = colors.HexColor("#14532d")


def _img(path: Path, max_w=7 * cm, max_h=7 * cm):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((900, 900), Image.Resampling.LANCZOS)
            w, h = im.size
            data = BytesIO()
            im.save(data, format="JPEG", quality=82, optimize=True)
            data.seek(0)
        s = min(max_w / w, max_h / h)
        return RLImage(data, width=w * s, height=h * s)
    except Exception:  # noqa: BLE001
        return Paragraph("<i>Image unavailable</i>", getSampleStyleSheet()["Normal"])


def _footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")


def generate_report(rec: dict, out_dir: Path | None = None, upload_dir=None, heatmap_dir=None) -> Path:
    out_dir = Path(out_dir or config.REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{rec['id']}.pdf"
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=GREEN)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=GREEN)
    body = ss["BodyText"]
    s = [Paragraph("Arecanut Leaf Verification &amp; Nutrient Deficiency Detection", h1),
         Paragraph(f"Prediction ID: {rec['id']} &nbsp;|&nbsp; Date: {rec['created_at']} (UTC)", body), Spacer(1, 10),
         Paragraph("Uploaded image", h2),
         _img(Path(upload_dir or config.UPLOAD_DIR) / Path(rec["image_path"]).name)]
    accepted = bool(rec["verification_accepted"])
    vc = rec.get("verification_confidence")
    s += [Paragraph("Stage 1 - Leaf verification", h2),
          Paragraph(f"Result: {rec.get('verification_class') or 'n/a'} "
                    f"({'accepted' if accepted else 'rejected'}); model score "
                    f"{vc * 100:.1f}%" if vc is not None else "Result: n/a", body)]
    if accepted and rec.get("nutrient_class"):
        s += [Paragraph("Stage 2 - Nutrient status (MobileNetV2)", h2),
              Paragraph(f"Predicted class: <b>{rec['nutrient_class'].replace('_', ' ')}</b> "
                        f"(model score {rec['nutrient_confidence'] * 100:.1f}%, not a calibrated probability)", body)]
        if rec.get("heatmap_path"):
            s += [Paragraph("Grad-CAM", h2), _img(Path(heatmap_dir or config.HEATMAP_DIR) / Path(rec["heatmap_path"]).name)]
        s += [Paragraph("Recommendation", h2), Paragraph(rec.get("recommendation") or "", body)]
    else:
        s += [Paragraph("Nutrient classification was not performed because the image failed leaf verification.", body)]
    s += [Spacer(1, 14), Paragraph("Limitations", h2),
          Paragraph("This is an automated screening estimate, not a confirmed agricultural diagnosis. Grad-CAM highlights "
                    "image regions that influenced the model and is not independent evidence of deficiency. "
                    "Confirm with soil/leaf testing and professional advice.", body)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm).build(s, onFirstPage=_footer, onLaterPages=_footer)
    return path
