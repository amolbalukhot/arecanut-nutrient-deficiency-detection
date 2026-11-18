"""Class-consistent, non-dosage recommendations. Screening guidance only."""
from config import CLASS_NAMES

GENERAL_NOTE = ("Similar visual symptoms can have several causes (other nutrients, water stress, disease, pests). "
                "Confirm with a soil or leaf-tissue test and consult a local agricultural officer before applying inputs.")

RECOMMENDATIONS = {
    "Healthy": {
        "summary": "No nutrient deficiency was predicted for this leaf.",
        "symptoms": "Leaf colour and structure appear normal to the model.",
        "advice": "Continue regular crop monitoring and balanced nutrient management based on periodic soil tests, "
                  "with adequate organic matter and irrigation."},
    "Nitrogen_Deficiency": {
        "summary": "The pattern is consistent with possible nitrogen deficiency.",
        "symptoms": "Typical signs include general yellowing (chlorosis), often starting on older leaves, and reduced vigour.",
        "advice": "Review nitrogen management against soil-test results, using a split-application approach and organic "
                  "sources where suitable. Avoid applying nitrogen without confirming the need."},
    "Potassium_Deficiency": {
        "summary": "The pattern is consistent with possible potassium deficiency.",
        "symptoms": "Typical signs include yellow-orange discolouration, marginal scorching or necrosis, mostly on older leaves.",
        "advice": "Assess soil and leaf potassium levels and adjust potassium supply to crop requirements. "
                  "Balance potassium with other nutrients, as excess can limit magnesium uptake."},
    "Magnesium_Deficiency": {
        "summary": "The pattern is consistent with possible magnesium deficiency.",
        "symptoms": "Typical signs include yellowing between veins or along leaf margins of older leaves.",
        "advice": "Confirm magnesium status through soil or leaf testing, consider soil pH and potassium-calcium balance, "
                  "and use a magnesium source only if the assessment supports it."},
}


def get_recommendation(class_name: str) -> dict:
    if class_name not in RECOMMENDATIONS:
        raise KeyError(f"Unknown class '{class_name}'. Expected one of {CLASS_NAMES}.")
    r = RECOMMENDATIONS[class_name]
    return {**r, "note": GENERAL_NOTE}


def recommendation_text(class_name: str) -> str:
    r = get_recommendation(class_name)
    return f"{r['summary']} {r['symptoms']} {r['advice']} {r['note']}"
