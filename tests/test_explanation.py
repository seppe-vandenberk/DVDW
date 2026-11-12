from pathlib import Path
import re
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient

from connect4.az import PolicyValue
from connect4.simulation_service import create_app


def test_explanation_matches_model_and_contains_accessible_svg_schemes():
    page = (Path(__file__).parents[1] / "connect4/static/explain.html").read_text(encoding="utf-8")
    count = sum(p.numel() for p in PolicyValue().parameters())
    assert f"{count:,}" in page
    diagrams = re.findall(r"<svg\b.*?</svg>", page, re.DOTALL)
    assert len(diagrams) == 3
    for diagram in diagrams:
        root = ET.fromstring(diagram)
        assert root.attrib["role"] == "img"
        assert root.find("title") is not None
        ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
        assert all(ref in ids for ref in root.attrib["aria-labelledby"].split())
        for element in root.iter():
            marker = element.attrib.get("marker-end")
            if marker:
                assert marker[5:-1] in ids


def test_explanation_is_served_locally_without_external_assets():
    with TestClient(create_app(run_worker=False)) as client:
        response = client.get("/explain")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "reinforcement learning" in response.text
        assert client.get("/static/explain-simple.js").status_code == 200
        technical = client.get("/static/explain-technical.html")
        assert technical.status_code == 200 and "policy-value-7x5-v1" in technical.text
        css = client.get("/static/explain.css")
        assert css.status_code == 200
        assert "https://" not in response.text and "@import" not in css.text
