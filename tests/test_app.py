import io
import os

os.environ["DISABLE_BERT"] = "1"

import app as app_module


def _multipart_payload(csv_text, model="vader", filename="input.csv"):
    return {
        "model": model,
        "file": (io.BytesIO(csv_text.encode("utf-8")), filename),
    }


def test_predict_endpoint_returns_confidence():
    client = app_module.app.test_client()
    response = client.post("/predict", json={"text": "I love this product", "model": "vader"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sentiment"] == "positive"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["label"] == payload["sentiment"]
    assert payload["score"] == payload["confidence"]


def test_predict_empty_text_is_neutral():
    client = app_module.app.test_client()
    response = client.post("/predict", json={"text": "", "model": "vader"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sentiment"] == "neutral"
    assert payload["confidence"] == 0.0
    assert payload["label"] == "neutral"
    assert payload["score"] == 0.0


def test_batch_predict_includes_label_and_score():
    client = app_module.app.test_client()
    csv_text = "text\nI love this\nI hate this\n"
    response = client.post(
        "/batch_predict",
        data=_multipart_payload(csv_text),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert set(payload[0].keys()) >= {"text", "sentiment", "confidence", "label", "score"}
    assert 0.0 <= float(payload[0]["confidence"]) <= 1.0
    assert payload[0]["label"] == payload[0]["sentiment"]
    assert float(payload[0]["score"]) == float(payload[0]["confidence"])


def test_batch_predict_rejects_missing_text_column():
    client = app_module.app.test_client()
    csv_text = "message\nHello\n"
    response = client.post(
        "/batch_predict",
        data=_multipart_payload(csv_text),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "text" in payload["error"].lower()


def test_batch_predict_rejects_malformed_csv():
    client = app_module.app.test_client()
    malformed_csv = 'text\n"unclosed quote\n'
    response = client.post(
        "/batch_predict",
        data=_multipart_payload(malformed_csv),
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload


def test_batch_predict_rejects_missing_file():
    client = app_module.app.test_client()
    response = client.post(
        "/batch_predict",
        data={"model": "vader"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert "no csv file uploaded" in payload["error"].lower()


def test_batch_predict_empty_text_row_is_neutral():
    client = app_module.app.test_client()
    csv_text = 'text\n""\nI love this\n'
    response = client.post(
        "/batch_predict",
        data=_multipart_payload(csv_text),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload) == 2
    assert payload[0]["sentiment"] == "neutral"
    assert float(payload[0]["confidence"]) == 0.0
    assert payload[0]["label"] == "neutral"
    assert float(payload[0]["score"]) == 0.0


def test_batch_predict_download_returns_csv_attachment():
    client = app_module.app.test_client()
    csv_text = "text\nThis is nice\nThis is awful\n"
    response = client.post(
        "/batch_predict_download",
        data=_multipart_payload(csv_text),
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment; filename=" in response.headers.get("Content-Disposition", "")
    body = response.get_data(as_text=True)
    assert "text,label,score,sentiment,confidence" in body
