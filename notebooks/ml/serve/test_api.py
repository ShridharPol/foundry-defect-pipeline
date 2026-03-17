"""
Unit tests for Foundry Defect Detection API.
Tests health, root, and prediction endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from main import app

client = TestClient(app)


# --- Helpers ---
def make_test_image(color: tuple = (128, 128, 128), size: tuple = (224, 224)) -> bytes:
    """Generate a synthetic RGB image as bytes for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- Health & Root ---
class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy(self):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestRootEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_model_info(self):
        response = client.get("/")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "MobileNetV2"
        assert "def_front" in data["classes"]
        assert "ok_front" in data["classes"]
        assert data["developed_by"] == "Hamdan InfoCom, Belagavi"

    def test_root_contains_endpoints(self):
        response = client.get("/")
        data = response.json()
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0


# --- Predict ---
class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_image(self):
        img_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_response_structure(self):
        img_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_returns_valid_class(self):
        img_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        data = response.json()
        assert data["prediction"] in ["def_front", "ok_front"]

    def test_predict_confidence_between_0_and_1(self):
        img_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_1(self):
        img_bytes = make_test_image()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        data = response.json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_rejects_non_image(self):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_predict_different_image_sizes(self):
        """Model should handle any image size — resized internally to 224x224."""
        for size in [(64, 64), (512, 512), (300, 400)]:
            img_bytes = make_test_image(size=size)
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )
            assert response.status_code == 200

    def test_predict_png_image(self):
        """API should accept PNG as well as JPEG."""
        img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        response = client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")}
        )
        assert response.status_code == 200