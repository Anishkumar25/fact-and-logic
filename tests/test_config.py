from src.config import device


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    assert device() == "cpu"
