"""Packaging-oriented tests."""

import yaml
from importlib.resources import files


def test_default_config_packaged_resource_exists():
    cfg_path = files("asl_qc").joinpath("default_config.yaml")
    assert cfg_path.is_file()
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    assert "pipeline" in data
