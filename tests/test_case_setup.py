"""Tests for rerun_viz.case_setup path helpers."""

from __future__ import annotations

from rerun_viz.case_setup import config_yaml_for_case, optimal_params_path


def test_config_yaml_cloth_and_package():
    assert config_yaml_for_case("foo_cloth_bar") == "configs/cloth.yaml"
    assert config_yaml_for_case("package_1") == "configs/cloth.yaml"


def test_config_yaml_real():
    assert config_yaml_for_case("real_scene") == "configs/real.yaml"


def test_optimal_params_path():
    assert optimal_params_path("my_case") == "experiments_optimization/my_case/optimal_params.pkl"
