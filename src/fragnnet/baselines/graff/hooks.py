"""GRAFF's hooks, registered for ``model_type: graff`` (see ``fragnnet.hooks``)."""

from __future__ import annotations

from typing import Any

from fragnnet.hooks import ModelHooks


class GrAFFHooks(ModelHooks):
    """GRAFF's inference-config overlay for its annotation dataset (``ann_params``)."""

    def apply_inference_config(
        self, config_d: dict[str, Any], *, disable_preproc: bool, output_formula_str: bool
    ) -> None:
        if "ann_params" in config_d:
            if disable_preproc:
                config_d["ann_params"]["preprocess"] = False
            config_d["ann_params"]["formula_str"] = output_formula_str
