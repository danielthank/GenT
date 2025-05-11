import pandas as pd
import pytest

from ml.app_denormalizer import Component, prepare_components
from ml.app_utils import GenTConfig

def test_prepare_components(tmp_path, config):
    raw_data = pd.DataFrame(
        [
            [1, 0, "top", 0, 1, False, 1],
            [1, 0, "a", 2, 2, True, 1],
        ],
        columns=[
            "traceId",
            "txStartTime",
            "chain",
            "gapFromParent_0",
            "duration_0",
            "hasError_0",
            "metadata_0_0",
        ],
    )
    components = prepare_components(raw_data, config)
    assert components == [
        Component(
            component_id="top",
            start_time=0,
            end_time=1,
            duration=1,
            children_ids=["top", "a"],
            group="top",
            has_error=False,
            metadata={},
        ),
        Component(
            component_id="a",
            start_time=2,
            end_time=4,
            duration=2,
            children_ids=[],
            group="a",
            has_error=True,
            metadata={},
        ),
    ]


def test_prepare_components_duplicate(config):
    raw_data = pd.DataFrame(
        [
            [2, 2, "a", 0, 1, True, 1],  # will be ignored
            [1, 0, "a", 1, 1, False, 1],
        ],
        columns=[
            "traceId",
            "txStartTime",
            "chain",
            "gapFromParent_0",
            "duration_0",
            "hasError_0",
            "metadata_0_0",
        ],
    )
    components = prepare_components(raw_data, config)
    assert components == [
        Component(
            component_id="a",
            start_time=1,
            end_time=2,
            duration=1,
            children_ids=[],
            group="a",
            has_error=False,
            metadata={},
        ),
    ]


def test_prepare_components_shared_root(tmp_path):
    raw_data = pd.DataFrame(
        [
            [0, 0, "a#b", 0, 1, True, 1, False, 1],
            [0, 0, "a#c", 1, 1, False, 1, False, 1],
        ],
        columns=[
            "traceId",
            "txStartTime",
            "chain",
            "gapFromParent_0",
            "duration_0",
            "hasError_0",
            "gapFromParent_1",
            "duration_1",
            "hasError_1",
        ],
    )

    config = GenTConfig(
        chain_length=2, metadata_int_size=0, metadata_str_size=0, is_test=True
    )

    components = prepare_components(raw_data, config)
    assert components == [
        Component(
            component_id="a",
            start_time=1,
            end_time=2,
            duration=1,
            children_ids=["b", "c"],
            group="a",
            has_error=False,
            metadata={},
        ),
        Component(
            component_id="b",
            start_time=2,
            end_time=2,
            duration=1,
            children_ids=[],
            group="b",
            has_error=True,
            metadata={},
        ),
        Component(
            component_id="c",
            start_time=2,
            end_time=2,
            duration=1,
            children_ids=[],
            group="c",
            has_error=True,
            
            metadata={},
        ),
    ]
