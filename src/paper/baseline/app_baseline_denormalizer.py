import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Union

import pandas as pd

from ml.app_denormalizer import prepare_tx_structure, Component
from paper.baseline.app_baseline_utils import get_component_fields, load_global_metadata


def extract_component_data_from_dataframe(
    raw_generated_rows: pd.DataFrame
) -> Dict[str, Dict[str, Union[int, str, Dict[str, Union[str, int]]]]]:
    """
    Extract the traces from the dataframe and manipulate back the metadata keys.
    """
    raw_components: Dict[str, Any] = {}

    for df_component in raw_generated_rows.iloc:
        component = {
            "componentName": df_component["componentName"],
            "parentComponentName": df_component["parentComponentName"],
            "startTime": df_component["startTime"],
            "endTime": df_component["endTime"],
            "hasError": df_component["hasError"],
            "metadata": {
                key: value for key, value in zip(
                    get_component_fields(df_component["componentName"]),
                    [
                        df_component["str_feature_1"],
                        df_component["str_feature_2"],
                        df_component["int_feature_1"],
                        df_component["int_feature_2"],
                        df_component["int_feature_3"],
                    ]
                )},
        }
        raw_components[component["componentName"]] = component
    return raw_components


def prepare_components(
    raw_generated_rows: pd.DataFrame
) -> List[Component]:
    raw_components = extract_component_data_from_dataframe(raw_generated_rows)

    parent_to_children = defaultdict(list)
    for raw_component in raw_components.values():
        parent_to_children[raw_component["parentComponentName"]].append(raw_component)
    for parent in parent_to_children:
        if parent not in raw_components:
            raw_components[parent] = {
                "componentName": parent,
                "parentComponentName": "top",
                "startTime": 0,
                "endTime": 0,
                "hasError": False,
                "metadata": {},
            }

    components: List[Component] = []
    for raw_component in raw_components.values():
        components.append(
            Component(
                component_id=str(raw_component["componentName"]),
                start_time=raw_component["startTime"],  # type: ignore
                end_time=raw_component["endTime"],  # type: ignore
                duration=raw_component["endTime"] - raw_component["startTime"],  # type: ignore
                children_ids=[
                    str(component["componentName"])
                    for component in parent_to_children[raw_component["componentName"]]
                    if component["componentName"] in raw_components
                ],
                group=str(raw_component["componentName"]),
                has_error=bool(raw_component["hasError"]),
                metadata=raw_component["metadata"],
            )
        )
    return list({c.component_id: c for c in components}.values())


def denormalize_data_baseline(
    input_dir: str, generated_data_dir: str
) -> None:
    if not os.listdir(input_dir):
        raise Exception("No data to denormalize")
    for filename in os.listdir(input_dir):
        df = pd.read_csv(os.path.join(input_dir, filename), error_bad_lines=False)

        os.makedirs(generated_data_dir, exist_ok=True)
        with open(
            os.path.join(generated_data_dir, filename), "w"
        ) as output_file:
            for txid, raw_generated_rows in df.groupby("traceId"):
                components = prepare_components(raw_generated_rows)
                tx = prepare_tx_structure(txid, components)
                if tx:
                    output_file.write(json.dumps(tx) + ",\n")
