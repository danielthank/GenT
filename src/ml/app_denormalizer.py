import json
import os
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Union, Set

import numpy as np
import pandas as pd

from drivers.base_driver import BaseDriver
from ml.app_utils import (
    ComponentType,
    GenTConfig,
    get_component_type,
    get_key_name,
    get_metadata_size, get_key_value,
)


class Component(NamedTuple):
    component_id: str
    start_time: int
    end_time: int
    duration: int
    children_ids: List[str]
    group: str
    has_error: bool
    component_type: Optional[ComponentType]
    metadata: Dict[str, Union[str, int]]


"""
The normalized traces is assumed to have the columns:
- timeFromStart
- duration
- traceId
- tree
"""

ExtractedComponentData = Dict[str, Union[int, str, Dict[str, Union[str, int]]]]


def extract_component_data_from_dataframe(
    raw_generated_rows: pd.DataFrame, config: GenTConfig = GenTConfig()
) -> Dict[str, ExtractedComponentData]:
    """
    Extract the traces from the dataframe and manipulate back the metadata keys.
    """
    raw_components: Dict[str, Any] = {}

    for df_component in raw_generated_rows.iloc:
        components: List[str] = df_component["chain"].split("#")
        if len(components) > config.chain_length:
            raise ValueError("The chain length is not as expected.")
        for component_index, component_name in enumerate(components):
            if component_index == 0:
                if component_name in raw_components:
                    parent = raw_components[component_name]["parentComponentName"]
                else:
                    parent = "top"
            else:
                parent = components[component_index - 1]
            component_metadata: Dict[str, Union[str, int]] = {}
            for metadata_index in range(get_metadata_size(config)):
                key_name = get_key_name(component_name, metadata_index, config=config)
                if not key_name:
                    continue
                component_metadata[key_name] = get_key_value(
                    component_name, key_name, df_component[f"metadata_{component_index}_{metadata_index}"], config
                )
                if isinstance(component_metadata[key_name], np.int64):
                    component_metadata[key_name] = int(component_metadata[key_name])
            component = {  # Unwrap the pd.Series to a dict
                "componentName": component_name,
                "parentComponentName": parent,
                "txStartTime": df_component["txStartTime"],
                "gapFromParent": df_component[f"gapFromParent_{component_index}"],
                "duration": df_component[f"duration_{component_index}"],
                "hasError": df_component[f"hasError_{component_index}"],
                "metadata": component_metadata,
            }
            # Assumes that each component can appear only once in the transaction
            raw_components[component["componentName"]] = component
    return raw_components


def prepare_components(
    raw_generated_rows: Optional[pd.DataFrame],
    config: GenTConfig,
    extracted_component_data: Optional[Dict[str, ExtractedComponentData]] = None
) -> List[Component]:
    """
    This function put all the information on the component object.
    """

    def recursive_update_times(component: dict, parent_time: int) -> None:
        """
        This function gets a root node (with `parentComponentName = top`) and updates all its descendents
            with the absolute startTime
        """
        if component.get("startTime", -1) >= parent_time:
            return  # Handle loops
        component["startTime"] = max(
            parent_time + component["gapFromParent"], component.get("startTime", 0)
        )
        for child in parent_to_children[component["componentName"]]:
            recursive_update_times(child, parent_time=component["startTime"])

    raw_components = extracted_component_data or extract_component_data_from_dataframe(
        raw_generated_rows, config=config
    )

    parent_to_children = defaultdict(list)
    for raw_component in raw_components.values():
        parent_to_children[raw_component["parentComponentName"]].append(raw_component)
    nodes_without_parent = [
        c
        for c in raw_components.values()
        if c["parentComponentName"] not in raw_components
        or c["parentComponentName"] == c["componentName"]
    ]
    tx_start_time = min(c["txStartTime"] for c in nodes_without_parent)  # type: ignore
    for root in nodes_without_parent:
        recursive_update_times(root, tx_start_time)  # type: ignore

    components: List[Component] = []
    for raw_component in raw_components.values():
        components.append(
            Component(
                component_id=str(raw_component["componentName"]),
                start_time=raw_component.get("startTime", raw_component["txStartTime"]),  # type: ignore
                end_time=raw_component.get(  # type: ignore
                    "startTime", raw_component["txStartTime"]
                )
                + raw_component["duration"],
                duration=max(raw_component["duration"], 1),  # type: ignore
                children_ids=[
                    str(component["componentName"])
                    for component in parent_to_children[raw_component["componentName"]]
                    if component["componentName"] in raw_components
                ],
                group=str(raw_component["componentName"]),
                has_error=bool(raw_component["hasError"]),
                metadata=raw_component["metadata"],  # type: ignore
                component_type=get_component_type(
                    str(raw_component["componentName"]), config=config
                ),
            )
        )
    return list({c.component_id: c for c in components}.values())


def split_to_connected_components(components: List[Component]) -> List[List[Component]]:
    """
    This function splits the components to connected components.
    """
    components_by_id = {c.component_id: c for c in components}
    component_id_to_connected_component: Dict[str, Set[str]] = {}

    def recursive_add(c: Component, connected_component: Set[str]) -> None:
        connected_component.add(c.component_id)
        component_id_to_connected_component[c.component_id] = connected_component
        for child_id in c.children_ids:
            connected_component.add(child_id)
            if child_id in component_id_to_connected_component:
                connected_component.update(component_id_to_connected_component[child_id])
                for c_id in component_id_to_connected_component[child_id]:
                    component_id_to_connected_component[c_id] = connected_component
            else:
                component_id_to_connected_component[child_id] = connected_component
                recursive_add(components_by_id[child_id], connected_component)

    for component in sorted(components, key=lambda c: c.start_time):
        if component.component_id in component_id_to_connected_component:
            continue
        recursive_add(component, set())

    connected_components: List[Set[str]] = list({id(a): a for a in component_id_to_connected_component.values()}.values())
    return [[components_by_id[c_id] for c_id in connected_component] for connected_component in connected_components]


def prepare_tx_structure(
    transaction_id: str, components: List[Component]
) -> Optional[dict]:
    """
    Each transaction is being translated to a few rows, one for each component in the transaction graph.
    """
    tx_start = min([c.start_time for c in components])
    tx_end = max([c.end_time for c in components])
    nodes = {}
    edges = []
    timeline_items = []
    for component in components:
        nodes[component.component_id] = {
            "resource": {
                "id": component.component_id,
                "name": component.component_id,
                "serviceType": component.component_type,
                **(
                    {"region": str(component.metadata["region"])}
                    if component.metadata.get("region")
                    else {}
                ),
            },
            "id": component.component_id,
            "startTime": component.start_time,
            "duration": component.duration,
            "appTagId": "app_0000000000",
            "hasData": 1 if component.metadata else 0,
            "environmentVariables": {"isTruncated": 0, "body": component.metadata},
            "issues": [
                {
                    "category": "ERROR",
                    "id": component.component_id,
                    "name": "Error",
                    "type": "Error",
                    "message": "Issue",
                    "description": "Issue",
                }
            ]
            if component.has_error
            else [],
            "type": component.component_type,
        }
        timeline_items.append(
            {
                "id": component.component_id,
                "childrenIds": component.children_ids,
                "group": component.group,
                "start": component.start_time,
                "end": component.end_time,
                "duration": component.duration,
            }
        )
        for children_id in component.children_ids:
            edges.append(
                {
                    "source": component.component_id,
                    "target": children_id,
                }
            )
    if not edges:
        return None
    return {
        "traceStatus": "manual-traced",
        "nodesData": nodes,
        "graph": {"status": "completed", "edges": edges},
        "timeline": {
            "status": "completed",
            "range": {"start": tx_start, "end": tx_end},
            "items": timeline_items,
        },
        "details": {
            "status": "completed",
            "transactionId": transaction_id,
            "startTime": tx_start,
            "duration": tx_end - tx_start,
            "cost": 0,
            "regions": [],
            "issues": {"count": 0},
            "triggeredService": {},
        },
    }


def denormalize_data(driver: BaseDriver) -> None:
    input_dir = driver.get_normalized_generated_data_folder()
    config = driver.gen_t_config
    generated_data_dir = driver.get_generated_data_folder()
    if not os.listdir(input_dir):
        raise Exception("No data to denormalize")
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            unique_filename = (
                os.path.join(root, filename.replace(".csv", ".json"))
                .replace(input_dir, "")
                .replace("/", "_")
                .lstrip("_")
            )
            df = pd.read_csv(os.path.join(root, filename))

            os.makedirs(generated_data_dir, exist_ok=True)
            with open(
                os.path.join(generated_data_dir, unique_filename), "w"
            ) as output_file:
                for (txid, tx_started), raw_generated_rows in df.groupby(["traceId", "txStartTime"]):
                    components = prepare_components(raw_generated_rows, config=config)
                    all_connected_components = split_to_connected_components(components)
                    for connected_components in all_connected_components:
                        tx = prepare_tx_structure(txid, connected_components)
                        if tx:
                            output_file.write(json.dumps(tx) + ",\n")
