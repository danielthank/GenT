import itertools
import os
import pickle
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union, Literal

MetadataType = Literal["str", "int"]
ComponentType = Literal["http", "lambda", "triggerBy", "jaeger"]
EMPTY = "Unknown"


@dataclass(frozen=True)
class GenTBaseConfig:
    chain_length: int = 3
    iterations: int = 100
    metadata_str_size: int = 2
    metadata_int_size: int = 3
    batch_size: int = 10
    is_test: bool = False

    def to_string(self) -> str:
        default_config = asdict(GenTBaseConfig())

        return ".".join(f"{k}={str(v).split('/')[-1]}" for k, v in asdict(self).items() if v != default_config.get(k))

    def replace(self, key: str, value: Any) -> "GenTBaseConfig":
        data = asdict(self)
        data[key] = value
        return self.__class__(**data)

    def get_raw_normalized_data_dir(self) -> str:
        return os.path.join(
            os.path.dirname(__file__) if not self.is_test else "/tmp",
            "raw_normalized_data",
            self.to_string(),
        )

    @lru_cache(maxsize=1)
    def get_raw_data_count(self) -> int:
        data_dir = self.get_raw_normalized_data_dir()
        return sum(1 for file in os.listdir(data_dir) for _ in open(os.path.join(data_dir, file)))

    @staticmethod
    def load(**kwargs) -> "GenTBaseConfig":
        return GenTBaseConfig(**kwargs)


@dataclass(frozen=True)
class GenTConfig(GenTBaseConfig):
    with_gcn: bool = True
    discriminator_dim: Tuple[int, ...] = (128,)
    generator_dim: Tuple[int, ...] = (128,)
    start_time_with_metadata: bool = False
    independent_chains: bool = False
    tx_start: int = 0
    tx_end: int = 1000
    # Remove default value once all initializations have traces_dir
    traces_dir: str = None

    @staticmethod
    def load(**kwargs) -> "GenTConfig":
        res = GenTConfig(**kwargs)
        if isinstance(res.discriminator_dim, list):
            res = res.replace("discriminator_dim", tuple(res.discriminator_dim))
        if isinstance(res.generator_dim, list):
            res = res.replace("generator_dim", tuple(res.generator_dim))
        return res


class MetadataItem(NamedTuple):
    metadata_index: int
    key: str
    values: Union[List[str], List[int]]
    metadata_type: MetadataType


class ComponentMetadataStructure(NamedTuple):
    key_to_item: Dict[str, MetadataItem]


class ComponentsMetadata(NamedTuple):
    component_to_metadata: Dict[str, ComponentMetadataStructure]
    # We want to use the same index for the same key in different components
    general_key_to_index_and_type: Dict[str, Tuple[int, MetadataType]]
    component_name_to_type: Dict[str, ComponentType]


global_metadata = ComponentsMetadata({}, {}, {})
_consumed_indexes: Set[int] = set()


def clear():
    global global_metadata
    global _consumed_indexes
    global_metadata = ComponentsMetadata({}, {}, {})
    _consumed_indexes = set()


def get_metadata_size(config: GenTConfig) -> int:
    return config.metadata_int_size + config.metadata_str_size


def _get_new_index(
    component: ComponentMetadataStructure,
    metadata_type: MetadataType,
    config: GenTConfig,
) -> int:
    """
    When we need to choose the index of the metadata, we want to use indexes 0-9 for string
        metadata and 10-19 for int metadata.
    Moreover, we prefer to not use an already consumed index.
    """
    str_size = config.metadata_str_size
    int_size = config.metadata_int_size
    if metadata_type == "str":
        relevant_indexes = set(range(0, str_size))
    else:
        relevant_indexes = set(range(str_size, str_size + int_size))
    global_available_indexes = relevant_indexes - _consumed_indexes
    if global_available_indexes:
        return min(global_available_indexes)
    available_indexes = relevant_indexes - set(
        v.metadata_index
        for v in component.key_to_item.values()
        if v.metadata_type == metadata_type
    )
    if len(available_indexes) == 0:
        raise ValueError("No more indexes available for this metadata type")
    return min(available_indexes)


def get_metadata_index(
    component_name: str,
    metadata_key: str,
    metadata_type: MetadataType,
    config: GenTConfig,
) -> int:
    """
    This function adds a metadata key to the component.
    If the key already exists in the component, it does nothing.
    If the key already exists in other components, it adds it to this component under the same index.
    Otherwise, it creates a new index for this key.
    """
    if component_name not in global_metadata.component_to_metadata:
        global_metadata.component_to_metadata[
            component_name
        ] = ComponentMetadataStructure({})
    component = global_metadata.component_to_metadata[component_name]
    # Check if this key has already been added to the component
    if metadata_key in component.key_to_item:
        if component.key_to_item[metadata_key].metadata_type != metadata_type:
            raise ValueError("Metadata type mismatch (to the same component)")
        return component.key_to_item[metadata_key].metadata_index
    # Check if this key has already been added to other components
    elif metadata_key in global_metadata.general_key_to_index_and_type:
        suggested_index, suggested_type = global_metadata.general_key_to_index_and_type[
            metadata_key
        ]
        if suggested_type != metadata_type:
            raise ValueError("Metadata type mismatch (to other component)")
        component.key_to_item[metadata_key] = MetadataItem(
            suggested_index, metadata_key, [], metadata_type
        )
        return suggested_index
    # Find the best next index and use it
    else:
        new_index = _get_new_index(component, metadata_type, config=config)
        component.key_to_item[metadata_key] = MetadataItem(
            new_index, metadata_key, [], metadata_type
        )
        global_metadata.general_key_to_index_and_type[metadata_key] = (
            new_index,
            metadata_type,
        )
        _consumed_indexes.add(new_index)
        return new_index


def get_metadata_value(
    component_name: str,
    metadata_key: str,
    metadata_value: Union[int, str],
    metadata_type: MetadataType,
):
    if metadata_type == "int":
        return metadata_value
    component = global_metadata.component_to_metadata[component_name]
    known_values = component.key_to_item[metadata_key].values
    if metadata_value not in known_values:
        known_values.append(metadata_value)
    return f"metadata_merged_value_{known_values.index(metadata_value)}"

def get_features(
    component_name: str,
    int_features: List[Tuple[str, MetadataType, int]],
    str_features: List[Tuple[str, MetadataType, str]],
    config: GenTConfig,
) -> List[Union[str, int]]:
    """
    This function returns the features in the correct order to the normalizer.
    """
    result: List[Any] = []
    result += [EMPTY] * config.metadata_str_size
    result += [0] * config.metadata_int_size
    if len(int_features) > config.metadata_int_size:
        # Trimming some of the int features due to config limit
        int_features = int_features[: config.metadata_int_size]
    if len(str_features) > config.metadata_str_size:
        # Trimming some of the str features due to config limit
        str_features = str_features[: config.metadata_str_size]
    if not all(str_features):
        raise ValueError("All metadata strings should ne non-empty")
    for feature_name, feature_type, feature_value in itertools.chain(
        int_features, str_features
    ):
        index = get_metadata_index(
            component_name, feature_name, feature_type, config=config
        )
        result[index] = get_metadata_value(component_name, feature_name, feature_value, feature_type)
    return result


def get_key_name(
    component_name: str, metadata_index: int, config: GenTConfig
) -> Optional[str]:
    """
    This function returns the name of the key in the metadata dictionary.
    """
    if not global_metadata.component_to_metadata:
        load_global_metadata(config)
    if component_name not in global_metadata.component_to_metadata:
        return None
    return next(
        (
            key
            for key, item in global_metadata.component_to_metadata[
                component_name
            ].key_to_item.items()
            if item.metadata_index == metadata_index
        ),
        None,
    )


def get_key_value(
    component_name: str, metadata_key: str, metadata_value: Union[int, str], config: GenTConfig
) -> Union[int, str]:
    if not global_metadata.component_to_metadata:
        load_global_metadata(config)
    if isinstance(metadata_value, str) and metadata_value.startswith("metadata_merged_value_"):
        values_index = int(metadata_value[len("metadata_merged_value_"):])
        if component_name not in global_metadata.component_to_metadata:
            # Not all components have metadata (e.g. entry points)
            return EMPTY
        values = global_metadata.component_to_metadata[component_name].key_to_item[metadata_key].values
        if values_index >= len(values):
            return values[0]  # We better guess than crash / return empty
        return values[values_index]
    return metadata_value


def remember_type(component_name: str, component_type: ComponentType) -> None:
    """
    This function remembers the type of the component.
    """
    global_metadata.component_name_to_type[component_name] = component_type


def get_component_type(
    component_name: str, config: GenTConfig
) -> Optional[ComponentType]:
    """
    This function returns the type of the component.
    """
    if not global_metadata.component_name_to_type:
        load_global_metadata(config)
    return global_metadata.component_name_to_type.get(component_name, None)


def store_global_metadata(config: GenTConfig) -> None:
    """
    This function stores the global metadata in the config.
    """
    if config.is_test:
        directory = "/tmp"
    else:
        directory = os.path.join(os.path.dirname(__file__), "metadata_mappings")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, f"{config.to_string()}.json"), "wb") as f:
        f.write(pickle.dumps(global_metadata))


def load_global_metadata(config: GenTConfig) -> None:
    """
    This function loads the global metadata from the config.
    """
    global global_metadata
    if config.is_test:
        directory = "/tmp"
    else:
        directory = os.path.join(os.path.dirname(__file__), "metadata_mappings")
    with open(os.path.join(directory, f"{config.to_string()}.json"), "rb") as f:
        global_metadata = pickle.loads(f.read())
