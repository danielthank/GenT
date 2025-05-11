import os
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Tuple, Literal

MetadataType = Literal["str", "int"]
EMPTY = "Unknown"

@dataclass(frozen=True)
class GenTBaseConfig:
    chain_length: int = 3
    iterations: int = 100
    metadata_str_size: int = 0
    metadata_int_size: int = 0
    batch_size: int = 10
    is_test: bool = False
    # Remove default value once all initializations have traces_dir
    traces_dir: str = None
    models_dir: str = None
    results_dir: str = None

    def to_string(self) -> str:
        default_config = asdict(GenTBaseConfig())

        return ".".join(f"{k}={str(v).split('/')[-1]}" for k, v in asdict(self).items() if v != default_config.get(k) and k != "traces_dir")

    def replace(self, key: str, value: Any) -> "GenTBaseConfig":
        data = asdict(self)
        data[key] = value
        return self.__class__(**data)

    def get_raw_normalized_data_dir(self) -> str:
        return os.path.join(
            os.path.dirname(__file__) if self.is_test else "/tmp",
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
    independent_chains: bool = False
    tx_start: int = 0
    tx_end: int = 1000

    @staticmethod
    def load(**kwargs) -> "GenTConfig":
        res = GenTConfig(**kwargs)
        if isinstance(res.discriminator_dim, list):
            res = res.replace("discriminator_dim", tuple(res.discriminator_dim))
        if isinstance(res.generator_dim, list):
            res = res.replace("generator_dim", tuple(res.generator_dim))
        return res
