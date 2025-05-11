import os
import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from typing import List, Dict, Union
from pathlib import Path
from ml.app_utils import GenTConfig
from drivers.gent.data import get_full_dataset_start_times

class StartTimesGenerator:
    def __init__(self, gen_t_config: GenTConfig, is_roll: bool = False) -> None:
        self.gen_t_config = gen_t_config
        self.models_dir = os.path.join(gen_t_config.models_dir, "start_time")
        self.is_roll = is_roll

    def train(self) -> None:
        print("StartTime generator started training")
        dataset = self.prepare()
        metadata = Metadata()
        metadata.add_table("transactions")
        metadata.add_column("graph", sdtype="categorical")
        metadata.add_column("startTime", sdtype="numerical")
        self.generator = CTGANSynthesizer(
            epochs=self.gen_t_config.iterations,
            batch_size=self.gen_t_config.batch_size,
            metadata=metadata,
            enforce_min_max_values=True,
            verbose=True
        )
        self.generator.fit(dataset)

    def prepare(self) -> None:
        dataset, all_dataset = get_full_dataset_start_times(self.gen_t_config, load_all=self.is_roll)

        all_graph_values = sorted(all_dataset["graph"].unique())
        dataset["graph"] = dataset["graph"].apply(all_graph_values.index)
        self.graph_counts = dataset["graph"].value_counts().to_dict()
        return dataset

    def _generate_corpus(self) -> Dict[str, List[int]]:
        rows = []
        for graph, count in self.graph_counts.items():
            rows.extend([{"graph": graph} for _ in range(count)])
        known_columns = pd.DataFrame(rows)
        return self.generator.sample_remaining_columns(
            known_columns=known_columns,
            max_tries_per_batch=500
        )

    def generate_timestamps_corpus(self) -> Dict[str, List[int]]:
        return self._generate_corpus()

    def save(self):
        path = self.models_dir
        os.makedirs(path, exist_ok=True)
        # This is a hack to make the model smaller
        # sampler, self.generator._data_sampler = self.generator._data_sampler, DataSampler(np.zeros((0, 0)), np.zeros((0, 0)), True)
        # self.generator.__doc__ = None
        self.generator.save(f"{path}/start_time_ctgan_generator.pkl")
        # self.generator._data_sampler = sampler

    def load(self):
        path = self.models_dir
        self.generator = CTGANSynthesizer.load(f"{path}/start_time_ctgan_generator.pkl")

    @staticmethod
    def get(gen_t_config: GenTConfig, is_roll: bool = False) -> "StartTimesGenerator":
        return StartTimesGenerator(
            gen_t_config,
            is_roll=is_roll,
        )


def train_and_save(gen_t_config: GenTConfig, path: Union[str, Path], is_roll: bool = False):
    """
    This function is here to support multiprocessing
    """
    gen = StartTimesGenerator.get(gen_t_config, is_roll=is_roll)
    gen.train()
    gen.save()
