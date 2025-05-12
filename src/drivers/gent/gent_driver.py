import os
import shutil
import tarfile
import tempfile
import time
from typing import List

from drivers.base_driver import BaseDriver, DriverType
from drivers.gent.metadata_generator_ctgan import MetadataGenerator
from drivers.gent.start_time_generator_ctgan import StartTimesGenerator
from ml.app_utils import GenTConfig

class GenTDriver(BaseDriver):
    def __init__(self, gen_t_config: GenTConfig):
        self.gen_t_config: GenTConfig
        super().__init__(gen_t_config)
        self.metadata_generator = None
        self.start_time_generator = None

    def get_driver_name(self) -> DriverType:
        return "genT"

    def pretty_name(self) -> str:
        return "GenT"

    def get_normalized_generated_data_folder(self):
        return os.path.join(self.get_results_folder(), "generated")
    
    def _get_model_files(self) -> List[str]:
        return [
            os.path.join(self.get_models_folder(), "start_time", "start_time_ctgan_generator.pkl"),
            os.path.join(self.get_models_folder(), "start_time", "graph_counts.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "root_ctgan_generator.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "chained_ctgan_generator.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "all_graph_values.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "all_chain_values.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "node_to_index.pkl"),
            os.path.join(self.get_models_folder(), "metadata", "graph_index_to_chains.pkl"),
        ]

    def get_model_size(self) -> int:
        return sum(os.path.getsize(f) for f in self._get_model_files())

    def get_model_gzip_file(self) -> str:
        print("Zipping model files")
        target_file = f"{tempfile.mkdtemp()}/model.tar.gz"
        tar = tarfile.open(target_file, "w:gz")
        for f in self._get_model_files():
            tar.add(f)
        tar.close()
        return target_file

    def train(self) -> None:
        shutil.rmtree(self.get_results_folder(), ignore_errors=True)
        start = time.time()

        start_time_generator = StartTimesGenerator.get(self.gen_t_config)
        start_time_generator.train()
        start_time_generator.save()
        metadata_generator = MetadataGenerator.get(self.gen_t_config)
        metadata_generator.train()
        metadata_generator.save()

        print(f"Training took {time.time() - start} seconds")

    def train_and_generate(self) -> None:
        self.train()
        self.generate()

    def generate(self, param: int = 0, from_downloaded: bool = False, suffix: str = '') -> None:
        start_time_generator = self.get_start_time_generator()
        metadata_generator = self.get_metadata_generator()

        start = time.time()
        print("Generating start times")
        ts_corpus = start_time_generator.generate_timestamps_corpus()
        metadata_generator.generate_traces_corpus(
            target_dir_path=self.get_generated_data_folder() + suffix,
            ts_corpus=ts_corpus,
        )
        print(f"Full Generation took {time.time() - start} seconds into {self.get_generated_data_folder()}")

    def get_start_time_generator(self) -> StartTimesGenerator:
        if not self.start_time_generator:
            self.start_time_generator = StartTimesGenerator.get(self.gen_t_config)
            self.start_time_generator.load()
        return self.start_time_generator

    def get_metadata_generator(self) -> MetadataGenerator:
        if not self.metadata_generator:
            self.metadata_generator = MetadataGenerator.get(self.gen_t_config)
            self.metadata_generator.load()
        return self.metadata_generator

    def store_metric(self, metric_name: str, value: float) -> None:
        open(os.path.join(self.get_results_folder(), f"{metric_name}.txt"), "w").write(str(value))

    def get_metric(self, metric_name: str) -> float:
        path = os.path.join(self.get_results_folder(), f"{metric_name}.txt")
        if not os.path.exists(path):
            raise Exception(f"Metric {metric_name} not found")
        return float(open(path, "r").read())
