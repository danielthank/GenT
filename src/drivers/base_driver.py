import os
from typing import Tuple, List, Literal
from ml.app_utils import GenTBaseConfig

DriverType = Literal["netshare", "tabFormer", "genT"]

class BaseDriver:
    def __init__(self, gen_t_config: GenTBaseConfig):
        self.gen_t_config: GenTBaseConfig = gen_t_config

    def get_driver_name(self) -> DriverType:
        pass

    def pretty_name(self) -> str:
        pass

    def get_normalized_generated_data_folder(self):
        pass

    def train_and_generate(self) -> None:
        pass

    def generate(self, param: int = 0, from_downloaded: bool = False) -> None:
        pass

    def get_model_directories(self) -> List[str]:
        pass

    def get_model_size(self) -> int:
        print("Calculating model size")
        return sum(
            os.path.getsize(os.path.join(iteration_dir, file))
            for iteration_dir in self.get_model_directories()
            for file in (os.listdir(iteration_dir) if os.path.isdir(iteration_dir) else [])
        )

    def get_model_gzip_size(self) -> int:
        target_file = self.get_model_gzip_file()
        compressed = os.path.getsize(target_file)
        os.remove(target_file)
        return compressed

    def get_models_folder(self) -> str:
        return self.gen_t_config.models_dir

    def get_results_folder(self) -> str:
        return self.gen_t_config.results_dir

    def get_generated_data_folder(self) -> str:
        return os.path.join(self.get_results_folder(), "normalized_data")

    def monitor_roc_path(self, number_of_bulks: int) -> str:
        return os.path.join(self.get_results_folder(), f"monitoring_roc_data_{number_of_bulks}.json")

    def forest_results_path(self, subtree_height: int) -> str:
        return os.path.join(self.get_results_folder(), f"forest_data_{subtree_height}.json")

    def bottleneck_path(self, number_of_bulks: int = 20) -> str:
        return os.path.join(self.get_results_folder(), f"bottleneck_data_{number_of_bulks}.json")

    def metadata_path(self, number_of_bulks: int = 20) -> str:
        return os.path.join(self.get_results_folder(), f"metadata_data_{number_of_bulks}.json")

    def get_results_key(self) -> Tuple[GenTBaseConfig, DriverType]:
        return self.gen_t_config, self.get_driver_name()
