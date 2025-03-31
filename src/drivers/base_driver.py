import os
import shutil
import tarfile
import tempfile
import boto3
import drivers
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

    def get_model_gzip_file(self) -> str:
        print("Zipping model files")
        target_file = f"{tempfile.mkdtemp()}/model.tar.gz"
        tar = tarfile.open(target_file, "w:gz")
        for directory in self.get_model_directories():
            directory = os.path.abspath(directory)
            tar.add(directory, arcname=directory.replace(self.get_work_folder(), ""))
        tar.close()
        return target_file

    def get_model_gzip_size(self) -> int:
        target_file = self.get_model_gzip_file()
        compressed = os.path.getsize(target_file)
        os.remove(target_file)
        return compressed

    def upload_and_clear(self, force_clear: bool = False):
        model_path = self.get_model_gzip_file()
        s3_path = f"{self.get_driver_name()}/{self.gen_t_config.to_string()}.tar.gz"
        s3 = boto3.session.Session(profile_name="gent").resource('s3')
        print("Uploading model files to s3")
        s3.Bucket("gent-results").upload_file(model_path, s3_path)
        os.remove(model_path)
        hostname = os.getenv("HOSTNAME", "")
        if force_clear or hostname.endswith(".emulab.net") or hostname.endswith("bridges2.psc.edu"):
            print("Removing work folder", self.get_work_folder())
            shutil.rmtree(self.get_work_folder())

    def download(self):
        s3_key = f"{self.get_driver_name()}/{self.gen_t_config.to_string()}.tar.gz"
        local_path = os.path.join(self.get_work_folder(), "downloaded.tar.gz")

        print("Downloading model files from s3")
        os.makedirs(self.get_work_folder(), exist_ok=True)
        s3 = boto3.session.Session(profile_name="gent").resource('s3')
        s3.Bucket("gent-results").download_file(s3_key, local_path)

        print("Unzipping model files")
        tar = tarfile.open(local_path, "r:gz")
        tar.extractall(path=self.get_work_folder())
        tar.close()

    def get_work_folder(self) -> str:
        if not self.gen_t_config.traces_dir:
            raise ValueError("traces_dir must be set in config")
        return os.path.abspath(os.path.join(
            self.gen_t_config.traces_dir,
            "results",
            self.get_driver_name(),
            self.gen_t_config.to_string(),
        ))

    def get_generated_data_folder(self) -> str:
        return os.path.join(self.get_work_folder(), "normalized_data")

    def monitor_roc_path(self, number_of_bulks: int) -> str:
        return os.path.join(self.get_work_folder(), f"monitoring_roc_data_{number_of_bulks}.json")

    def forest_results_path(self, subtree_height: int) -> str:
        return os.path.join(self.get_work_folder(), f"forest_data_{subtree_height}.json")

    def bottleneck_path(self, number_of_bulks: int = 20) -> str:
        return os.path.join(self.get_work_folder(), f"bottleneck_data_{number_of_bulks}.json")

    def metadata_path(self, number_of_bulks: int = 20) -> str:
        return os.path.join(self.get_work_folder(), f"metadata_data_{number_of_bulks}.json")

    def get_results_key(self) -> Tuple[GenTBaseConfig, DriverType]:
        return self.gen_t_config, self.get_driver_name()
