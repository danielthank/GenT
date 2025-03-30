import math
import os
import shutil
import subprocess
from typing import Optional, List

from drivers.base_driver import BaseDriver, DriverType
from ml.app_denormalizer import denormalize_data
from ml.app_normalizer import normalize_data
from ml.app_utils import GenTConfig
import ml
TAB_FORMER_DIR = os.path.abspath(os.path.join(os.path.dirname(ml.__path__[0]), "..", "..", "TabFormer"))
TAB_FORMER_INTERPRETER = os.path.join(TAB_FORMER_DIR, "venv", "bin", "python")
BASE_TRAIN_PARAMS = [
    "--data_extension",
    "",
    "--data_fname",
    "observability.v1",
    "--do_train",
    "--field_ce",
    "--field_hs",
    "768",
    "--flatten",
    "--lm_type",
    "gpt2",
    "--vocab_file",
    "observability_vocab.nb",
]
BASE_GENERATE_PARAMS = [
    "--data_extension",
    "",
    "--data_fname",
    "observability.v1",
    "--hidden_size",
    "1020",
    "--num_seed_trans",
    "1",
    "--store_csv",
]


class TabFormerDriver(BaseDriver):
    def get_driver_name(self) -> DriverType:
        return "tabFormer"

    def pretty_name(self) -> str:
        return "TabFormer"

    def get_normalized_generated_data_folder(self):
        for gpt_dir in os.listdir(self.get_work_folder()):
            if gpt_dir.startswith("gpt2-"):
                for file in os.listdir(os.path.join(self.get_work_folder(), gpt_dir)):
                    if file.startswith("checkpoint-"):
                        return os.path.join(self.get_work_folder(), gpt_dir, file)
        raise ValueError("Could not find generated data folder")

    def _train(self) -> None:
        cmd = [
            TAB_FORMER_INTERPRETER,
            "main.py",
            *BASE_TRAIN_PARAMS,
            "--num_train_epochs",
            str(self.gen_t_config.iterations),
            "--save_steps",
            "1000",
            "--data_root",
            self.gen_t_config.get_raw_normalized_data_dir(),
            "--output_dir",
            self.get_work_folder(),
            "--data_type",
            "observability",
            "--chain_length",
            str(self.gen_t_config.chain_length),
            "--metadata_int_columns",
            str(self.gen_t_config.metadata_int_size),
            "--metadata_str_columns",
            str(self.gen_t_config.metadata_str_size),
        ]
        print("Executed training command:", " ".join(cmd))
        subprocess.check_output(cmd, cwd=TAB_FORMER_DIR, env={"WANDB_LOG_MODEL": "checkpoint", "WANDB_WATCH": "all", "WANDB_PROJECT": "GenT"})

    def _generate(self, lines_to_generate: Optional[int] = None) -> None:
        max_checkpoint = self.get_highest_checkpoint(self.get_work_folder())
        cmd = [
            TAB_FORMER_INTERPRETER,
            "gpt_eval.py",
            *BASE_GENERATE_PARAMS,
            "--checkpoint",
            str(max_checkpoint),
            "--data_dir",
            self.gen_t_config.get_raw_normalized_data_dir(),
            "--output_dir",
            self.get_work_folder(),
            "--data_type",
            "observability",
            "--chain_length",
            str(self.gen_t_config.chain_length),
            "--metadata_int_columns",
            str(self.gen_t_config.metadata_int_size),
            "--metadata_str_columns",
            str(self.gen_t_config.metadata_str_size),
            "--lines_to_generate",
            str(lines_to_generate or min(self.gen_t_config.get_raw_data_count(), 10_000))
        ]
        print("Executed generation command:", " ".join(cmd))
        subprocess.check_output(cmd, cwd=TAB_FORMER_DIR)  # Needed to create the output dir
        output_files = os.listdir(self.get_normalized_generated_data_folder())
        if len(output_files) != 1:
            raise ValueError(f"Expected 1 output file, got {len(output_files)} in {self.get_normalized_generated_data_folder()}")
        os.rename(
            os.path.join(self.get_normalized_generated_data_folder(), output_files[0]),
            os.path.join(self.get_normalized_generated_data_folder(), "generated.csv")
        )
        denormalize_data(self)

    def train_and_generate(self) -> None:
        shutil.rmtree(self.get_work_folder(), ignore_errors=True)
        normalize_data(self.gen_t_config.traces_dir, self.gen_t_config)
        self._train()
        self._generate()

    def generate(self, param: int, from_downloaded: bool = False) -> None:
        self._generate(lines_to_generate=param)
        denormalize_data(self)

    @staticmethod
    def get_highest_checkpoint(work_folder: str) -> int:
        highest = 0
        for file in os.listdir(work_folder):
            if file.startswith("checkpoint-"):
                highest = max(highest, int(file.split("-")[1]))
        return highest


    def get_model_directories(self) -> List[str]:
        highest = self.get_highest_checkpoint(self.get_work_folder())
        return [os.path.join(self.get_work_folder(), f"checkpoint-{highest}")]


if __name__ == "__main__":
    TabFormerDriver(GenTConfig(chain_length=2)).train_and_generate()
