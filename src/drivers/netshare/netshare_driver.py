import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd

from drivers.base_driver import BaseDriver
from ml.app_denormalizer import denormalize_data
from ml.app_normalizer import normalize_data
try:
    from netshare.configs import load_from_file
    from netshare.generate.generate import generate
    from netshare.api import Generator
except ImportError:
    pass


class NetShareDriver(BaseDriver):
    def get_driver_name(self) -> str:
        return "netshare"

    def pretty_name(self) -> str:
        return "NetShare"

    def get_config_path(self) -> str:
        dirname = os.path.join(os.path.dirname(__file__), "configs")
        os.makedirs(dirname, exist_ok=True)
        return os.path.join(
            dirname,
            "traces+" + self.gen_t_config.to_string() + ".json",
        )

    def get_normalized_generated_data_folder(self):
        return os.path.join(self.get_work_folder(), "output_data")

    def as_file(self, generate_num_train_sample: int = 1500, num_real_samples: Optional[int] = None) -> str:
        timeseries_fields: List[Dict[Any, Any]] = [
            {"column": "chain", "type": "string", "encoding": "categorical"}
        ]
        for i in range(self.gen_t_config.chain_length):
            timeseries_fields.extend(
                [
                    {
                        "column": f"gapFromParent_{i}",
                        "type": "integer",
                        "encoding": "bit",
                        "n_bits": 16,
                        "truncate": True,
                    },
                    {
                        "column": f"duration_{i}",
                        "type": "integer",
                        "encoding": "bit",
                        "n_bits": 16,
                        "truncate": True,
                    },
                    {
                        "column": f"hasError_{i}",
                        "type": "integer",
                        "encoding": "bit",
                        "n_bits": 1,
                    },
                ]
            )

            for j in range(self.gen_t_config.metadata_str_size):
                timeseries_fields.append(
                    {
                        "column": f"metadata_{i}_{j}",
                        "type": "string",
                        "encoding": "word2vec",
                    }
                )
            for j in range(self.gen_t_config.metadata_int_size):
                timeseries_fields.append(
                    {
                        "column": f"metadata_{i}_{self.gen_t_config.metadata_str_size + j}",
                        "type": "float",
                        "normalization": "ZERO_ONE",
                    }
                )
        base_config = json.load(
            open(os.path.join(os.path.dirname(__file__), "observability.json"))
        )
        base_config["learn"]["timeseries"] = timeseries_fields
        base_config["model"]["config"]["iteration"] = self.gen_t_config.iterations
        base_config["model"]["config"]["generate_num_train_sample"] = generate_num_train_sample
        base_config["model"]["config"][
            "batch_size"
        ] = self.gen_t_config.batch_size
        base_config["src"] = {
            "chain_length": self.gen_t_config.chain_length,
            "metadata_str_size": self.gen_t_config.metadata_str_size,
            "metadata_int_size": self.gen_t_config.metadata_int_size,
        }
        base_config["global_config"]["work_folder"] = self.get_work_folder()
        base_config["input_adapters"]["data_source"][
            "input_folder"
        ] = self.gen_t_config.get_raw_normalized_data_dir()
        if num_real_samples is not None:
            base_config["model"]["config"]["num_real_samples"] = num_real_samples

        config_path = self.get_config_path()
        json.dump(base_config, open(config_path, "w"), indent=4)
        return config_path

    def train_and_generate(self) -> None:
        shutil.rmtree(self.get_work_folder(), ignore_errors=True)
        normalize_data(self.gen_t_config.traces_dir, self.gen_t_config)
        generator = Generator(config=self.as_file(
            generate_num_train_sample=6000,
        ))
        generator.train()
        self.generate(param=6000, from_downloaded=False)

    @staticmethod
    def generate_loop(generator: "Generator", generated_data_dir_name: str, target_rows: int) -> None:
        temp_agg_file = tempfile.NamedTemporaryFile(delete=False).name
        final_path = os.path.join(generated_data_dir_name, "generated.csv")
        current_rows_count = 0
        file_exists = False
        if os.path.exists(final_path):
            file_exists = True
            current_rows_count = len(open(final_path,'r').readlines())
            print(f"Found existing generated data ({current_rows_count} rows), appending to it (target: {target_rows} rows)")
            shutil.copyfile(final_path, temp_agg_file)
        while current_rows_count < target_rows:
            generator.generate()
            all_new_data = pd.concat([
                pd.read_csv(os.path.join(generated_data_dir_name, filename), index_col=None, header=0)
                for filename in os.listdir(generated_data_dir_name) if filename != "generated.csv"
            ], axis=0, ignore_index=True)
            all_new_data.to_csv(temp_agg_file, mode='a', header=not file_exists, index=False)
            shutil.copyfile(temp_agg_file, final_path)
            file_exists = True
            [os.remove(os.path.join(generated_data_dir_name, filename)) for filename in
             os.listdir(generated_data_dir_name) if filename != "generated.csv"]
            current_rows_count += len(all_new_data)
            print(f"Number of generated rows: {current_rows_count}/{target_rows}")

    def generate(self, param: int, from_downloaded: bool = False, target_rows: Optional[int] = None) -> None:
        generator = Generator(config=self.as_file(
            generate_num_train_sample=param,
            num_real_samples=10000 if from_downloaded else None
        ))
        self.generate_loop(
            generator=generator,
            generated_data_dir_name=self.get_normalized_generated_data_folder(),
            target_rows=target_rows or self.gen_t_config.get_raw_data_count(),
        )
        denormalize_data(self)

    def get_model_directories(self) -> List[str]:
        checkpoint = os.path.join(
            self.get_work_folder(),
            "models",
            "pre_processed_data",
            "checkpoint",
            "iteration_id-0",
        )
        if not os.path.isdir(checkpoint):
            checkpoint = os.path.join(
                self.get_work_folder(),
                "models",
                "checkpoint",
                "iteration_id-0",
            )
        if not os.path.isdir(checkpoint):
            raise ValueError("Could not find model directory")

        highest = 0
        filename = ""
        for file in os.listdir(os.path.join(checkpoint, "..")):
            if file.startswith("iteration_id-"):
                highest = max(highest, int(file.split("-")[-1]))
                filename = file
        model_path = os.path.join(checkpoint, "..", filename)
        extra_paths = [
            os.path.join(self.get_work_folder(), "pre_processed_data", "data_feature_output.pkl"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "data_attribute_output.pkl"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "data_feature_fields.pkl"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "data_attribute_fields.pkl"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "annoy_idx_ele_dict.json"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "word2vec.ann"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "word2vec_vecSize_10.model"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "gt_lengths.npy"),
            os.path.join(self.get_work_folder(), "pre_processed_data", "data_train_npz", "data_train_0.npz")
        ]
        return extra_paths + [model_path]
