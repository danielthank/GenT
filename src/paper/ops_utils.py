import json
import os
import time
import boto3
from typing import Dict, List, NamedTuple, Tuple
from dataclasses import asdict

from drivers.base_driver import BaseDriver
from drivers.base_driver import DriverType
from ml.app_utils import GenTConfig, GenTBaseConfig

class FidelityResult(NamedTuple):
    generation_speed: List[float] = []
    model_size: int = 0
    gzip_model_size: int = 0
    bottleneck_score: float = 0.


def store_and_upload_results(
    driver: BaseDriver, result: FidelityResult, driver_type: DriverType
) -> None:
    results_path = os.path.join(driver.get_work_folder(), "../results.json")
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("{}")
    with open(results_path, "r") as f:
        results = json.load(f)
    results[json.dumps((asdict(driver.gen_t_config), driver_type))] = result._asdict()
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    upload_result(driver, result, driver_type)


def upload_result(driver: BaseDriver, result: FidelityResult, driver_type: DriverType) -> None:
    s3 = boto3.session.Session(profile_name="gent").resource('s3').Bucket("gent-results")

    print("Uploading single config result to s3")
    filename = json.dumps((asdict(driver.gen_t_config), driver_type))
    s3.Object(key=f"results/{filename}_{int(time.time())}.json").put(Body=json.dumps(result._asdict()))

    print("Uploading results file to s3")
    results_path = os.path.join(driver.get_work_folder(), "../results.json")
    s3.upload_file(results_path, f"results/{int(time.time())}.json")


def load_results(driver: BaseDriver) -> Dict[Tuple[GenTBaseConfig, DriverType], FidelityResult]:
    results_path = os.path.join(driver.get_work_folder(), "../results.json")
    if not os.path.exists(results_path):
        return {}
    with open(results_path, "r") as f:
        results = json.load(f)
    return {
        (GenTConfig.load(**json.loads(k)[0]), json.loads(k)[1]): FidelityResult(**v)
        for k, v in results.items()
    }
