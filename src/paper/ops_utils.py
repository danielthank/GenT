import json
import os
import time
import boto3
from typing import Dict, List, NamedTuple, Tuple
from dataclasses import asdict

from drivers.base_driver import DriverType
from drivers.netshare.netshare_driver import NetShareDriver
from drivers.tabFormer.tab_former_driver import TabFormerDriver
from ml.app_utils import GenTConfig, GenTBaseConfig

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.json")


class FidelityResult(NamedTuple):
    generation_speed: List[float] = []
    model_size: int = 0
    gzip_model_size: int = 0
    bottleneck_score: float = 0.


def store_and_upload_results(
    config: GenTBaseConfig, result: FidelityResult, driver: DriverType
) -> None:
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("{}")
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    results[json.dumps((asdict(config), driver))] = result._asdict()
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    upload_result(config, result, driver)


def upload_result(config: GenTBaseConfig, result: FidelityResult, driver: DriverType) -> None:
    s3 = boto3.session.Session(profile_name="gent").resource('s3').Bucket("gent-results")

    print("Uploading single config result to s3")
    filename = json.dumps((asdict(config), driver))
    s3.Object(key=f"results/{filename}_{int(time.time())}.json").put(Body=json.dumps(result._asdict()))

    print("Uploading results file to s3")
    s3.upload_file(RESULTS_FILE, f"results/{int(time.time())}.json")


def download_result() -> None:
    s3 = boto3.session.Session(profile_name="gent").resource('s3')
    print("Downloading results file from s3")
    latest_time = 0
    for obj in s3.Bucket("gent-results").objects.filter(Prefix="results/"):
        key = obj.key.split('/')[-1].split(".")[0]
        if not key.isdigit():
            continue
        obj_time = int(key)
        if obj_time > latest_time:
            latest_time = obj_time
    s3.Bucket("gent-results").download_file(f"results/{latest_time}.json", RESULTS_FILE)


def load_results() -> Dict[Tuple[GenTBaseConfig, DriverType], FidelityResult]:
    if not os.path.exists(RESULTS_FILE):
        return {}
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    return {
        (GenTConfig.load(**json.loads(k)[0]), json.loads(k)[1]): FidelityResult(**v)
        for k, v in results.items()
    }


def new_configuration_update() -> None:
    """
    This function is here to update all the file system paths and traces that include the old configuration.
    :return:
    """
    # Change results file
    new_results = {
        json.dumps((asdict(config), driver)): result._asdict()
        for (config, driver), result in load_results().items()
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(new_results, f, indent=4)

    # Change working directories
    base_dir = os.path.join(NetShareDriver(GenTConfig()).get_work_folder(), "..")
    for dir_name in os.listdir(base_dir):
        config = NetShareDriver(
            GenTConfig(
                **{k.split("=")[0]: k.split("=")[1] for k in dir_name.split(".")}
            )
        )
        old_path = os.path.join(base_dir, dir_name)
        if old_path != config.get_work_folder():
            print("Renaming", old_path, "to", config.get_work_folder())
            os.rename(old_path, config.get_work_folder())


def new_fidelity_update(fidelity_attr: str, func) -> None:
    """
    This function executes the new fidelity function and updates the results file.
    """
    for (config, driver), result in load_results().items():
        driver_class = NetShareDriver if driver == "netshare" else TabFormerDriver
        result_dict = result._asdict()
        if True: # driver == "tabFormer":  # result_dict[fidelity_attr] == 0.0:
            if not os.path.exists(driver_class(config).get_generated_data_folder()):
                print("Skipping", config, "because it has not been generated yet.")
                continue
            result_dict[fidelity_attr] = func(driver_class(config).get_generated_data_folder())
            store_and_upload_results(config, FidelityResult(**result_dict), driver)
            print(f"Updating result of {config} ({driver}) with {fidelity_attr} to {result_dict[fidelity_attr]}")


if __name__ == "__main__":
    # new_configuration_update()
    # new_fidelity_update("bottleneck_score", get_bottleneck_score)
    # new_fidelity_update("monitor_score", get_monitor_score)
    # upload_result()
    download_result()
