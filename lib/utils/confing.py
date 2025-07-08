import os
import yaml
import numpy as np
from datetime import datetime

project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def load_config(config_name):
    """
    YAML 설정 파일을 로드하는 함수

    Args:
        config_name (str): 설정 파일 이름 (binance_api, data_preprocess, model)

    Returns:
        dict: 설정 정보를 담고 있는 딕셔너리
    """
    # 3단계 부모 디렉토리를 찾아서 설정 파일 경로 지정
    config_path = os.path.join(project_dir, f"config/{config_name}.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 특정 설정 파일에 대한 추가 처리
    if config_name == "data_preprocess":
        # 기본 경로 설정
        if config["paths"]["project_dir"] is None:
            config["paths"]["project_dir"] = project_dir

        if config["paths"]["data_dir"] is None:
            config["paths"]["data_dir"] = os.path.join(
                project_dir,
                "data/binance/futures/um/monthly/aggTrades/BTCUSDT",
            )

        if config["paths"]["output_dir"] is None:
            config["paths"]["output_dir"] = os.path.join(
                config["paths"]["project_dir"],
                f"data/binance/futures/um/monthly/position_distribution/{config['preprocess']['window_size']}",
            )

        if config["paths"]["log_dir"] is None:
            config["paths"]["log_dir"] = os.path.join(
                config["paths"]["project_dir"],
                "log",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

        if config["data_processing"]["quantile_bins"] is None:
            config["data_processing"]["quantile_bins"] = np.linspace(
                -1, 1, 2 * config["preprocess"]["quantile"] + 1
            ).tolist()  # numpy 배열을 리스트로 변환

    elif config_name == "train":
        # dataloader.target_cols 자동 생성
        if config["dataloader"]["columns"]["target_cols"] is None:
            n_bins = config["dataloader"]["n_bins"]
            long_cols = [f"l_{i}" for i in range(n_bins)]
            short_cols = [f"s_{i}" for i in range(n_bins)]
            config["dataloader"]["columns"]["target_cols"] = long_cols + short_cols

    elif config_name == "inference":
        # dataloader.target_cols 자동 생성
        if config["dataloader"]["columns"]["target_cols"] is None:
            n_bins = config["dataloader"]["n_bins"]
            long_cols = [f"l_{i}" for i in range(n_bins)]
            short_cols = [f"s_{i}" for i in range(n_bins)]
            config["dataloader"]["columns"]["target_cols"] = long_cols + short_cols

        # model.save_path.path 자동 생성
        if config["model"]["hbt"]["save_path"]["path"] is None:
            config["model"]["hbt"]["save_path"][
                "path"
            ] = f"{project_dir}/log/inference/{config['model']['hbt']['save_path']['version']}/"

    else:
        raise ValueError(f"Invalid config name: {config_name}")

    return config
