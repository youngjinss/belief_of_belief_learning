import os
import yaml
import numpy as np
from datetime import datetime

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_name):
    """
    YAML 설정 파일을 로드하는 함수

    Args:
        config_name (str): 설정 파일 이름 (binance_api, data_preprocess, data_loader, model)

    Returns:
        dict: 설정 정보를 담고 있는 딕셔너리
    """
    # 2단계 부모 디렉토리를 찾아서 설정 파일 경로 지정
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), f"config/{config_name}.yaml"
    )

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

    elif config_name == "model":
        # agent_action_cols 자동 생성
        if config["columns"]["agent_action_cols"] is None:
            long_cols = [f"l_{i}" for i in range(config["model"]["tft"]["n_bins"])]
            short_cols = [f"s_{i}" for i in range(config["model"]["tft"]["n_bins"])]
            config["columns"]["agent_action_cols"] = long_cols + short_cols

    return config
