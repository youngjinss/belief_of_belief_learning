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

        # log_dir 자동 설정
        if "log_dir" not in config:
            config["log_dir"] = os.path.join(
                project_dir, "log", datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            )

        # 모델 저장 경로 자동 설정
        selected_model = config["model"]["select"]
        if selected_model in ["hbt", "tft"]:
            config["active_model"] = selected_model
            config["active_model_config"] = config["model"][selected_model]

            # 모델 저장 경로를 log_dir 안에 설정
            if "save_path" in config["model"][selected_model]:
                proposed_filename = f"best_{selected_model}_model.pth"
                benchmark_filename = f"best_{selected_model}_benchmark.pth"

                if isinstance(config["model"][selected_model]["save_path"], dict):
                    config["model"][selected_model]["save_path"]["proposed"] = (
                        os.path.join(config["log_dir"], proposed_filename)
                    )
                    if "benchmark" in config["model"][selected_model]["save_path"]:
                        config["model"][selected_model]["save_path"]["benchmark"] = (
                            os.path.join(config["log_dir"], benchmark_filename)
                        )
                else:
                    config["model"][selected_model]["save_path"] = os.path.join(
                        config["log_dir"], proposed_filename
                    )

            # 텐서보드 로그 디렉토리 설정
            if (
                "training" in config["model"]
                and "tensorboard" in config["model"]["training"]
            ):
                if config["model"]["training"]["tensorboard"].get("enabled", False):
                    config["model"]["training"]["tensorboard"]["log_dir"] = (
                        os.path.join(config["log_dir"], "tensorboard")
                    )
        else:
            raise ValueError(
                f"지원되지 않는 모델 타입입니다: {selected_model}. 'hbt' 또는 'tft'를 선택하세요."
            )

    return config
