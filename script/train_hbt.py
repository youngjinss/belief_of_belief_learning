#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset

# 프로젝트 디렉토리 경로 설정
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)
# lib 폴더 경로 추가
sys.path.insert(0, project_dir)

# 설정 로드
from lib.utils.confing import load_config
from lib.utils.train import train_model
from lib.utils.data import prepare_dataloaders

# 로거 설정
logger = logging.getLogger("train_hbt")
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 포맷 설정
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(console_handler)


def create_directory(directory_path):
    """디렉토리가 존재하지 않으면 생성하는 함수"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"[PROCESS] 디렉토리 생성됨: {directory_path}")
    else:
        logger.info(f"[PROCESS] 디렉토리가 이미 존재함: {directory_path}")


def load_data(retail_path, institutional_path):
    """소매 및 기관 투자자 데이터를 로드하는 함수"""
    # 데이터 로드
    logger.info(f"[PROCESS] 소매 투자자 데이터 로드: {retail_path}")
    retail_df = pd.read_csv(retail_path, index_col=0)

    logger.info(f"[PROCESS] 기관 투자자 데이터 로드: {institutional_path}")
    institutional_df = pd.read_csv(institutional_path, index_col=0)

    # 데이터 기본 정보 확인
    logger.info(f"[RESULT] 소매 투자자 데이터 크기: {retail_df.shape}")
    logger.info(f"[RESULT] 기관 투자자 데이터 크기: {institutional_df.shape}")

    # 타임스탬프 변환 (필요한 경우)
    if "timestamp" in retail_df.columns:
        retail_df["timestamp"] = pd.to_datetime(retail_df["timestamp"])
        retail_df.set_index("timestamp", inplace=True)

    if "timestamp" in institutional_df.columns:
        institutional_df["timestamp"] = pd.to_datetime(institutional_df["timestamp"])
        institutional_df.set_index("timestamp", inplace=True)

    logger.info("[RESULT] 데이터 로드 완료")

    return retail_df, institutional_df


def train_and_save_model(
    model_type,
    train_loader,
    val_loader,
    ohlcv_dim,
    action_dim,
    hidden_dim,
    context_length,
    learning_rate,
    config,
):
    """모델 생성, 학습 및 저장 함수"""
    logger.info(f"[PROCESS] {model_type} 모델 생성 중...")

    if model_type == "proposed":
        from lib.model.hbt import create_proposed_model, HierarchicalModelTrainer

        model = create_proposed_model(
            ohlcv_dim=ohlcv_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )
        trainer = HierarchicalModelTrainer(model=model, lr=learning_rate)
    elif model_type == "benchmark":
        from lib.model.hbt import create_benchmark_model, HierarchicalModelTrainer

        model = create_benchmark_model(
            ohlcv_dim=ohlcv_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
        )
        trainer = HierarchicalModelTrainer(model=model, lr=learning_rate)
    else:
        logger.error(f"[ERROR] 지원되지 않는 모델 유형: {model_type}")
        return None

    # 모델 학습
    logger.info(f"[PROCESS] {model_type} 모델 학습 시작...")

    trainer, history = train_model(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,  # 로거 전달
    )

    # 모델 저장
    save_path = config["model"][config["model"]["select"]]["save_path"]
    trainer.save_model(save_path)
    logger.info(
        f"[RESULT] 모델이 성공적으로 학습되었으며 {save_path}에 저장되었습니다."
    )

    return trainer, history


def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description="계층적 신념 트랜스포머(HBT) 학습 스크립트"
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="프로젝트 디렉토리 경로",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/train.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["proposed", "benchmark", "both"],
        default="proposed",
        help="학습할 모델 유형",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="로그 파일 저장 디렉토리 경로",
    )
    args = parser.parse_args()

    # 프로젝트 디렉토리 경로 설정
    project_dir = args.project_dir
    os.chdir(project_dir)

    # lib 폴더 경로 추가
    sys.path.append(project_dir + "/lib")

    logger.info("[PROCESS] HBT 모델 학습 시작")
    logger.info(f"[PROCESS] 프로젝트 디렉토리: {project_dir}")

    config_path = os.path.join(project_dir, args.config_path)
    logger.info(f"[PROCESS] 설정 파일 로드: {config_path}")
    config = load_config("train")

    # 로그 디렉토리 생성
    if args.log_dir is None:
        raise ValueError("log_dir 인자가 제공되지 않았습니다.")
    else:
        log_dir = args.log_dir
        create_directory(log_dir)

    # 파일 로그 핸들러 추가
    log_file = os.path.join(log_dir, "result.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"[PROCESS] 로그 디렉토리: {log_dir}")
    logger.info(f"[PROCESS] 로그 파일: {log_file}")

    # tensorboard 디렉토리를 log_dir 밑에 생성
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    create_directory(tensorboard_dir)
    config["model"]["training"]["tensorboard"]["log_dir"] = tensorboard_dir
    tensorboard_port = config["model"]["training"]["tensorboard"]["port"]
    logger.info(f"[PROCESS] 텐서보드 활성화됨 - 로그 디렉토리: {tensorboard_dir}")
    logger.info(
        f"[HELP] 학습 진행 상황을 보려면 다음 명령어 실행: tensorboard --logdir={tensorboard_dir} --port={tensorboard_port}"
    )

    # 모델 저장 디렉토리를 log_dir 밑에 생성
    selected_model = config["model"]["select"]
    config["model"][selected_model]["save_path"]["proposed"] = os.path.join(
        log_dir, f"best_{selected_model}_model.pth"
    )
    config["model"][selected_model]["save_path"]["benchmark"] = os.path.join(
        log_dir, f"best_{selected_model}_benchmark.pth"
    )
    # 모델 저장 디렉토리 업데이트 알림
    logger.info(
        f"[PROCESS] proposed 모델 저장 경로 업데이트: {config['model'][selected_model]['save_path']['proposed']}"
    )
    logger.info(
        f"[PROCESS] benchmark 모델 저장 경로 업데이트: {config['model'][selected_model]['save_path']['benchmark']}"
    )

    # 소매 및 기관 투자자 데이터 경로
    retail_path = os.path.join(
        project_dir, "data/binance/futures/um/dataset/retail_train.csv"
    )
    institutional_path = os.path.join(
        project_dir, "data/binance/futures/um/dataset/institutional_train.csv"
    )

    # 데이터 로드
    retail_df, institutional_df = load_data(retail_path, institutional_path)

    # 데이터 소스 딕셔너리 생성
    data_sources = {"retail": retail_df, "institutional": institutional_df}

    # 데이터로더 준비

    logger.info("[PROCESS] 데이터로더 준비 중...")
    loaders = prepare_dataloaders(data_sources, config)

    # 학습 데이터셋 통합
    combined_train_dataset = ConcatDataset(
        [loaders["retail"]["train"].dataset, loaders["institutional"]["train"].dataset]
    )
    combined_train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config["model"]["training"]["batch_size"],
        shuffle=True,
    )

    combined_val_dataset = ConcatDataset(
        [loaders["retail"]["val"].dataset, loaders["institutional"]["val"].dataset]
    )
    combined_val_loader = DataLoader(
        combined_val_dataset,
        batch_size=config["model"]["training"]["batch_size"],
        shuffle=True,
    )

    # 모델 파라미터 추출
    model_config = config["model"]["hbt"]
    ohlcv_dim = config["dataloader"]["ohlcv_dim"]
    action_dim = model_config["action_dim"]
    hidden_dim = model_config["hidden_dim"]
    context_length = model_config["context_length"]
    learning_rate = config["model"]["training"]["learning_rate"]

    # 선택된 모델 유형에 따라 모델 학습
    if args.model_type == "proposed" or args.model_type == "both":
        proposed_trainer, proposed_history = train_and_save_model(
            model_type="proposed",
            train_loader=combined_train_loader,
            val_loader=combined_val_loader,
            ohlcv_dim=ohlcv_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
            learning_rate=learning_rate,
            config=config,
        )

    if args.model_type == "benchmark" or args.model_type == "both":
        benchmark_trainer, benchmark_history = train_and_save_model(
            model_type="benchmark",
            train_loader=combined_train_loader,
            val_loader=combined_val_loader,
            ohlcv_dim=ohlcv_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            context_length=context_length,
            learning_rate=learning_rate,
            config=config,
        )

    logger.info("[RESULT] 모든 모델 학습 완료")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"총 처리 시간: {end_time - start_time}")
