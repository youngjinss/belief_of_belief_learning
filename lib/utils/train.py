import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# ... existing code ...
def train_model(trainer, train_loader, val_loader, config=None, logger=None):
    """
    Train the model with early stopping.

    Args:
        trainer: Trainer
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: 모델 설정 파일 경로
        logger: 로깅을 위한 로거 객체

    Returns:
        model: Trained model
        history: Training history
    """
    if config is None:
        raise ValueError("config is None")

    # 로거가 없으면 기본 print 사용
    log_info = logger.info if logger else print

    # 설정에서 매개변수 추출
    num_epochs = config["model"]["training"]["num_epochs"]
    patience = config["model"]["training"]["patience"]
    device = config["model"]["training"]["device"]
    save_path = config["model"][config["model"]["select"]]["save_path"]

    # 텐서보드 설정
    tensorboard_enabled = (
        config["model"]["training"].get("tensorboard", {}).get("enabled", False)
    )
    tensorboard_log_dir = (
        config["model"]["training"].get("tensorboard", {}).get("log_dir", "runs")
    )

    # 텐서보드 로거 설정
    if tensorboard_enabled:
        log_dir = os.path.join(
            tensorboard_log_dir,
            f'{config["model"]["select"]}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        )
        writer = SummaryWriter(log_dir=log_dir)
        log_info(f"텐서보드 로그: {log_dir}")

    # 모델을 장치에 전송
    trainer.model.to(device)
    log_info(f"[PROCESS] 모델을 {device} 장치로 이동했습니다")

    best_val_loss = float("inf")
    counter = 0
    history = {"train_loss": [], "val_loss": []}

    log_info(f"[PROCESS] 총 {num_epochs}개 에폭 학습을 시작합니다")

    for epoch in range(num_epochs):
        # Training
        trainer.model.train()
        train_losses = []

        for batch in train_loader:
            x_ohlcv = batch["ohlcv"].to(device)
            x_self_actions = batch["self_actions"].to(device)
            x_other_actions = batch["other_actions"].to(device)
            y_target = batch["target"].to(device)

            loss = trainer.train_step(
                x_ohlcv, x_self_actions, x_other_actions, y_target
            )
            train_losses.append(loss)

        avg_train_loss = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg_train_loss)

        # Validation
        trainer.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x_ohlcv = batch["ohlcv"].to(device)
                x_self_actions = batch["self_actions"].to(device)
                x_other_actions = batch["other_actions"].to(device)
                y_target = batch["target"].to(device)

                val_loss, _ = trainer.evaluate(
                    x_ohlcv, x_self_actions, x_other_actions, y_target
                )
                val_losses.append(val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        history["val_loss"].append(avg_val_loss)

        # 텐서보드에 로깅
        if tensorboard_enabled:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)

            # 모델 파라미터의 히스토그램 추가 (선택적)
            for name, param in trainer.model.named_parameters():
                writer.add_histogram(f"Parameters/{name}", param.data, epoch)

        # 에피소드 진행 로깅
        log_info(
            f"[PROGRESS] 에폭 {epoch+1}/{num_epochs}: 학습 손실: {avg_train_loss:.4f}, 검증 손실: {avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model
            trainer.save_model(save_path)
            log_info(
                f"[RESULT] 새로운 최적 검증 손실: {best_val_loss:.4f}, 모델 저장: {save_path}"
            )
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                log_info(
                    f"[PROCESS] 조기 종료: 에폭 {epoch+1}에서 {patience}회 연속 검증 손실 미개선"
                )
                break

    # 텐서보드 종료
    if tensorboard_enabled:
        writer.close()

    # Load the best model
    trainer.load_model(save_path)
    log_info(f"[RESULT] 학습 완료. 최종 최적 검증 손실: {best_val_loss:.4f}")
    return trainer, history
