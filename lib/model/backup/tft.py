import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    """
    시간 단계에 걸쳐 모듈을 적용하는 래퍼 클래스.

    입력 텐서의 각 시간 단계에 대해 동일한 모듈을 적용합니다.
    """

    def __init__(self, module: nn.Module):
        """
        Args:
            module (nn.Module): 각 시간 단계에 적용할 PyTorch 모듈
        """
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        각 시간 단계에 모듈을 적용합니다.

        Args:
            x (torch.Tensor): 입력 텐서, 형태 [batch_size, time_steps, ...]

        Returns:
            torch.Tensor: 각 시간 단계에 모듈이 적용된 출력 텐서, 형태 [batch_size, time_steps, ...]
        """
        # 입력 형태: [batch, time_steps, ...]
        # 배치와 시간 단계를 합침
        batch_size, time_steps = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(-1, *x.size()[2:])

        # 모듈 적용
        y = self.module(x_reshaped)

        # 원래 형태로 복원
        return y.contiguous().view(batch_size, time_steps, *y.size()[1:])


class GatedResidualNetwork(nn.Module):
    """
    변수 선택 및 처리를 위한 게이트 잔차 네트워크.

    입력을 변환하고 게이트 메커니즘을 통해 잔차 연결을 적용합니다.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ELU(),
        context_size: int = None,
    ):
        """
        Args:
            input_size (int): 입력 특성의 차원
            hidden_size (int): 은닉층의 차원
            output_size (int): 출력 특성의 차원
            dropout (float): 드롭아웃 비율
            activation (nn.Module): 활성화 함수
            context_size (int, optional): 컨텍스트 특성의 차원 (없으면 None)
        """
        super(GatedResidualNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size

        # 주요 네트워크 레이어
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = activation
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        # 입력과 출력 차원이 다를 경우 스킵 연결
        self.add_skip_connection = input_size != output_size
        if self.add_skip_connection:
            self.skip_layer = nn.Linear(input_size, output_size)

        # 컨텍스트 통합 (제공된 경우)
        self.context_integration = context_size is not None
        if self.context_integration:
            self.context_layer = nn.Linear(context_size, hidden_size)

        # 게이트 메커니즘
        self.gate_layer = nn.Linear(input_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        네트워크를 통해 입력을 전달합니다.

        Args:
            x (torch.Tensor): 입력 텐서
            context (torch.Tensor, optional): 컨텍스트 텐서 (없으면 None)

        Returns:
            torch.Tensor: 게이트 잔차 네트워크의 출력
        """
        # 주요 네트워크
        residual = x

        # 첫 번째 변환 적용
        hidden = self.fc1(x)

        # 컨텍스트 통합 (제공된 경우)
        if self.context_integration and context is not None:
            context_hidden = self.context_layer(context)
            hidden = hidden + context_hidden

        hidden = self.elu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        # 필요한 경우 스킵 연결 적용
        if self.add_skip_connection:
            skip_connection = self.skip_layer(x)
        else:
            skip_connection = x

        # 게이트 메커니즘
        gate = torch.sigmoid(self.gate_layer(x))

        # 잔차 연결과 결합
        output = gate * hidden + (1 - gate) * skip_connection

        # 레이어 정규화
        return self.layer_norm(output)


class VariableSelectionNetwork(nn.Module):
    """
    관련 입력 변수를 선택하기 위한 네트워크.

    여러 입력 변수에 가중치를 할당하고 이를 결합합니다.
    """

    def __init__(
        self,
        input_sizes: list,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
    ):
        """
        Args:
            input_sizes (list): 각 입력 변수의 차원 리스트
            hidden_size (int): 은닉층의 차원
            output_size (int): 출력 특성의 차원
            dropout (float): 드롭아웃 비율
            context_size (int, optional): 컨텍스트 특성의 차원 (없으면 None)
        """
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)

        # 각 입력 변수에 대한 GRN
        self.grns = nn.ModuleList()
        for input_size in input_sizes:
            self.grns.append(
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    dropout=dropout,
                )
            )

        # 가중치 계산을 위한 GRN
        self.weights_grn = GatedResidualNetwork(
            input_size=sum(input_sizes),
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout=dropout,
            context_size=context_size,
        )

    def forward(self, inputs: list, context: torch.Tensor = None) -> tuple:
        """
        변수 선택 네트워크를 통해 입력을 처리합니다.

        Args:
            inputs (list): 입력 텐서 리스트
            context (torch.Tensor, optional): 컨텍스트 텐서 (없으면 None)

        Returns:
            tuple: (선택된 변수 출력, 변수 가중치)
        """
        # 각 변수를 자체 GRN으로 처리
        transformed_inputs = []
        for i, inp in enumerate(inputs):
            transformed_inputs.append(self.grns[i](inp))

        # 변수 선택을 위한 가중치 계산
        flat_inputs = torch.cat(inputs, dim=-1)
        weights = self.weights_grn(flat_inputs, context)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)

        # 변수 출력에 가중치를 부여하고 결합
        combined_inputs = []
        for i, transformed_input in enumerate(transformed_inputs):
            combined_inputs.append(transformed_input * weights[:, i])

        var_selected_inputs = torch.sum(torch.stack(combined_inputs, dim=-2), dim=-2)

        return var_selected_inputs, weights


class TemporalFusionTransformer(nn.Module):
    """
    시장 행동 예측을 위한 다중 에이전트 시간적 융합 트랜스포머.

    OHLCV 데이터와 에이전트 행동을 처리하여 다음 행동을 예측합니다.
    """

    def __init__(
        self,
        ohlcv_features: int = 10,
        agent_actions: int = 40,  # 20 롱 빈 + 20 숏 빈
        other_actions: int = 40,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        lstm_layers: int = 1,
        context_length: int = 5,
    ):
        """
        Args:
            ohlcv_features (int): OHLCV 특성의 수
            agent_actions (int): 에이전트 행동 빈의 수
            other_actions (int): 다른 플레이어 행동 빈의 수
            hidden_size (int): 은닉층의 차원
            num_heads (int): 어텐션 헤드의 수
            dropout (float): 드롭아웃 비율
            num_encoder_layers (int): 인코더 레이어의 수
            lstm_layers (int): LSTM 레이어의 수
            context_length (int): 고려할 과거 시간 단계의 수
        """
        super(TemporalFusionTransformer, self).__init__()

        self.hidden_size = hidden_size
        self.context_length = context_length

        # 입력 특성 크기 정의
        self.ohlcv_features = ohlcv_features
        self.agent_actions = agent_actions
        self.other_actions = other_actions

        # 정적 메타데이터 인코더
        self.static_grn = GatedResidualNetwork(
            input_size=1,  # 정적 특성을 위한 플레이스홀더
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 다양한 특성 그룹에 대한 변수 선택 네트워크 정의

        # 1. OHLCV 변수 선택 (o_candle에 해당)
        self.ohlcv_vsn = VariableSelectionNetwork(
            input_sizes=[1] * ohlcv_features,  # 각 특성은 개별적으로 처리됨
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 2. 에이전트 자신의 행동 변수 선택 (a_i에 해당)
        self.agent_actions_vsn = VariableSelectionNetwork(
            input_sizes=[1] * agent_actions,  # 각 행동 빈은 개별적으로 처리됨
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 3. 다른 플레이어의 행동 변수 선택 (a_{-i}에 해당)
        self.other_actions_vsn = VariableSelectionNetwork(
            input_sizes=[1] * other_actions,  # 각 행동 빈은 개별적으로 처리됨
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 특성 임베딩
        self.ohlcv_embed = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
        )

        self.agent_actions_embed = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
        )

        self.other_actions_embed = TimeDistributed(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            )
        )

        # 시퀀스 인코딩을 위한 LSTM 레이어 (변수 간 공유)
        self.lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # 신념 임베딩 함수

        # f_i: OHLCV와 자신의 과거 행동을 기반으로 한 환경에 대한 에이전트의 신념
        self.f_i = GatedResidualNetwork(
            input_size=hidden_size * 2,  # OHLCV와 에이전트의 행동 결합
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # g_i: 다른 사람들의 행동을 기반으로 한 다른 사람들의 신념에 대한 에이전트의 신념
        self.g_i = GatedResidualNetwork(
            input_size=hidden_size,  # 다른 사람들의 행동 기반
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # 시간적 의존성을 위한 셀프 어텐션 인코더
        self.self_attn_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

        # 중첩된 신념 모델링을 위한 크로스 어텐션
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # h_i: 다음 행동 예측 함수
        self.h_i = nn.Sequential(
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
            ),
            nn.Linear(hidden_size, agent_actions),  # 행동 빈에 대한 출력 분포
        )

        # 최종 출력을 위한 레이어 정규화
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x_ohlcv: torch.Tensor,
        x_agent_actions: torch.Tensor,
        x_other_actions: torch.Tensor,
    ) -> tuple:
        """
        TFT 모델의 순방향 패스.

        Args:
            x_ohlcv (torch.Tensor): OHLCV 데이터, 형태 [batch_size, time_steps, ohlcv_features]
            x_agent_actions (torch.Tensor): 에이전트의 과거 행동, 형태 [batch_size, time_steps, agent_actions]
            x_other_actions (torch.Tensor): 다른 플레이어의 행동, 형태 [batch_size, time_steps, other_actions]

        Returns:
            tuple: (다음 단계 행동 분포, 해석을 위한 어텐션 가중치 딕셔너리)
        """
        batch_size, time_steps, _ = x_ohlcv.shape

        # 각 특성 그룹을 변수 선택 네트워크를 통해 처리
        ohlcv_inputs = [x_ohlcv[:, :, i : i + 1] for i in range(self.ohlcv_features)]
        agent_action_inputs = [
            x_agent_actions[:, :, i : i + 1] for i in range(self.agent_actions)
        ]
        other_action_inputs = [
            x_other_actions[:, :, i : i + 1] for i in range(self.other_actions)
        ]

        # 변수 선택 네트워크 적용
        ohlcv_selected, ohlcv_weights = self.ohlcv_vsn(ohlcv_inputs)
        agent_actions_selected, agent_actions_weights = self.agent_actions_vsn(
            agent_action_inputs
        )
        other_actions_selected, other_actions_weights = self.other_actions_vsn(
            other_action_inputs
        )

        # 특성 변환 적용
        ohlcv_transformed = self.ohlcv_embed(ohlcv_selected)
        agent_actions_transformed = self.agent_actions_embed(agent_actions_selected)
        other_actions_transformed = self.other_actions_embed(other_actions_selected)

        # LSTM을 통한 처리 (시간적 특성 추출)
        ohlcv_encoded, _ = self.lstm_layer(ohlcv_transformed)
        agent_actions_encoded, _ = self.lstm_layer(agent_actions_transformed)
        other_actions_encoded, _ = self.lstm_layer(other_actions_transformed)

        # 신념 임베딩 계산

        # f_i: 환경에 대한 에이전트의 신념
        env_beliefs = self.f_i(
            torch.cat([ohlcv_encoded, agent_actions_encoded], dim=-1)
        )

        # g_i: 다른 사람들의 신념에 대한 에이전트의 신념
        others_beliefs = self.g_i(other_actions_encoded)

        # 신념의 시간적 의존성을 위한 셀프 어텐션
        env_beliefs_attn = self.self_attn_encoder(env_beliefs)
        others_beliefs_attn = self.self_attn_encoder(others_beliefs)

        # 중첩된 신념 모델링을 위한 크로스 어텐션
        nested_beliefs, cross_attn_weights = self.cross_attn(
            query=env_beliefs_attn, key=others_beliefs_attn, value=others_beliefs_attn
        )

        # 레이어 정규화를 통한 신념 결합
        combined_beliefs = self.layer_norm(env_beliefs_attn + nested_beliefs)

        # h_i를 사용하여 다음 행동 예측
        next_actions = self.h_i(combined_beliefs)

        # 다음 시간 단계만 예측하는 데 관심이 있음
        next_step_actions = next_actions[:, -1, :]

        # 해석을 위한 어텐션 가중치 수집
        attention_weights = {
            "ohlcv_weights": ohlcv_weights.detach(),
            "agent_actions_weights": agent_actions_weights.detach(),
            "other_actions_weights": other_actions_weights.detach(),
            "cross_attention": cross_attn_weights.detach(),
        }

        return next_step_actions, attention_weights


class TFTModelTrainer:
    """
    시간적 융합 트랜스포머 모델을 위한 트레이너 클래스.

    모델 훈련, 평가 및 저장 기능을 제공합니다.
    """

    def __init__(
        self,
        model: TemporalFusionTransformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """
        Args:
            model (TemporalFusionTransformer): 훈련할 TFT 모델
            learning_rate (float): 학습률
            weight_decay (float): 가중치 감쇠 계수
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # 분포 예측을 위한 KL 발산 손실 사용
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train_step(
        self,
        x_ohlcv: torch.Tensor,
        x_agent_actions: torch.Tensor,
        x_other_actions: torch.Tensor,
        y_agent_actions: torch.Tensor,
    ) -> float:
        """
        단일 훈련 단계를 수행합니다.

        Args:
            x_ohlcv (torch.Tensor): OHLCV 입력 데이터
            x_agent_actions (torch.Tensor): 에이전트 행동 입력 데이터
            x_other_actions (torch.Tensor): 다른 플레이어 행동 입력 데이터
            y_agent_actions (torch.Tensor): 목표 에이전트 행동 분포

        Returns:
            float: 훈련 손실 값
        """
        self.optimizer.zero_grad()

        # 순방향 패스
        predictions, _ = self.model(x_ohlcv, x_agent_actions, x_other_actions)

        # KL 발산을 위한 로그 소프트맥스 적용
        log_pred = F.log_softmax(predictions, dim=-1)

        # 손실 계산 (예측 분포와 실제 분포 간의 KL 발산)
        loss = self.criterion(log_pred, y_agent_actions)

        # 역방향 패스
        loss.backward()

        # 폭발하는 그래디언트 방지를 위한 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 파라미터 업데이트
        self.optimizer.step()

        return loss.item()

    def evaluate(
        self,
        x_ohlcv: torch.Tensor,
        x_agent_actions: torch.Tensor,
        x_other_actions: torch.Tensor,
        y_agent_actions: torch.Tensor,
    ) -> tuple:
        """
        모델을 평가합니다.

        Args:
            x_ohlcv (torch.Tensor): OHLCV 입력 데이터
            x_agent_actions (torch.Tensor): 에이전트 행동 입력 데이터
            x_other_actions (torch.Tensor): 다른 플레이어 행동 입력 데이터
            y_agent_actions (torch.Tensor): 목표 에이전트 행동 분포

        Returns:
            tuple: (평가 손실 값, 어텐션 가중치)
        """
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(
                x_ohlcv, x_agent_actions, x_other_actions
            )
            log_pred = F.log_softmax(predictions, dim=-1)
            loss = self.criterion(log_pred, y_agent_actions)

            # 여기에 추가 메트릭을 추가할 수 있음

        self.model.train()
        return loss.item(), attention_weights

    def save_model(self, path: str) -> None:
        """
        모델을 저장합니다.

        Args:
            path (str): 모델을 저장할 경로
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str) -> None:
        """
        저장된 모델을 로드합니다.

        Args:
            path (str): 로드할 모델 파일 경로
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# 사용 예시:
if __name__ == "__main__":
    # 모델 파라미터 정의
    ohlcv_features = 10  # OHLCV 특성
    agent_actions = 40  # 20 롱 + 20 숏 빈
    other_actions = 40  # 20 롱 + 20 숏 빈
    hidden_size = 128
    context_length = 5  # 고려할 과거 시간 단계의 수

    # 모델 초기화
    model = TemporalFusionTransformer(
        ohlcv_features=ohlcv_features,
        agent_actions=agent_actions,
        other_actions=other_actions,
        hidden_size=hidden_size,
        context_length=context_length,
    )

    # 트레이너 초기화
    trainer = TFTModelTrainer(model)

    # 예시 데이터 차원 (batch_size, time_steps, features)
    batch_size = 64
    time_steps = 10

    # 시연을 위한 랜덤 데이터 생성
    x_ohlcv = torch.rand(batch_size, time_steps, ohlcv_features)
    x_agent_actions = torch.rand(batch_size, time_steps, agent_actions)
    x_other_actions = torch.rand(batch_size, time_steps, other_actions)

    # 목표: 다음 단계 행동 분포 (KL 발산을 위해 합이 1이어야 함)
    y_agent_actions = torch.rand(batch_size, agent_actions)
    y_agent_actions = y_agent_actions / y_agent_actions.sum(dim=1, keepdim=True)

    # 예시 훈련 루프
    for epoch in range(5):
        loss = trainer.train_step(
            x_ohlcv, x_agent_actions, x_other_actions, y_agent_actions
        )
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # 평가
    eval_loss, attention_weights = trainer.evaluate(
        x_ohlcv, x_agent_actions, x_other_actions, y_agent_actions
    )
    print(f"Evaluation loss: {eval_loss:.4f}")

    # 모델 저장
    trainer.save_model("tft_model.pth")
