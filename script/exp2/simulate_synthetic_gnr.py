import os
import json
from datetime import datetime
from typing import Dict, Tuple, Protocol, Optional
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
from scipy.special import softmax

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LEVEL = 2
N_SAMPLES = 15
DEFAULT_BETA = 2.0
DEFAULT_N_ITEMS = 2
DEFAULT_MAX_VALUE = 10.0
DEFAULT_N_PREFERENCE_POINTS = 101
DEFAULT_N_PRICE_POINTS = 81
EPSILON = 1e-10
EXPERIMENT_TIME = datetime.now().strftime("%Y%m%d_%H%M")
RESULT_DIR = f"./results/exp2/{EXPERIMENT_TIME}/"
PLOT_DIR = f"{RESULT_DIR}/analysis_plot.png"
DATA_DIR = f"{RESULT_DIR}/simulation_results.json"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class BuyerAgentProtocol(Protocol):
    """
    구매자 에이전트 프로토콜: 1단계에서 P^{b,t=1}_{k=l}(a_1 | **r**, **d**)를 구현
    사용 용도: 판매자가 베이지안 추론 시 구매자 모델로 활용
    결과물: 주어진 선호도와 거리에 대한 아이템 선택 확률 분포
    """

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray:
        """선호도 **r**와 거리 **d**에 대한 1단계 정책 P^{b,t=1}(a_1|**r**,**d**) 반환"""
        ...


class SellerAgentProtocol(Protocol):
    """
    판매자 에이전트 프로토콜: 베이지안 추론과 가격 최적화를 구현
    사용 용도: 구매자의 2단계 시뮬레이션에서 판매자 행동 예측
    결과물: 구매자 행동 관측 후 추론된 선호도 분포와 최적 가격
    """

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """베이지안 추론: p^s(**r** | a_1) \propto P^b(a_1 | **r**, **d**) * p(**r**)"""
        ...

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """최적 가격 **m***_{k=l} 계산: E[U^{t=3}_s(**m**)] 최대화"""
        ...


class NotationUtils:
    """
    수학적 표기법을 따르는 유틸리티 함수들
    사용 용도: 벡터 정규화, 격자 생성, 소프트맥스 정책 계산 등 공통 연산
    결과물: 정규화된 벡터, 균등 분포, 소프트맥스 확률 분포
    """

    @staticmethod
    def normalize_vector_to_sum(
        values: np.ndarray, target_sum: float = DEFAULT_MAX_VALUE
    ) -> np.ndarray:
        """
        제약 조건 적용: sum(**v**) = target_sum
        목적: 선호도나 가격 벡터가 총합 제약을 만족하도록 정규화
        """
        return values * target_sum / values.sum()

    @staticmethod
    def create_preference_grid_r(
        n_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> np.ndarray:
        """
        선호도 grid 생성: 베이지안 추론용 **r** 후보값들
        계산: np.linspace(0, DEFAULT_MAX_VALUE, n_points)로 균등 분할
        """
        return np.linspace(0, DEFAULT_MAX_VALUE, n_points)

    @staticmethod
    def create_uniform_prior_p_r(size: int) -> np.ndarray:
        """
        균등 사전 분포 p(**r**) 생성
        계산: 1/size로 균등 확률 할당
        """
        return np.ones(size) / size

    @staticmethod
    def compute_softmax_policy_P(
        q_values: np.ndarray, beta: float = DEFAULT_BETA
    ) -> np.ndarray:
        """
        소프트맥스 정책 계산: P^{agent,t}(a | Q)
        계산: sigma(beta * (Q(a_0) - Q(a_1))) for 2-item case
        또는: softmax(beta * Q) for general case
        """
        q_values = np.array(q_values)
        if len(q_values) == 2:
            diff = beta * (q_values[0] - q_values[1])
            prob_0 = 1 / (1 + np.exp(-diff))
            return np.array([prob_0, 1 - prob_0])
        else:
            return softmax(beta * q_values)


class BayesianIRLMixin:
    """
    베이지안 역강화학습 기능: p^s(**r** | a_1) 구현
    사용 용도: 판매자가 구매자의 1단계 행동을 보고 선호도 추론
    결과물: 사후 확률 분포 p^s(**r** | a_1)
    """

    def p_r_given_a_impl(
        self,
        observed_action_a1: int,
        vector_d: np.ndarray,
        buyer_model: BuyerAgentProtocol,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        베이지안 추론 구현: p^s(**r** | a_1) ∝ P^b(a_1 | **r**, **d**) * p(**r**)
        """
        preference_grid_r = NotationUtils.create_preference_grid_r(n_preference_points)  # 선호도 grid r_apple \in [0, 10] 생성
        posterior_p_r_given_a = np.zeros_like(preference_grid_r)  # 사후 확률 분포 초기화
        prior_p_r = NotationUtils.create_uniform_prior_p_r(len(preference_grid_r))  # 균등 사전 분포 생성

        for i, r_apple in enumerate(preference_grid_r):  # 각 선호도 값에 대해
            r_orange = DEFAULT_MAX_VALUE - r_apple  # 상대 아이템의 선호도 계산 (r_orange = 10 - r_apple로 설정)
            vector_r = np.array([r_apple, r_orange])  # 선호도 벡터 생성

            # 구매자 모델에서 P^b(a_1 | **r**, **d**) 계산
            P_b_t1 = buyer_model.get_P_t1(vector_d, vector_r)
            likelihood_P_b_a1 = P_b_t1[observed_action_a1]
            posterior_p_r_given_a[i] = likelihood_P_b_a1 * prior_p_r[i]  # 우도 × 사전분포로 비정규화 사후분포 계산

        # 정규화하여 p^s(**r** | a_1) 반환
        posterior_p_r_given_a = posterior_p_r_given_a / np.sum(posterior_p_r_given_a)
        return preference_grid_r, posterior_p_r_given_a


class PriceOptimizationMixin:
    """
    가격 최적화 기능: **m***_{k=l} 계산
    사용 용도: 판매자가 추론된 선호도 분포를 바탕으로 수익 최대화 가격 설정
    결과물: 최적 가격 벡터 **m***
    """

    def optimize_vector_m_star(
        self,
        preference_grid_r: np.ndarray,
        posterior_p_r_given_a: np.ndarray,
        beta: float = DEFAULT_BETA,
        n_price_points: int = DEFAULT_N_PRICE_POINTS,
    ) -> np.ndarray:
        """
        최적 가격 **m*** 계산: E[U^{t=3}_s(**m**)] 최대화
        제약조건: sum(**m**) = DEFAULT_MAX_VALUE
        """
        best_vector_m = np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])  # 초기 가격 설정 (m_apple = m_orange = 5)
        best_expected_U_s = 0  # 최대 기대 수익 초기화

        for m_apple in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):
            m_orange = DEFAULT_MAX_VALUE - m_apple  # m_orange = 10 - m_apple
            test_vector_m = np.array([m_apple, m_orange])

            expected_U_s = self._calculate_E_U_s_t3(  # 각 가격에 대해 기대 수익 E[U^{t=3}_s] 계산
                preference_grid_r, posterior_p_r_given_a, test_vector_m, beta
            )

            if expected_U_s > best_expected_U_s:  # 최대 기대 수익을 갖는 가격 선택
                best_expected_U_s = expected_U_s
                best_vector_m = test_vector_m

        return best_vector_m

    def _calculate_E_U_s_t3(
        self,
        preference_grid_r: np.ndarray,
        posterior_p_r_given_a: np.ndarray,
        vector_m: np.ndarray,
        beta: float,
    ) -> float:
        """
        기대 판매자 수익 계산: E[U^{t=3}_s(**m**)]
        """
        expected_U_s = 0
        for i, r_apple in enumerate(preference_grid_r):
            r_orange = DEFAULT_MAX_VALUE - r_apple
            vector_r = np.array([r_apple, r_orange])

            # Q^{b,t=3}(i) = r(i) - m(i)
            q_values_b_t3 = vector_r - vector_m
            
            # P^b(a_3=i | **m**, **r**) = sigma(beta * (Q^{b,t=3}(i) - Q^{b,t=3}(j)))
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_b_t3, beta)

            # E[U^{t=3}_s]
            U_s_t3 = np.sum(P_b_t3 * vector_m)  # \sum_i P^b(a_3=i | **m**, **r**) * m(i)
            expected_U_s += posterior_p_r_given_a[i] * U_s_t3  # \sum_r p^s(r|a_1) * \sum_i P^b(a_3=i | **m**, **r**) * m(i)

        return expected_U_s


class StrategicPlanningMixin:
    """
    전략적 계획 기능: Q^{b,t=1}_{k=l} 계산
    사용 용도: level-l 구매자가 판매자의 반응을 예측하여 1단계 행동 최적화
    결과물: 전략적 Q값 (즉석 보상 + 미래 기대 보상)
    """

    def compute_Q_b_t1_kl(
        self,
        vector_d: np.ndarray,
        vector_r: np.ndarray,
        seller_model: SellerAgentProtocol,
        beta: float = DEFAULT_BETA,
    ) -> np.ndarray:
        """
        level-l 구매자의 전략적 Q값 계산
        
        공식: Q^{b,t=1}_{k=l}(i_1) = U^{t=1}_b(i_1) + E[U^{t=3}_b(**m***_{k=(l-1)})]
        """
        q_values_Q_b_t1 = np.zeros(DEFAULT_N_ITEMS)

        for action_a1 in range(DEFAULT_N_ITEMS):  # 각 1단계 행동 a_1에 대해:
            # 현재 보상 U^{t=1}_b = r(a_1) - d(a_1)
            immediate_U_b_t1 = vector_r[action_a1] - vector_d[action_a1]

            # 판매자 시뮬레이션
            preference_grid_r, posterior_p_r_given_a = seller_model.p_r_given_a(
                action_a1, vector_d
            )  # p^s(**r** | a_1)
            vector_m_star = seller_model.compute_m_star(
                preference_grid_r, posterior_p_r_given_a
            )  # **m***

            # 미래 기대 보상: E[U^{t=3}_b] = \sum_i  * ()
            q_values_stage3 = vector_r - vector_m_star  # r(i) - m***(i)
            P_b_t3 = NotationUtils.compute_softmax_policy_P(q_values_stage3, beta)  # P^b(a_3=i | **m***)
            expected_U_b_t3 = np.sum(P_b_t3 * q_values_stage3)  # E[U^{t=3}_b] = \sum_i P^b(a_3=i | **m***) * (r(i) - m***(i))

            # Q^{b,t=1}_{k=l} = U^{t=1}_b + E[U^{t=3}_b]
            q_values_Q_b_t1[action_a1] = immediate_U_b_t1 + expected_U_b_t3

        return q_values_Q_b_t1


class BaseAgent:
    """
    기본 에이전트 클래스: 공통 기능 제공
    사용 용도: 소프트맥스 정책 계산 및 행동 선택
    결과물: 확률적 행동 선택
    """

    def __init__(self, beta: float = DEFAULT_BETA):
        self.beta = beta

    def compute_softmax_policy_P(self, q_values: np.ndarray) -> np.ndarray:
        """Q값을 소프트맥스 정책 P^{agent,t}로 변환"""
        return NotationUtils.compute_softmax_policy_P(q_values, self.beta)

    def select_action_a(self, q_values: np.ndarray) -> int:
        """소프트맥스 정책에 따른 확률적 행동 선택"""
        policy_probs = self.compute_softmax_policy_P(q_values)
        return np.random.choice(len(q_values), p=policy_probs)


class BuyerSellerEnvironment:
    """
    3단계 구매자-판매자 상호작용 환경
    사용 용도: 게임 시뮬레이션 및 상태 관리
    결과물: 각 단계별 상태 전이 및 최종 보상 계산
    """

    def __init__(
        self,
        n_items: int = DEFAULT_N_ITEMS,
        max_distance: float = DEFAULT_MAX_VALUE,
        max_price: float = DEFAULT_MAX_VALUE,
    ):
        self.n_items = n_items
        self.max_distance = max_distance
        self.max_price = max_price
        self.items = ["apple", "orange"]

    def reset(self, vector_r: np.ndarray, vector_d: np.ndarray):
        """
        환경 초기화: 구매자 선호도 **r**와 거리 **d** 설정
        제약조건 적용: sum(**r**) = sum(**d**) = DEFAULT_MAX_VALUE
        """
        assert len(vector_r) == self.n_items
        assert len(vector_d) == self.n_items

        self.vector_r = NotationUtils.normalize_vector_to_sum(vector_r)
        self.vector_d = NotationUtils.normalize_vector_to_sum(vector_d)

        self.stage_t = 1
        self.buyer_choice_a1 = None
        self.vector_m = None

        return self.get_state()

    def get_state(self):
        """현재 환경 상태 반환"""
        return {
            "stage_t": self.stage_t,
            "vector_d": self.vector_d,
            "vector_r": self.vector_r,
            "buyer_choice_a1": self.buyer_choice_a1,
            "vector_m": self.vector_m,
        }

    def step(self, action: Dict):
        """
        행동 실행 및 다음 단계로 전이
        """
        if self.stage_t == 1:  # t=1: 구매자의 아이템 선택 저장
            self.buyer_choice_a1 = action["buyer_choice_a1"]
            self.stage_t = 2

        elif self.stage_t == 2:  # t=2: 판매자의 가격 설정
            self.vector_m = action["vector_m"]
            self.stage_t = 3

        elif self.stage_t == 3:  # t=3: 최종 구매 및 보상 계산
            buyer_choice_a3 = action["buyer_choice_a3"]

            # Calculate U^{total}_b and U^{total}_s
            utility_U_b_t1 = (
                self.vector_r[self.buyer_choice_a1]
                - self.vector_d[self.buyer_choice_a1]
            )
            utility_U_b_t3 = (
                self.vector_r[buyer_choice_a3] - self.vector_m[buyer_choice_a3]
            )
            utility_U_b_total = utility_U_b_t1 + utility_U_b_t3

            utility_U_s_total = self.vector_m[buyer_choice_a3]

            self.stage_t = 4

            return {
                "utility_U_b_total": utility_U_b_total,
                "utility_U_s_total": utility_U_s_total,
                "done": True,
            }

        return {"done": False}


class LevelLBuyer(BaseAgent, StrategicPlanningMixin):
    """
    Level-l 구매자: Q^{b,t}_{k=l} 기반 행동 선택
    사용 용도: 다양한 수준의 전략적 사고를 가진 구매자 모델링
    결과물: 레벨별로 다른 1단계 정책, 동일한 3단계 정책
    """

    def __init__(
        self,
        level_l: int,
        seller_model: Optional[SellerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.seller_model = seller_model  # Level-(l-1) seller model -> optimal deterministic 가정

    def act_t1(self, state: Dict) -> int:
        """
        1단계 행동 선택: level에 따라 다른 Q값 계산
        """
        if self.level_l == 0:
            # Level-0: Q^{b,t=1}_{k=0} = **r** - **d** (근시안적)
            q_values_Q_b_t1_k0 = state["vector_r"] - state["vector_d"]
            return self.select_action_a(q_values_Q_b_t1_k0)
        else:
            # Level-l: Q^{b,t=1}_{k=l} = Q^{b,t=1}_{k=0} + E[U^{t=3}_b] (전략적)
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                state["vector_d"], state["vector_r"], self.seller_model, self.beta
            )
            return self.select_action_a(q_values_Q_b_t1_kl)

    def act_t3(self, state: Dict) -> int:
        """
        3단계 행동 선택: 모든 레벨이 동일
        """
        # Q^{b,t=3} = **r** - **m** (가격이 이미 정해진 상태)
        q_values_Q_b_t3 = state["vector_r"] - state["vector_m"]
        return self.select_action_a(q_values_Q_b_t3)

    def get_P_t1(self, vector_d: np.ndarray, vector_r: np.ndarray) -> np.ndarray:
        """
        1단계 정책 확률 반환: P^{b,t=1}_{k=l}(a_1 | **r**, **d**)
        판매자의 베이지안 추론에서 우도 함수로 사용
        """
        if self.level_l == 0:
            # Base case: P^{b,t=1}_{k=0}
            q_values_Q_b_t1_k0 = vector_r - vector_d
            return self.compute_softmax_policy_P(q_values_Q_b_t1_k0)
        else:
            # Recursive case: P^{b,t=1}_{k=l}
            q_values_Q_b_t1_kl = self.compute_Q_b_t1_kl(
                vector_d, vector_r, self.seller_model, self.beta
            )
            return self.compute_softmax_policy_P(q_values_Q_b_t1_kl)


class LevelLSeller(BaseAgent, BayesianIRLMixin, PriceOptimizationMixin):
    """
    Level-l 판매자: p^s(**r** | a_1) 추론 및 **m***_{k=l} 최적화
    사용 용도: 다양한 수준의 구매자 모델링을 통한 전략적 가격 설정
    결과물: 관측된 행동 기반 선호도 추론과 수익 최대화 가격
    """

    def __init__(
        self,
        level_l: int,
        buyer_model: Optional[BuyerAgentProtocol] = None,
        beta: float = DEFAULT_BETA,
        n_preference_points: int = DEFAULT_N_PREFERENCE_POINTS,
        defensive: bool = False,
    ):
        super().__init__(beta)
        self.level_l = level_l
        self.buyer_model = buyer_model  # Level-(l-1) buyer model -> optimal deterministic 가정
        self.n_preference_points = n_preference_points
        self.defensive = defensive

    def p_r_given_a(
        self, observed_action_a1: int, vector_d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        베이지안 선호도 추론: p^s(**r** | a_1)
        a_1: 구매자가 stage 1에서 선택한 아이템
        """
        if self.level_l == 0:
            # Level-0: 균등 분포 (구매자 행동 무시)
            preference_grid_r = NotationUtils.create_preference_grid_r(
                self.n_preference_points
            )
            uniform_posterior = NotationUtils.create_uniform_prior_p_r(
                len(preference_grid_r)
            )
            return preference_grid_r, uniform_posterior
        else:
            # Level-l: P^b_{k=(l-1)}(a_1 | **r**, **d**)를 우도로 사용한 베이지안 업데이트
            return self.p_r_given_a_impl(
                observed_action_a1, vector_d, self.buyer_model, self.n_preference_points
            )

    def compute_m_star(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """
        최적 가격 계산: **m***_{k=l}
        """
        if self.level_l == 0:
            # Level-0: 균등 가격 (5, 5)
            return np.array([DEFAULT_MAX_VALUE / 2, DEFAULT_MAX_VALUE / 2])
        elif self.defensive:
            # Defensive: 전략적 구매자에 대한 방어적 가격
            return self._compute_defensive_pricing(
                preference_grid_r, posterior_p_r_given_a
            )
        else:
            # Standard: 표준 수익 최대화 가격
            return self.optimize_vector_m_star(
                preference_grid_r, posterior_p_r_given_a, self.beta
            )

    def _compute_defensive_pricing(
        self, preference_grid_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> np.ndarray:
        """
        방어적 가격 계산: 전략적 구매자의 블러핑에 방지 -> 왜 방어적인지?
        """
        vector_m = np.zeros(DEFAULT_N_ITEMS)
        n_price_points = DEFAULT_N_PRICE_POINTS

        for item in range(DEFAULT_N_ITEMS):  # 각 아이템별로 독립적인 가격 최적화
            best_price = 0
            best_expected_U_s = 0

            for price in np.linspace(0, DEFAULT_MAX_VALUE, n_price_points):  # 각 아이템의 가격별로
                expected_U_s = 0

                for i, r_apple in enumerate(preference_grid_r):  # 각 선호도 포인트별로
                    r_orange = DEFAULT_MAX_VALUE - r_apple
                    vector_r = np.array([r_apple, r_orange])

                    # 현재 아이템의 가격 벡터
                    test_vector_m = (
                        np.array([price, DEFAULT_MAX_VALUE - price])
                        if item == 0
                        else np.array([DEFAULT_MAX_VALUE - price, price])
                    )

                    # Q^{b,t=3} = sigma(beta * (**r** - **m**))
                    q_values_Q_b_t3 = vector_r - test_vector_m
                    P_b_t3 = NotationUtils.compute_softmax_policy_P(
                        q_values_Q_b_t3, self.beta
                    )

                    # 현재 아이템의 가격 벡터에 대한 기대 판매자 수익 E[U^{t=3}_s]
                    expected_U_s += posterior_p_r_given_a[i] * P_b_t3[item] * price

                if expected_U_s > best_expected_U_s:  # 가장 높은 기대 수익을 가지는 가격 선택
                    best_expected_U_s = expected_U_s
                    best_price = price

            vector_m[item] = best_price

        # 아이템별 최적가격 조합 후 제약조건 적용
        return NotationUtils.normalize_vector_to_sum(vector_m)


class InformationTheoryMetrics:
    """
    정보 이론 지표 계산: I(**r**, a_1) 및 D_KL
    사용 용도: 선호도-행동 간 상호정보량, 믿음 업데이트 측정
    결과물: 결과물에 사용되는 정보 이론적 지표들
    """

    @staticmethod
    def compute_I_r_a1(preference_values: np.ndarray, policies: np.ndarray) -> float:
        """
        상호정보량 계산: I(**r**, a_1)
        
        공식: I(r,a) = \sum_r \sum_a p(r,a) \log(p(r,a)/(p(r)p(a)))
        
        해석: 선호도와 행동 간의 의존성 측정
        - 높은 값: 행동이 선호도를 잘 반영
        - 낮은 값: 행동이 선호도와 무관 (기만 성공)
        """
        n_prefs, n_actions = policies.shape

        # Marginal distribution over actions
        action_probs = np.mean(policies, axis=0)

        # Calculate MI using definition
        mi = 0
        for i in range(n_prefs):
            for a in range(n_actions):
                if policies[i, a] > EPSILON:  # Avoid log(0)
                    mi += (
                        (1 / n_prefs)
                        * policies[i, a]
                        * np.log(policies[i, a] / action_probs[a])
                    )

        return mi

    @staticmethod
    def D_KL(p: np.ndarray, q: np.ndarray) -> float:
        """
        KL 발산 계산: D_KL(p || q)
        
        공식: \sum_i p(i) \log(p(i)/q(i))
        
        해석: 분포 p와 q 간의 차이 측정
        """
        # Avoid log(0) by adding small epsilon
        p = p + EPSILON
        q = q + EPSILON
        p = p / np.sum(p)
        q = q / np.sum(q)
        return np.sum(p * np.log(p / q))

    @staticmethod
    def update_belief(
        prior_p_r: np.ndarray, posterior_p_r_given_a: np.ndarray
    ) -> float:
        """
        믿음 업데이트 측정: D_KL(p(**r** | a_1) || p(**r**))
        
        용도: 판매자의 회의론(skepticism) 정도를 측정
        해석: 구매자 행동으로 인한 판매자의 믿음 변화량으로 해석 가능
        """
        return InformationTheoryMetrics.D_KL(posterior_p_r_given_a, prior_p_r)

    @staticmethod
    def policy_mismatch(assumed_policy: np.ndarray, actual_policy: np.ndarray) -> float:
        """
        정책 불일치 측정: D_KL(actual_policy || assumed_policy)
        
        용도: 판매자의 구매자 모델 오류 측정
        해석: 판매자가 가정한 구매자 행동과 실제 행동 간의 차이 (출력 값 차이)
        """
        return InformationTheoryMetrics.D_KL(actual_policy, assumed_policy)


def create_agent_hierarchy(max_level: int, beta: float = DEFAULT_BETA) -> Dict:
    """
    에이전트 계층 구조 생성: Level-0부터 Level-max_level까지
    사용 용도: 재귀적 에이전트 정의를 통한 다양한 수준의 전략적 사고 모델링
    결과물: 각 레벨별 구매자/판매자 에이전트 딕셔너리
    
    구조:
    - Level-0 구매자 → Level-1 판매자 → Level-1 구매자 → ... → Level-max_level
    - 각 Level-l 에이전트는 Level-(l-1) 상대방 모델을 사용
    - defensive seller: 전략적 구매자의 블러핑에 방지
    - advanced buyer: 전략적 구매자 (간간히 블러핑 시도)
    """
    agents = {}

    # Base case: Level-0 buyer
    agents["buyer_l0"] = LevelLBuyer(level_l=0, seller_model=None, beta=beta)

    # Build hierarchy recursively
    for level in range(1, max_level + 1):
        # Level-l seller uses Level-(l-1) buyer model
        buyer_model_prev = agents[f"buyer_l{level-1}"]
        seller_standard = LevelLSeller(
            level_l=level, buyer_model=buyer_model_prev, beta=beta, defensive=False
        )
        seller_defensive = LevelLSeller(
            level_l=level, buyer_model=buyer_model_prev, beta=beta, defensive=True
        )

        agents[f"seller_l{level}"] = seller_standard
        agents[f"seller_l{level}_defensive"] = seller_defensive

        # Level-l buyer uses Level-l seller model
        buyer_standard = LevelLBuyer(
            level_l=level, seller_model=seller_standard, beta=beta
        )
        buyer_advanced = LevelLBuyer(
            level_l=level, seller_model=seller_defensive, beta=beta
        )

        agents[f"buyer_l{level}"] = buyer_standard
        agents[f"buyer_l{level}_advanced"] = buyer_advanced

    return agents


def compute_single_combination(args):
    """
    단일 (r_apple, d_apple) 조합에 대한 지표 계산
    사용 용도: 병렬 처리를 위한 워커 함수
    결과물: 해당 조합에서의 정책, 믿음 업데이트, 정책 불일치 지표들
    
    과정:
    1. 
    2. 베이지안 추론을 통한 믿음 업데이트 계산
    3. 정책 불일치 (판매자 오류) 계산
    """
    r_apple, d_apple, beta, max_level = args

    # Create generalized agent hierarchy for this worker
    agents = create_agent_hierarchy(max_level=max_level, beta=beta)

    vector_r = np.array([r_apple, DEFAULT_MAX_VALUE - r_apple])
    vector_d = np.array([d_apple, DEFAULT_MAX_VALUE - d_apple])

    # Get policies P^{agent,t=1} from generalized agents
    P_level0_current = agents["buyer_l0"].get_P_t1(vector_d, vector_r)
    P_level1_current = agents["buyer_l1"].get_P_t1(vector_d, vector_r)
    P_level1_advanced_current = agents["buyer_l1_advanced"].get_P_t1(vector_d, vector_r)
    P_level2_advanced_current = agents["buyer_l2_advanced"].get_P_t1(vector_d, vector_r)

    # Determine which item is closer for the seller (closer item is notation "a_1")
    a_1_buyer = 0 if d_apple < (DEFAULT_MAX_VALUE - d_apple) else 1

    # Calculate belief updates D_KL only for the closer item selection
    # Level1 "standard" seller belief update when closer item is chosen
    _, posterior_p_r_given_a_level1 = agents["seller_l1"].p_r_given_a(
        a_1_buyer, vector_d
    )
    prior_p_r = NotationUtils.create_uniform_prior_p_r(
        len(posterior_p_r_given_a_level1)
    )
    belief_update_level1 = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a_level1
    )  # level-1 판매자의 믿음 업데이트 정도: 판매자가 가정한 구매자 행동(posterior)과 실제 행동(prior_p_r) 간의 차이

    # Level1 "defensive" seller belief update when closer item is chosen
    _, posterior_p_r_given_a_defensive = agents["seller_l1_defensive"].p_r_given_a(
        a_1_buyer, vector_d
    )
    belief_update_level1_def = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a_defensive
    )

    # Level2 "defensive" seller belief update when closer item is chosen
    _, posterior_p_r_given_a_level2_def = agents["seller_l2_defensive"].p_r_given_a(
        a_1_buyer, vector_d
    )
    belief_update_level2_def = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a_level2_def
    )

    # Calculate policy mismatch 
    # What Lv1 seller assumes vs actual Lv1 buyer --> 왜 P_lvel0가 Lv1 seller assumption인지?
    policy_mismatch_l0_vs_l1 = InformationTheoryMetrics.policy_mismatch(
        P_level0_current, P_level1_current
    )

    # What Lv1 seller assumes vs actual Lv1 advanced buyer
    policy_mismatch_l0_vs_l1_adv = InformationTheoryMetrics.policy_mismatch(
        P_level0_current, P_level1_advanced_current
    )

    # What Lv2 defensive seller assumes vs actual Lv2 buyer
    policy_mismatch_l1_adv_vs_l2_adv = InformationTheoryMetrics.policy_mismatch(
        P_level1_advanced_current, P_level2_advanced_current
    )

    return {
        "r_apple": r_apple,
        "d_apple": d_apple,
        "policies": {
            "P_level0": P_level0_current,
            "P_level1": P_level1_current,
            "P_level1_advanced": P_level1_advanced_current,
            "P_level2_advanced": P_level2_advanced_current,
        },
        "belief_updates": {
            "level1": belief_update_level1,
            "level1_def": belief_update_level1_def,
            "level2_def": belief_update_level2_def,
        },
        "policy_mismatches": {
            "l0_vs_l1": policy_mismatch_l0_vs_l1,
            "l0_vs_l1_adv": policy_mismatch_l0_vs_l1_adv,
            "l1_adv_vs_l2_adv": policy_mismatch_l1_adv_vs_l2_adv,
        },
        "closer_item": a_1_buyer,
    }


def run_systematic_analysis(
    beta: float = DEFAULT_BETA,
    n_samples: int = 20,
    max_level: int = 2,
    n_processes: int = None,
):
    """
    체계적 분석 실행: 다양한 선호도/거리 조합에서 정보 이론 지표 계산
    사용 용도: Figure 3F,G,H 데이터 생성을 위한 대규모 시뮬레이션
    결과물: 거리별 집계된 상호정보량, 믿음 업데이트, 정책 오류 지표
    
    분석 방법:
    1. 병렬 처리로 모든 (선호도, 거리) 조합 계산
    2. 거리별로 결과 집계 (선호도에 대한 평균/분포)
    3. 각 에이전트 조합별 정보 이론 지표 계산
    """

    if n_processes is None:
        n_processes = mp.cpu_count() - 1

    # Create parameter grids for **r** and **d**
    candidate_vector_r = np.linspace(1, 9, n_samples)
    candidate_vector_d = np.linspace(1, 9, n_samples)

    # Store results - x축을 distance로 변경
    results = {
        "L0B_vs_L1S": {
            "mutual_info_I_r_a1": [],  # distance별 MI
            "belief_updates_D_KL": [],  # distance별 belief update (closer item chosen)
            "policies_P": [],  # distance별 평균 policy
            "seller_error": [],  # distance별 policy mismatch (Figure 3H)
        },
        "L1B_vs_L1S": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
        "L1B_vs_L1S_D": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
        "L2B_vs_L2S_D": {
            "mutual_info_I_r_a1": [],
            "belief_updates_D_KL": [],
            "policies_P": [],
            "seller_error": [],
        },
    }

    logger.info(f"Starting parallel computation with {n_processes} processes")
    logger.info(
        f"Total combinations: {len(candidate_vector_r)} × {len(candidate_vector_d)} = {len(candidate_vector_r) * len(candidate_vector_d)}"
    )

    # 모든 (r_apple, d_apple) 조합 생성
    all_combinations = [
        (r_apple, d_apple, beta, max_level)
        for r_apple in candidate_vector_r
        for d_apple in candidate_vector_d
    ]

    # 병렬 처리
    with mp.Pool(processes=n_processes) as pool:
        parallel_results = list(
            tqdm(
                pool.imap(compute_single_combination, all_combinations),
                total=len(all_combinations),
                desc="Processing combinations in parallel",
            )
        )

    logger.info("Parallel computation completed. Aggregating results...")

    # 결과를 d_apple별로 그룹화하고 집계
    for dist_idx, d_apple in tqdm(
        enumerate(candidate_vector_d),
        desc="Aggregating results by distance",
        total=len(candidate_vector_d),
    ):
        # 현재 d_apple에 해당하는 결과들 필터링
        current_d_results = [
            r for r in parallel_results if np.isclose(r["d_apple"], d_apple)
        ]

        # 정렬 (r_apple 순서대로)
        current_d_results.sort(key=lambda x: x["r_apple"])

        # 각 레벨의 정책들 수집 (모든 preference에 대한 정책)
        P_level0_all_prefs = np.array(
            [r["policies"]["P_level0"] for r in current_d_results]
        )
        P_level1_all_prefs = np.array(
            [r["policies"]["P_level1"] for r in current_d_results]
        )
        P_level1_advanced_all_prefs = np.array(
            [r["policies"]["P_level1_advanced"] for r in current_d_results]
        )
        P_level2_advanced_all_prefs = np.array(
            [r["policies"]["P_level2_advanced"] for r in current_d_results]
        )

        # Calculate mutual information I(**r**, a_1) for current distance
        preference_values = np.array([r["r_apple"] for r in current_d_results])

        mi_I_r_a1_level0 = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level0_all_prefs
        )
        mi_I_r_a1_level1 = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level1_all_prefs
        )
        mi_I_r_a1_level1_advanced = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level1_advanced_all_prefs
        )
        mi_I_r_a1_level2_advanced = InformationTheoryMetrics.compute_I_r_a1(
            preference_values, P_level2_advanced_all_prefs
        )

        # Get belief updates and policy mismatches (averages)
        belief_updates_level1 = np.mean(
            [r["belief_updates"]["level1"] for r in current_d_results]
        )
        belief_updates_level1_def = np.mean(
            [r["belief_updates"]["level1_def"] for r in current_d_results]
        )
        belief_updates_level2_def = np.mean(
            [r["belief_updates"]["level2_def"] for r in current_d_results]
        )

        policy_mismatch_l0_vs_l1 = np.mean(
            [r["policy_mismatches"]["l0_vs_l1"] for r in current_d_results]
        )
        policy_mismatch_l0_vs_l1_adv = np.mean(
            [r["policy_mismatches"]["l0_vs_l1_adv"] for r in current_d_results]
        )
        policy_mismatch_l1_adv_vs_l2_adv = np.mean(
            [r["policy_mismatches"]["l1_adv_vs_l2_adv"] for r in current_d_results]
        )

        # Store results
        results["L0B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level0)
        results["L0B_vs_L1S"]["belief_updates_D_KL"].append(belief_updates_level1)
        results["L0B_vs_L1S"]["policies_P"].append(P_level0_all_prefs.mean(axis=0))
        results["L0B_vs_L1S"]["seller_error"].append(0.0)  # No error for baseline

        results["L1B_vs_L1S"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level1)
        results["L1B_vs_L1S"]["belief_updates_D_KL"].append(belief_updates_level1)
        results["L1B_vs_L1S"]["policies_P"].append(P_level1_all_prefs.mean(axis=0))
        results["L1B_vs_L1S"]["seller_error"].append(policy_mismatch_l0_vs_l1)

        results["L1B_vs_L1S_D"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level1_advanced)
        results["L1B_vs_L1S_D"]["belief_updates_D_KL"].append(belief_updates_level1_def)
        results["L1B_vs_L1S_D"]["policies_P"].append(
            P_level1_advanced_all_prefs.mean(axis=0)
        )
        results["L1B_vs_L1S_D"]["seller_error"].append(policy_mismatch_l0_vs_l1_adv)

        results["L2B_vs_L2S_D"]["mutual_info_I_r_a1"].append(mi_I_r_a1_level2_advanced)
        results["L2B_vs_L2S_D"]["belief_updates_D_KL"].append(belief_updates_level2_def)
        results["L2B_vs_L2S_D"]["policies_P"].append(
            P_level2_advanced_all_prefs.mean(axis=0)
        )
        results["L2B_vs_L2S_D"]["seller_error"].append(policy_mismatch_l1_adv_vs_l2_adv)

        logger.info(
            f"[Systematic Analysis] Aggregated results for distance {dist_idx + 1}/{len(candidate_vector_d)}"
        )

    # Save results with notation
    results_serializable = {}
    for key, value in results.items():
        results_serializable[key] = {
            "mutual_info_I_r_a1": [float(x) for x in value["mutual_info_I_r_a1"]],
            "belief_updates_D_KL": [float(x) for x in value["belief_updates_D_KL"]],
            "policies_P": [policy.tolist() for policy in value["policies_P"]],
            "seller_error": [float(x) for x in value["seller_error"]],
        }

    with open(DATA_DIR, "w") as f:
        json.dump(results_serializable, f, indent=2)

    return results, candidate_vector_d


def plot_results(results: Dict, candidate_vector_d: np.ndarray):
    """Plot results following notation - including Figure 3G,H"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot mutual information I(**r**, a₁) - Figure 3F
    ax = axes[0, 0]
    ax.plot(
        candidate_vector_d,
        results["L0B_vs_L1S"]["mutual_info_I_r_a1"],
        label="Lv0 buyer",
        linewidth=2,
        color="darkgreen",
    )
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L1S"]["mutual_info_I_r_a1"],
        label="Lv1 buyer",
        linewidth=2,
        color="orange",
    )
    ax.plot(
        candidate_vector_d,
        results["L2B_vs_L2S_D"]["mutual_info_I_r_a1"],
        label="Lv2 buyer",
        linewidth=2,
        color="lightgreen",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("Mutual Information I(r, a₁)")
    ax.set_title("Figure 3F: Preference-Policy MI")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot belief updates D_KL - Figure 3G
    ax = axes[0, 1]
    ax.plot(
        candidate_vector_d,
        results["L0B_vs_L1S"]["belief_updates_D_KL"],
        label="Lv1 seller",
        linewidth=2,
        color="darkblue",
    )
    ax.plot(
        candidate_vector_d,
        results["L2B_vs_L2S_D"]["belief_updates_D_KL"],
        label="Lv2 seller",
        linewidth=2,
        color="lightblue",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Figure 3G: Seller Skepticism")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot seller error - Figure 3H
    ax = axes[0, 2]
    ax.plot(
        candidate_vector_d,
        results["L1B_vs_L1S"]["seller_error"],
        label="Lv1 seller error",
        linewidth=2,
        color="darkgreen",
    )
    ax.plot(
        candidate_vector_d,
        results["L2B_vs_L2S_D"]["seller_error"],
        label="Lv2 seller error",
        linewidth=2,
        color="lightgreen",
    )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Figure 3H: Seller Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot action probabilities P^{agent,t=1}
    ax = axes[1, 0]
    for key in ["L0B_vs_L1S", "L1B_vs_L1S", "L2B_vs_L2S_D"]:
        policies_P = np.array(results[key]["policies_P"])
        ax.plot(
            candidate_vector_d,
            policies_P[:, 0],
            label=f"{key.split('_vs_')[0]}",
            linewidth=2,
        )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("P(Choose Apple)")
    ax.set_title("Buyer Policies")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot deception effectiveness (ΔI)
    ax = axes[1, 1]
    naive_mi_I_r_a1 = results["L0B_vs_L1S"]["mutual_info_I_r_a1"]
    for key in ["L1B_vs_L1S", "L2B_vs_L2S_D"]:
        deception_delta_I = np.array(naive_mi_I_r_a1) - np.array(
            results[key]["mutual_info_I_r_a1"]
        )
        ax.plot(
            candidate_vector_d,
            deception_delta_I,
            label=key.split("_vs_")[0],
            linewidth=2,
        )
    ax.set_xlabel("Distance to Apple")
    ax.set_ylabel("Deception Effectiveness (ΔI)")
    ax.set_title("Information Hiding")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Empty plot for layout
    ax = axes[1, 2]
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOT_DIR, dpi=300, bbox_inches="tight")
    logger.info(f"Plot saved to {PLOT_DIR}")


# Example usage
if __name__ == "__main__":
    logger.info("Running systematic analysis with generalized level-l agents...")

    # Run analysis with level hierarchy up to 2 and parallel processing
    results, candidate_vector_d = run_systematic_analysis(
        beta=DEFAULT_BETA,
        n_samples=N_SAMPLES,
        max_level=MAX_LEVEL,
        n_processes=None,
    )

    logger.info("Plotting results...")
    # Plot results
    plot_results(results, candidate_vector_d)

    # Log summary statistics
    logger.info("\n=== Summary Statistics (Generalized Level-l Notation) ===")
    for key in results.keys():
        mi_I_r_a1_avg = np.mean(results[key]["mutual_info_I_r_a1"])
        belief_D_KL_avg = np.mean(results[key]["belief_updates_D_KL"])
        seller_error_avg = np.mean(results[key]["seller_error"])
        logger.info(f"{key}:")
        logger.info(f"  Average I(**r**, a₁): {mi_I_r_a1_avg:.3f}")
        logger.info(f"  Average D_KL(p(**r**|a₁)||p(**r**)): {belief_D_KL_avg:.3f}")
        logger.info(f"  Average Seller Error: {seller_error_avg:.3f}")

    # Demonstrate specific example with generalized agents
    logger.info("\n=== Example Interaction (Generalized Level-l) ===")
    vector_r = np.array([7, 3])  # Strong preference for apple
    vector_d = np.array([3, 7])  # Apple is closer

    env = BuyerSellerEnvironment()

    # Create agents using generalized hierarchy
    agents = create_agent_hierarchy(max_level=2, beta=2.0)
    buyer_l1 = agents["buyer_l1"]
    seller_l1 = agents["seller_l1"]

    state = env.reset(vector_r, vector_d)

    # Stage t=1
    buyer_choice_a1 = buyer_l1.act_t1(state)
    env.step({"buyer_choice_a1": buyer_choice_a1})

    # Stage t=2
    preference_grid_r, posterior_p_r_given_a = seller_l1.p_r_given_a(
        buyer_choice_a1, vector_d
    )
    vector_m = seller_l1.compute_m_star(preference_grid_r, posterior_p_r_given_a)
    env.step({"vector_m": vector_m})

    # Calculate metrics
    prior_p_r = np.ones_like(posterior_p_r_given_a) / len(posterior_p_r_given_a)
    belief_update_D_KL = InformationTheoryMetrics.update_belief(
        prior_p_r, posterior_p_r_given_a
    )

    logger.info(f"Buyer **r**: Apple={vector_r[0]:.1f}, Orange={vector_r[1]:.1f}")
    logger.info(f"Distance **d**: Apple={vector_d[0]:.1f}, Orange={vector_d[1]:.1f}")
    logger.info(f"Stage t=1 choice a₁: {'Apple' if buyer_choice_a1 == 0 else 'Orange'}")
    logger.info(f"Seller **m**: Apple={vector_m[0]:.2f}, Orange={vector_m[1]:.2f}")
    logger.info(f"Belief update D_KL: {belief_update_D_KL:.3f}")

    # Demonstrate extensibility
    logger.info("\n=== Agent Hierarchy Extensibility ===")
    logger.info(f"Current hierarchy: {list(agents.keys())}")
    logger.info(
        "Can easily extend to level-3, level-4, etc. by changing max_level parameter"
    )
