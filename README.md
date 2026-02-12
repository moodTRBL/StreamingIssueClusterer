## Overview

본 프로젝트는 실시간으로 유입되는 뉴스 데이터 스트림을 처리하여, 비슷한 사건을 다루는 기사들을 하나의 이슈로 그룹화하는 **이벤트 기반 클러스터링 엔진**입니다.

### Background

일반적인 클러스터링 알고리즘은 이미 수집된 정적인 데이터셋을 전제로 작동합니다. 그러나 뉴스 환경은 다음과 같은 특수한 제약 사항을 가집니다:

1.  **Continuous Data Stream:** 데이터가 끊임없이 실시간으로 추가되며, 전체 데이터셋의 크기가 무한히 증가합니다.
2.  **Real-time Constraint:** 새로운 기사가 유입될 때마다 전체 데이터를 재학습하거나 재연산하는 비용을 감당할 수 없습니다.
3.  **Temporal Sensitivity:** 뉴스의 가치는 시간에 따라 급격히 감소하며, 동일한 키워드라도 발생 시점에 따라 전혀 다른 사건을 의미할 수 있습니다.

따라서 본 엔진은 기존의 배치 방식이 아닌, 데이터가 유입되는 즉시 기존 클러스터와의 유사성을 판단하여 **병합**하거나 **새로운 이슈를 생성**하는 **Online Clustering**을 채택했습니다. 이를 위해 벡터 임베딩, 시간 감쇠, 그리고 동적 임계값을 결합하여 클러스터의 품질을 유지합니다.

---

## Algorithm Logic

### 1. Vector Embedding
기사의 제목과 본문을 조합하여 텍스트의 맥락을 함축하는 벡터를 생성합니다.
- **Input:** Title + Content
- **Output:** Dense Vector 

### 2. Candidate Scoring (Semantic & Temporal)
새로운 기사 $A$가 입력되면, 현재 활성화된 클러스터 $C$들과의 적합도를 평가합니다. 이때 단순한 의미적 유사도뿐만 아니라 시간적 근접성을 핵심 지표로 활용합니다.

1.  **Semantic Similarity (Cosine Similarity)**
    벡터 간 코사인 유사도를 통해 문맥적 일치도를 계산합니다.

    $$Sim(A, C) = \frac{A \cdot C}{\|A\| \|C\|}$$

2.  **Time Decay Weight (Time Awareness)**
    이슈의 최신성을 반영하기 위해, 마지막 업데이트 시점과의 차이($\Delta t$)에 따라 점수를 지수적으로 감쇠시킵니다. 이는 과거의 유사 사건이 아닌 현재 진행 중인 사건에 우선순위를 부여합니다.

    $$W_{time} = e^{-\lambda \times |\Delta t|}$$

3.  **Final Score Calculation**
    유사도와 시간 가중치를 결합하여 최종 랭킹 점수를 산출합니다.

    $$Score = (\alpha \times Sim_{semantic}) + (\beta \times W_{time})$$

### 3. Dynamic Thresholding (동적 임계값)
고정된 임계값은 시간이 지난 이슈와 최신 이슈를 동일한 기준으로 처리하는 한계가 있습니다. 본 엔진은 **시간이 지날수록 병합 기준을 동적으로 강화**하는 방식을 사용합니다.

- **Logic:**
    - **최신 이슈 ($\Delta t \approx 0$):** 사건이 진행 중이므로 문맥이 다소 변하더라도 병합을 허용 (낮은 임계값 적용).
    - **과거 이슈 ($\Delta t \gg 0$):** 이미 종료된 사건일 가능성이 높으므로, 매우 높은 문맥 일치도가 아니면 병합을 거부하고 새로운 이슈로 분리 (높은 임계값 적용).

- **Threshold Function:**
    $$T_{dynamic}(t) = T_{base} + (1.0 - T_{base}) \times (1.0 - W_{time})$$

### 4. Cluster Validation (Separability)
높은 점수를 받은 후보 클러스터라도, 병합 시 클러스터의 응집도를 해칠 수 있습니다. 이를 방지하기 위해 실루엣 계수(Silhouette Coefficient) 개념을 차용한 **분리도(Separability)** 검사를 수행합니다.

- **Internal Distance ($a$):** 병합 대상 클러스터와의 거리

  $$a = 1.0 - Sim(A, C_{target})$$

- **External Distance ($b$):** 두 번째로 가까운 이웃 클러스터와의 거리

  $$b = 1.0 - Sim(A, C_{neighbor})$$

- **Separability Calculation:**

  $$Separability = \frac{b - a}{\max(a, b)}$$

**Decision:** 최종적으로 `Score`가 높고 `Separability`가 양수인 경우에만 병합을 승인하며, 그렇지 않을 경우 **새로운 이슈(새로운 클러스터)**를 생성합니다.

### 5. Centroid Update (Moving Average)
기사가 클러스터에 병합될 때마다, 클러스터의 대표 벡터를 이동 평균 방식으로 갱신하여 이슈의 성격 변화를 점진적으로 반영합니다.

$$C_{new} = \frac{(C_{old} \times N) + V_{new}}{N + 1}$$

*(N: Number of articles in the cluster)*

참고 논문: https://aclanthology.org/2021.eacl-main.198.pdf