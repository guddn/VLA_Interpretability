"""vlamod — VLA 모달리티 기여 비율 분석 툴킷.

설계 원칙
---------
1. 하드코딩 금지. 토큰 개수/위치는 **실측**한다. (문서의 "256개"를 믿지 않는다)
2. attention 은 주장의 근거가 아니라 **검정 대상**이다. 인과 개입(KL)이 주 증거다.
3. 모든 지표는 raw / per-token-normalized / value-norm-weighted 3종을 함께 낸다.
"""

__version__ = "0.1.0"
