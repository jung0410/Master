import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 신호 생성 (2개의 독립적인 신호)
t = np.linspace(0, 10, 2000)
s1 = np.sin(2 * t)  # 첫 번째 신호 (사인파)
s2 = np.sign(np.sin(3 * t))  # 두 번째 신호 (정현파)

S = np.c_[s1, s2]  # 독립 신호들
S += 0.2 * np.random.normal(size=S.shape)  # 약간의 노이즈 추가

# 신호 혼합 (임의의 혼합 행렬 사용)
A = np.array([[1, 1], [0.5, 2]])
X = np.dot(S, A.T)  # 관측된 혼합 신호

# ICA 수행
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # 원래 신호 복원

# 결과 시각화
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title('Mixed Signal')
plt.plot(X)

plt.subplot(3, 1, 2)
plt.title('Recovered Signals by ICA')
plt.plot(S_)

plt.subplot(3, 1, 3)
plt.title('Original Signals')
plt.plot(S)

plt.tight_layout()
plt.show()
