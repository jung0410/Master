import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# 파일 경로 및 파일 이름 지정.
# file_path = 'C:/Users/Win/Desktop/detect/E4.txt'


file_path = 'C:/Users/Win/Desktop/detect/E10.txt'

# 데이터 로드
data = np.loadtxt(file_path, delimiter='\t')
t = data[:, 0]
signal = data[:, 1]

# EMD 객체 생성
emd = EMD()

# IMF 분해 수행
imfs = emd.emd(signal, t)

# IMF 시각화 (시간 도메인)
n_imfs = imfs.shape[0]
plt.figure(figsize=(12, 8))

plt.subplot(n_imfs + 1, 1, 1)
plt.plot(t, signal, 'r')
plt.title("Original Signal")

for i in range(n_imfs):
    plt.subplot(n_imfs + 1, 1, i + 2)
    plt.plot(t, imfs[i], 'g')
    plt.title(f"IMF {i + 1}")

plt.tight_layout()
plt.show()

# 주파수 도메인에서 IMF 시각화
plt.figure(figsize=(12, 8))

# 샘플링 간격 계산 (t[1] - t[0]은 시간 간격)
sampling_interval = t[1] - t[0]
# 주파수 도메인에서 IMF 시각화 (4개씩 끊어서)
for start in range(0, imfs.shape[0], 4):
    plt.figure(figsize=(12, 8))

    for i in range(start, min(start + 4, imfs.shape[0])):
        plt.subplot(4, 1, i - start + 1)

        # FFT 적용
        fft_values = fft(imfs[i])

        # 주파수 벡터 계산
        freq = fftfreq(len(imfs[i]), d=sampling_interval)

        # 양수 주파수만 표시
        positive_freq_idx = freq > 0
        positive_freq = freq[positive_freq_idx] / 1000  # Hz -> kHz로 변환
        positive_amplitude = np.abs(fft_values[positive_freq_idx])

        # 최대 진폭을 가지는 주파수 찾기
        max_amp_index = np.argmax(positive_amplitude)
        max_amp_freq = positive_freq[max_amp_index]
        max_amp_value = positive_amplitude[max_amp_index]

        # 최대 주파수 출력
        print(f"IMF {i + 1}: 최대 진폭을 가지는 주파수 = {max_amp_freq:.2f} kHz")

        # 그래프 그리기
        plt.plot(positive_freq, positive_amplitude, label='Frequency Spectrum')
        plt.scatter(max_amp_freq, max_amp_value, color='red', marker='o',
                    label=f'Max Frequency: {max_amp_freq:.2f} kHz')

        plt.title(f"IMF {i + 1} - Frequency Domain")
        plt.xlabel("Frequency (kHz)")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plt.show()