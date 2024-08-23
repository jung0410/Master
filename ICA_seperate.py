import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# 파일 경로 및 파일 이름 지정.
##shallow
# file_path = 'C:/Users/Win/Desktop/detect/F15.txt'
# file_path = 'C:/Users/Win/Desktop/detect/J4.txt'
# file_path = 'C:/Users/Win/Desktop/detect/J10.txt'
#
#
# file_path = 'C:/Users/Win/Desktop/detect/E10.txt'
# file_path = 'C:/Users/Win/Desktop/detect/I16.txt'
# file_path = 'C:/Users/Win/Desktop/detect/K14.txt'

##shallow
# file_path = 'C:/Users/Win/Desktop/detect/D12.txt'
# file_path = 'C:/Users/Win/Desktop/detect/J8.txt'
# file_path = 'C:/Users/Win/Desktop/detect/M8.txt'


#poor
# file_path = 'C:/Users/Win/Desktop/detect/D4.txt'
file_path = 'C:/Users/Win/Desktop/detect/E4.txt'
# 데이터 로드
data = np.loadtxt(file_path, delimiter='\t')
t = data[:, 0]
signal = data[:, 1]

# EMD 객체 생성
emd = EMD()

# IMF 분해 수행
imfs = emd.emd(signal, t)

# 샘플링 간격 계산 (t[1] - t[0]은 시간 간격)
sampling_interval = t[1] - t[0]

# IMF 시각화 (시간 도메인)
n_imfs = imfs.shape[0]

### Original만 표시함
# Original Signal 시각화
# 원본 신호 (시간 도메인) 시각화
plt.figure(figsize=(12, 10))

# 시간 도메인 신호
plt.subplot(2, 1, 1)
plt.plot(t, signal, color='b')
plt.title('Original Signal (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)

# FFT를 사용해 주파수 도메인으로 변환
n = len(signal)
signal_fft = fft(signal)
frequency = fftfreq(n, d=sampling_interval)

# 양수 주파수만 선택
positive_freq = frequency[:n // 2]
positive_amplitude = np.abs(signal_fft[:n // 2]) * (2.0 / n)

# 주파수 스펙트럼 (주파수 대 진폭) 시각화
plt.subplot(2, 1, 2)
plt.plot(positive_freq / 1e3, positive_amplitude, color='r')  # kHz 단위로 변환
plt.title('Original Signal (Frequency Domain)')
plt.xlabel('Frequency [kHz]')
plt.ylabel('Amplitude')
plt.grid(True)

# 플롯을 화면에 표시
plt.tight_layout()
plt.show()




# 신호 및 IMF 시각화 (4개씩 끊어서)
for start in range(0, n_imfs, 4):
    plt.figure(figsize=(12, 8))

    # 원본 신호 표시
    # if start == 0:
    #     plt.subplot(5, 1, 1)  # 첫 번째 그래프에 원본 신호 포함
    #     plt.plot(t, signal, 'r')
    #     plt.title("Original Signal")

    for i in range(start, min(start + 4, n_imfs)):
        plt.subplot(5, 1, i - start + 2)
        plt.plot(t, imfs[i], 'g')
        plt.title(f"IMF {i + 1}")

    plt.tight_layout()
    plt.show()
# 전체 IMF에서 최대 진폭 주파수를 추적하기 위한 변수 초기화
overall_max_freq = 0
overall_max_amp = 0
# 주파수 도메인에서 IMF 시각화 (4개씩 끊어서)
for start in range(0, imfs.shape[0], 4):
    ### plt rmfla zmrl
    plt.figure(figsize=(8, 8))

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
        # 전체 IMF 중 최대 주파수를 추적
        if max_amp_value > overall_max_amp:
            overall_max_amp = max_amp_value
            overall_max_freq = max_amp_freq


        # 그래프 그리기
        plt.plot(positive_freq, positive_amplitude, label='Frequency Spectrum')
        plt.scatter(max_amp_freq, max_amp_value, color='red', marker='o', label=f'Max Frequency: {max_amp_freq:.2f} kHz')

        plt.title(f"IMF {i + 1} - Frequency Domain")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        ##x축 몇까지
        plt.xlim([0, 100])
        plt.legend()

    plt.tight_layout()
    plt.show()
print(f'H치 최대값{overall_max_freq}')