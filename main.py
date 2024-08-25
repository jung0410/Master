import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt, find_peaks

# 데이터 파일 불러오기. 데이터는 첫 번째 열이 시간, 두 번째 열이 진폭인 형태로 가정.

# 파일 경로 및 파일 이름 지정.
# # file_path = 'C:/Users/Win/Desktop/detect/F15.txt'
# # file_path = 'C:/Users/Win/Desktop/detect/J4.txt'
# file_path = 'C:/Users/Win/Desktop/detect/J10.txt'
# #
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


# Band-Pass 필터를 설계하는 함수
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    print(f"nayq {nyq}")
    low = lowcut / nyq
    print(f"low {low}")
    high = highcut / nyq
    print(f" high  { high }")
    print()
    b, a = butter(order, [low, high], btype='band')
    return b, a


# Band-Pass 필터를 적용하는 함수
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# 피크 주파수를 찾는 함수
def find_peak_frequency(frequencies, amplitudes, low_freq, high_freq):
    # 지정한 주파수 대역 내에서 피크를 찾습니다.
    mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    filtered_frequencies = frequencies[mask]
    filtered_amplitudes = amplitudes[mask]

    # 피크 찾기
    peaks, _ = find_peaks(filtered_amplitudes)

    if peaks.size > 0:
        # 가장 큰 피크의 주파수를 반환
        peak_freq = filtered_frequencies[peaks[np.argmax(filtered_amplitudes[peaks])]]
        return peak_freq
    else:
        return None


# 시간 벡터와 진폭 데이터를 각각 분리
time = data[:, 0]
print(time)
amplitude = data[:, 1]

# DC offset 제거 (신호의 평균값 빼기)
amplitude = amplitude - np.mean(amplitude)

# 샘플링 주기를 계산
dt = time[1] - time[0]
print(dt)

# 샘플링 주파수를 계산
Fs = 1.0 / dt
print(f"fs입니다: {Fs}")

# Band-Pass 필터링 수행
lowcut = 300.0  # 낮은 컷오프 주파수 (Hz)
highcut = 30000.0  # 높은 컷오프 주파수 (Hz)

filtered_amplitude = butter_bandpass_filter(amplitude, lowcut, highcut, Fs, order=2)

# FFT를 수행하여 주파수 영역으로 변환
n = len(filtered_amplitude)
amplitude_fft = fft(filtered_amplitude)
frequency = np.fft.fftfreq(n, d=dt)

# 양의 주파수 성분만을 가져옴
positive_freq = frequency[:n // 2]

positive_amplitude = np.abs(amplitude_fft[:n // 2]) * (2.0 / n)

# 시간 영역 신호와 주파수 스펙트럼을 플롯
plt.figure(figsize=(12, 10))

# 시간 영역 신호 플롯
plt.subplot(2, 1, 1)
plt.plot(time * 1e3, filtered_amplitude, color='r')
plt.title('Time Domain Signal')
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.grid(True)

# 주파수 스펙트럼 플롯
plt.subplot(2, 1, 2)
plt.plot(positive_freq / 1e3, positive_amplitude, color='b')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [kHz]')
plt.xlim([0, 30])
plt.ylabel('Amplitude')
plt.grid(True)

# 플롯을 화면에 표시
plt.tight_layout()
plt.show()

# 지정한 주파수 대역 내에서 피크 주파수 찾기
low_freq = 500.0  # 피크를 찾을 주파수 대역의 낮은 주파수 (Hz)
high_freq = 30000.0  # 피크를 찾을 주파수 대역의 높은 주파수 (Hz)

peak_freq = find_peak_frequency(positive_freq, positive_amplitude, low_freq, high_freq)

if peak_freq is not None:
    plt.axvline(x=peak_freq / 1e3, color='g', linestyle='--', label=f'Peak Frequency: {peak_freq / 1e3:.2f} kHz')
    plt.legend()

# 플롯을 화면에 표시
plt.tight_layout()
plt.show()

# 피크 주파수를 출력
if peak_freq is not None:
    print(f'Peak frequency within {low_freq} Hz and {high_freq} Hz: {peak_freq:.2f} Hz')
else:
    print(f'No peak found within {low_freq} kHz and {high_freq} Hz.')

