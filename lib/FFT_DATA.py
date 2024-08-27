import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
import os

def ensure_directory_exists(path):
    # 디렉터리 경로 추출
    directory = os.path.dirname(path)

    # 디렉터리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

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


def process_and_find_peak(data, lowcut=300.0, highcut=30000.0, order=2):
    # time = data.iloc[:, 0]
    amplitude=data.iloc[:, 0]

    ####문제 발생### 매우 중요
    ### 이게 3초짜리 정보인지 6초짜리 정보인지 어덯게암??? 야 이게 중대 문제
    ###샘플사이즈 400000
    length_of_data = len(amplitude)
    Hz_given=400000
    sample_given=3000
    # print(length_of_data)
    time = np.arange(0, sample_given/ Hz_given , 1 / Hz_given)
    time=pd.DataFrame(time)
    # print(time)
    # 샘플링 주기를 계산
    dt = time.loc[1, 0] - time.loc[0, 0]
    # print(dt)

    # 샘플링 주파수를 계산
    fs = 1.0 / dt
    # print(fs)

    # DC offset 제거 (신호의 평균값 빼기)
    amplitude = amplitude - np.mean(amplitude)

    # Band-Pass 필터 설계 및 적용
    ###특정 주파수 신호만 받고 그 밖 신호 차단
    ### 예시 데이터에서 하기에 넣음
    ### 어지간히 데이터가 안터지면 사용 안 될 듯
    nyq = 0.5 * fs
    # print(nyq)
    low = lowcut / nyq
    # print(low)
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    # print(b)
    filtered_data = filtfilt(b, a, amplitude)

    # FFT 수행
    n = len(filtered_data)
    data_fft = fft(filtered_data)
    frequencies = fftfreq(n, d=(dt))

    # 양의 주파수 성분만을 사용
    ##kHZ를 위해 나누기 1000
    positive_frequencies = frequencies[:n // 2]
    positive_amplitudes = np.abs(data_fft[:n // 2]) * (2.0 / n)

    # 헤르츠(Hz)에서 킬로헤르츠(kHz)로 주파수 단위 변환
    positive_frequencies_kHz = positive_frequencies / 1000.0  # 모든 주파수 값을 1000으로 나눔

    # 피크 주파수 찾기
    # 찾는 범위 또한 kHz 단위로 변환
    lowcut_kHz = lowcut / 1000.0
    highcut_kHz = highcut / 1000.0
    peak_freq = find_peak_frequency(positive_frequencies_kHz, positive_amplitudes, lowcut_kHz, highcut_kHz)

    # 결과 출력
    if peak_freq:
        print(f'Peak frequency within { lowcut_kHz} Hz and {highcut_kHz} Hz: {peak_freq:.2f} Hz')
    else:
        print(f'No peak found within {lowcut_kHz} Hz and {highcut_kHz} Hz.')

    # 그래프로 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time, data, label='Original Signal')
    plt.plot(time, filtered_data, label='Filtered Signal', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot( positive_frequencies_kHz, positive_amplitudes, label='FFT of Filtered Signal')
    if peak_freq:
        plt.axvline(x=peak_freq, color='green', linestyle='--', label=f'Peak Frequency: {peak_freq:.2f} kHz')
    plt.xlabel('Frequency (kHz)')
    plt.xlim([0, 30])
    plt.ylabel('Amplitude')
    plt.title('Frequency Domain')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return peak_freq




def process_and_find_peak_nograph(data,column, lowcut=300.0, highcut=30000.0, order=2):
    # time = data.iloc[:, 0]
    amplitude=data.iloc[:, 0]
    # name=string(column)

    ####문제 발생### 매우 중요
    ### 이게 3초짜리 정보인지 6초짜리 정보인지 어덯게암??? 야 이게 중대 문제
    ###샘플사이즈 400000
    length_of_data = len(amplitude)
    Hz_given=400000
    sample_given=3000
    # print(length_of_data)
    time = np.arange(0, sample_given/ Hz_given , 1 / Hz_given)
    time=pd.DataFrame(time)
    # print(time)
    # 샘플링 주기를 계산
    dt = time.loc[1, 0] - time.loc[0, 0]
    # print(dt)

    # 샘플링 주파수를 계산
    fs = 1.0 / dt
    # print(fs)

    # DC offset 제거 (신호의 평균값 빼기)
    amplitude = amplitude - np.mean(amplitude)

    # Band-Pass 필터 설계 및 적용
    ###특정 주파수 신호만 받고 그 밖 신호 차단
    ### 예시 데이터에서 하기에 넣음
    ### 어지간히 데이터가 안터지면 사용 안 될 듯
    nyq = 0.5 * fs
    # print(nyq)
    low = lowcut / nyq
    # print(low)
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    # print(b)
    filtered_data = filtfilt(b, a, amplitude)

    # FFT 수행
    n = len(filtered_data)
    data_fft = fft(filtered_data)
    frequencies = fftfreq(n, d=(dt))

    # 양의 주파수 성분만을 사용
    ##kHZ를 위해 나누기 1000
    positive_frequencies = frequencies[:n // 2]
    positive_amplitudes = np.abs(data_fft[:n // 2]) * (2.0 / n)

    # 헤르츠(Hz)에서 킬로헤르츠(kHz)로 주파수 단위 변환
    positive_frequencies_kHz = positive_frequencies / 1000.0  # 모든 주파수 값을 1000으로 나눔

    # 피크 주파수 찾기
    # 찾는 범위 또한 kHz 단위로 변환
    lowcut_kHz = lowcut / 1000.0
    highcut_kHz = highcut / 1000.0
    peak_freq = find_peak_frequency(positive_frequencies_kHz, positive_amplitudes, lowcut_kHz, highcut_kHz)

    # 결과 출력
    # if peak_freq:
    #     print(f'Peak frequency within { lowcut_kHz} Hz and {highcut_kHz} Hz: {peak_freq:.2f} Hz')
    # else:
    #     print(f'No peak found within {lowcut_kHz} Hz and {highcut_kHz} Hz.')
    # 그래프로 결과 시각화
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(time, data, label='Original Signal')
    plt.plot(time, filtered_data, label='Filtered Signal', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(positive_frequencies_kHz, positive_amplitudes, label='FFT of Filtered Signal')
    if peak_freq:
        plt.axvline(x=peak_freq, color='green', linestyle='--', label=f'Peak Frequency: {peak_freq:.2f} kHz')
    plt.xlabel('Frequency (kHz)')
    plt.xlim([0, 30])
    plt.ylabel('Amplitude')
    plt.title('Frequency Domain')
    plt.legend()
    plt.tight_layout()
    # 파일 경로
    file_path = f'C:/Users/Win/Desktop/data_result/{column}/{column}_Frequency_norm.jpg'

    # 파일 경로에서 디렉터리 경로만 추출
    directory = os.path.dirname(file_path)

    # 해당 디렉터리가 존재하지 않는 경우 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(f'C:/Users/Win/Desktop/data_result/{column}/{column}_Frequency_norm.jpg')
    # plt.show()


    return peak_freq

# 예제 사용
peak_frequency =  process_and_find_peak
