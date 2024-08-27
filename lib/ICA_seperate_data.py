import numpy as np
import PyEMD
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fftpack import fft, fftfreq
from PyEMD.EMD import EMD
import os
from lib.FFT_DATA import process_and_find_peak_nograph


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
        ###여기서 오류 발생
        # max_peak_index = [np.argmax(filtered_amplitudes[peaks])]
        peak_freq = filtered_frequencies[peaks[np.argmax(filtered_amplitudes[peaks])]]
        max_peak_index = filtered_amplitudes[peaks[np.argmax(filtered_amplitudes[peaks])]]
        return peak_freq,max_peak_index
    else:
        return None , None




def seperate_and_find_peak(data,column, lowcut=300.0, highcut=30000.0, order=2):
    # time = data.iloc[:, 0]
    signal=data.iloc[:, 0]

    ####문제 발생### 매우 중요
    ### 이게 3초짜리 정보인지 6초짜리 정보인지 어덯게암??? 야 이게 중대 문제
    ###샘플사이즈 400000
    length_of_data = len(signal)
    Hz_given=400000
    sample_given=3000
    # print(length_of_data)
    time = np.arange(0, sample_given/ Hz_given , 1 / Hz_given)
    time=pd.DataFrame(time)

    # DC offset 제거 (신호의 평균값 빼기)
    signal = signal - np.mean(signal)

    # 샘플링 주기를 계산
    sampling_interval= time.loc[1, 0] - time.loc[0, 0]
    # print(sampling_interval)
    time=np.array(time)

    # print(signal)
    time=np.array(time)
    time=time.flatten()
    # print(time)

    # print(signal)
    # 두 번째 줄의 값(인덱스 1)을 NumPy 배열로 변환
    signal = np.array(signal)

    # signal = signal['0'].values
    # print(signal)


    # 샘플링 주파수를 계산
    fs = 1.0 / sampling_interval

    nyq = 0.5 * fs
    # print(nyq)
    low = lowcut / nyq
    # print(low)
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    # print(b)
    signal = filtfilt(b, a, signal)

    # EMD 객체 생성
    emd = EMD()
    # print(emd)
    # IMF 분해 수행
    imfs = emd.emd(signal, time)
    # IMF 시각화 (시간 도메인)
    n_imfs = imfs.shape[0]
    sampling_interval = time[1] - time[0]
    num = 0
    # 신호 및 IMF 시각화 (4개씩 끊어서)
    for start in range(0, n_imfs, 4):

        plt.figure(figsize=(12, 8))

        # 원본 신호 표시
        if start == 0:
            plt.subplot(5, 1, 1)  # 첫 번째 그래프에 원본 신호 포함
            plt.plot(time, signal, 'r')
            plt.title("Original Signal")

        for i in range(start, min(start + 4, n_imfs)):
            plt.subplot(5, 1, i - start + 2)
            plt.plot(time, imfs[i], 'g')
            plt.title(f"IMF {i + 1}")

        plt.tight_layout()

        file_path = f'C:/Users/Win/Desktop/data_result/{column}/seperate/sep_{column}_{num}_Domain_norm.jpg'

        # 파일 경로에서 디렉터리 경로만 추출
        directory = os.path.dirname(file_path)

        # 해당 디렉터리가 존재하지 않는 경우 생성
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(f'C:/Users/Win/Desktop/data_result/{column}/seperate/sep_{column}_{num}_Domain_norm.jpg')
        # plt.show()
        num +=1



    ###주파수임
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
            # print(sampling_interval)
            # 주파수 벡터 계산
            freq = fftfreq(len(imfs[i]), d=sampling_interval)
            n=len(imfs[i])

            # 양수 주파수만 표시
            ###데이터 비교 필요!!1
            ####여기 뭔가 이상함
            positive_freq_idx = freq > 0
            # print(positive_freq_idx)
            # positive_freq_idx = freq
            positive_freq = freq[positive_freq_idx]/1000 # Hz -> kHz로 변환
            # print(positive_freq )
            positive_amplitude = np.abs(fft_values[positive_freq_idx])* (2.0 / n)

            # positive_freq_idx = freq[:n // 2]
            # ###에러 주의!!
            # positive_freq = positive_freq_idx / 1000
            # positive_amplitude = np.abs(fft_values[:n // 2]) * (2.0 / n)

            # 최대 진폭을 가지는 주파수 찾기
            # max_amp_index = np.argmax(positive_amplitude)
            # max_amp_freq = positive_freq[max_amp_index]
            # max_amp_value = positive_amplitude[max_amp_index]

            # 찾는 범위 또한 kHz 단위로 변환
            lowcut_kHz = lowcut / 1000.0
            # print(lowcut_kHz)
            highcut_kHz = highcut / 1000.0
            # print(highcut_kHz )
            max_amp_freq,max_amp_value = find_peak_frequency(positive_freq, positive_amplitude, lowcut_kHz, highcut_kHz)
            # print(max_amp_freq)
            # print(max_amp_value)


            # 최대 주파수 출력
            # print(f"IMF {i + 1}: 최대 진폭을 가지는 주파수 = {max_amp_freq:.2f} kHz")
            # 전체 IMF 중 최대 주파수를 추적
            if max_amp_value !=None and max_amp_value > overall_max_amp:
                overall_max_amp = max_amp_value
                overall_max_freq = max_amp_freq

            # 그래프 그리기
            plt.plot(positive_freq, positive_amplitude, label='Frequency Spectrum')
            if max_amp_freq is not None:
                plt.scatter(max_amp_freq, max_amp_value, color='red', marker='o',
                            label=f'Max Frequency: {max_amp_freq:.2f} kHz')

            plt.title(f"IMF {i + 1} - Frequency Domain")
            plt.xlabel("Frequency (kHz)")
            plt.ylabel("Amplitude")
            ##x축 몇까지
            plt.xlim([0, 30])
            plt.legend()

        plt.tight_layout()
        # 파일 경로
        file_path = f'C:/Users/Win/Desktop/data_result/{column}/seperate/sep_{column}_{i}_Frequency_norm.jpg'

        # 파일 경로에서 디렉터리 경로만 추출
        directory = os.path.dirname(file_path)

        # 해당 디렉터리가 존재하지 않는 경우 생성
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(f'C:/Users/Win/Desktop/data_result/{column}/seperate/sep_{column}_{i}_Frequency_norm.jpg')
        # plt.show()

    # print(f'H치 최대값{overall_max_freq}')
    return overall_max_freq






# 예제 사용
# peak_frequency =  process_and_find_peak

def create_custom_df(file_path):
    # 엑셀 파일 읽기
    data = pd.read_excel(file_path, header=None)

    # 34번째 열(Col 34, 인덱스는 33)의 데이터를 기반으로 열 이름 설정
    col_data = data.iloc[33:, 2]  # 34번째 행은 인덱스 33
    # print(col_data)
    # 열 이름에 대한 중복 처리
    name_count = {}

    # 새로운 DataFrame을 위한 빈 딕셔너리 생성
    new_data = {}
    col_name_past=f"noname"
    name_number=1

    # 각 col_data 항목에 대해 base_name의 데이터를 추가
    for i, col_name in enumerate(col_data):

        # i+1 행의 F열 이후 데이터를 가져옴
        row_data = data.iloc[33 + i, 5:]
        if col_name ==col_name_past:
            new_col_name = f"{col_name}_{name_number}"
            name_number+=1
        else:
            name_number = 1
            new_col_name = f"{col_name}_{name_number}"
            name_number+=1

        # 해당 col_name을 열 이름으로 사용하여 데이터 저장
        col_name_past = col_name
        new_data[new_col_name] = row_data.tolist()

    df=pd.DataFrame(new_data)

    ### 평균값 뽑아주는 구간
    # col_data에서 고유한 기준 이름(base_name)들 추출
    unique_bases = col_data.unique()

    # 각 unique_base에 대해 평균 계산
    for base_name in unique_bases:
        # 해당 base_name으로 시작하는 모든 열을 찾아서 평균 계산
        pattern = f"{base_name}_"
        columns = [col for col in df.columns if col.startswith(pattern)]

        if columns:
            # 선택된 열에 대한 평균 계산 및 새 열 추가
            df[f"{base_name}_avg"] = df[columns].mean(axis=1)

    # print(pd.DataFrame(df))
    # 새 DataFrame 생성
    return pd.DataFrame(df)
file_path = "C:/Users/Win/Desktop/data/example_data.xlsx"
custom_df = create_custom_df(file_path)
peak1=process_and_find_peak_nograph(pd.DataFrame(custom_df["A1_avg"]),"D1_2", lowcut=300.0, highcut=30000.0, order=2)
peak2=seperate_and_find_peak(pd.DataFrame(custom_df["A1_avg"]),"D1_2", lowcut=300.0, highcut=30000.0, order=2)
