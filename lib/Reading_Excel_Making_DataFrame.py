import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.FFT_DATA import process_and_find_peak, process_and_find_peak_nograph

## 본 코드는 엑셀 파일을 읽고 데이터를 만드는 코드입니다.
## (엑셀 파일명만 넣어주면 됨)
##혹시 모르니까 행과 인덱스 확인할것
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

###데이터 그려주기
### 산점도로 그려줌
### (df,그리고자 하는 데이터 열 이름)으로 넣어주면 됨
def plot_data(df, column_name):
    # 데이터 선택
    data = df[column_name]

    # 막대 그래프
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    plt.scatter(data.index, data.values, color='red', marker='o', s=10)  # 산점도 생성, s는 점 크기
    # 선으로 데이터 포인트 연결
    plt.plot(data.index, data.values, color='blue', linestyle='-', linewidth=1)
    plt.xlabel('Index')  # x축 레이블
    plt.ylabel('Value')  # y축 레이블
    plt.title(f' Graph of {column_name}')  # 그래프 제목
    plt.grid(True)  # 그리드 표시
    plt.show()  # 그래프 표시

### 각자 평균 낼때 사용
##현재 사용하지 않음 참고용
def calculate_average_columns(df, base_name):
    # 패턴에 맞는 모든 열 선택
    pattern = f"{base_name}_"
    columns = [col for col in df.columns if col.startswith(pattern)]

    # 선택된 열에 대한 평균 계산
    df[f"{base_name}_avg"] = df[columns].mean(axis=1)

    return df

##peak 값으로 데이터 만들기

def find_peaks_in_dataframe(df, lowcut, highcut, order):
    # 피크 값을 저장할 새 데이터 프레임 생성
    peaks_df = pd.DataFrame(index=['Peak'], columns=df.columns)

    # df의 모든 열에 대해 반복
    for column in df.columns:
        # 현재 열의 데이터 추출
        data = df[column]

        # 데이터가 NaN만 포함하고 있는지 체크
        if data.isna().all():
            peak = np.nan
        else:
            # NaN 값 제거
            clean_data = data.dropna()

            # 피크 주파수 계산
            # 이를 위해 데이터 프레임을 생성 (임시로 하나의 열만 있는 DataFrame 사용)
            temp_df = pd.DataFrame(clean_data)
            temp_df.columns = ['Data']

            # 피크 찾기 함수 호출
            peak = process_and_find_peak_nograph(temp_df, lowcut=lowcut, highcut=highcut, order=order)

        # 피크 값을 peaks_df에 저장
        peaks_df.at['Peak', column] = peak

    return peaks_df


# 파일 경로 지정 및 함수 호출
file_path = "C:/Users/Win/Desktop/data/example_data.xlsx"
custom_df = create_custom_df(file_path)
# print(custom_df)

# print(pd.DataFrame(custom_df["B1_1"]))
peak=process_and_find_peak(pd.DataFrame(custom_df["D1_1"]), lowcut=300.0, highcut=30000.0, order=2)
# peaks_df = find_peaks_in_dataframe(custom_df, 300.0, 30000.0, 2)
# print(peaks_df)
# data_array = custom_df["A1_avg"].to_numpy()
# print(data_array)
# peak=process_and_find_peak(data_array, lowcut=300.0, highcut=30000.0, order=2)
# print(peak)

# column_name = 'B1_avg'
# plot_data(custom_df, column_name)