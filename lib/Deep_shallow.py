import pandas as pd
import openpyxl

def calculate_height(frequency, vp):
    # Impact echo 식: h = 0.96 * Vp / (2 * f)
    h = 0.96 * vp / (2 * frequency)
    return h  # 높이 mm 단위

def filter_height(height, ranges):
    # 각 필터링 범위에 따른 값 반환
    a, b, c, d, e, f = ranges
    if a <= height <= b:
        return 0 ### 정상(Normal)부분
    elif c <= height <= d:
        return 1 ### Deep 부분
    elif e <= height <= f:
        return 2 ### Shallow 부분
    else:
        return 3 ### Poor compaction 부분


##받는 데이터 KHZ
####데이터 구분 방식
####구분 방식 접근 필요!!!!!
##저장되는 데이터 결과값은 0,1,2,3 정상(Normal)부분,Deep 부분,Shallow 부분,Poor compaction 부분
def deep_shallow(file_path,filtered_output_file_path):
    try:
        # 속도 Vp 입력 (m/s)
        vp = 4000  #### 설정된 속도 (m/s)

        # Step 1: 주파수 데이터를 사용해 result_H.xlsx 파일 생성
        # file_path = r'C:\Users\Win\Desktop\진단학회_Impact-eco\2_실습 데이터 및 Python 코드\result_KZ.xlsx'  ### 엑셀 파일 경로
        df = pd.read_excel(file_path, header=None, sheet_name='Sheet1')  # 데이터프레임으로 엑셀 파일 읽기

        # 빈 데이터프레임을 동일한 크기로 생성하여 결과를 저장
        height_df = pd.DataFrame(index=df.index, columns=df.columns)

        # 각 셀의 주파수 데이터를 가져와 높이 계산 후 저장 (첫 행과 첫 열은 무시)
        for i in df.index[1:]:  # 첫 번째 행 무시
            for j in df.columns[1:]:  # 첫 번째 열 무시
                frequency = df.at[i, j]
                if pd.notna(frequency) and isinstance(frequency, (int, float)):  # 빈 셀 및 숫자가 아닌 셀 무시
                    height = calculate_height(frequency, vp)
                    height_df.at[i, j] = round(height, 2)  # 소수점 두 자리로 반올림
                else:
                    height_df.at[i, j] = frequency  # 원본 데이터를 그대로 유지 (NaN 포함)

        # 첫 행과 첫 열의 값을 원본에서 복사하여 유지
        height_df.iloc[0, :] = df.iloc[0, :]
        height_df.iloc[:, 0] = df.iloc[:, 0]

        # 계산된 높이를 엑셀 파일로 저장 (원본과 동일한 위치에 저장)
        # output_file_path = r'C:\Users\Win\Desktop\진단학회_Impact-eco\2_실습 데이터 및 Python 코드\result_H.xlsx'  ### 저장할 엑셀 파일 경로
        # with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        #     height_df.to_excel(writer, index=False, header=False, sheet_name='Calculated_Heights')
        #
        # print("Height calculations saved successfully to", output_file_path)

        # Step 2: 생성된 result_H.xlsx 파일을 불러와 필터링 진행
        ranges = (5, 10, 15, 20, 25, 30)  ### 필터링 범위: a<=h<=b, c<=h<=d, e<=h<=f ###

        filtered_df = pd.DataFrame(index=height_df.index, columns=height_df.columns)

        # 각 셀의 h 값을 필터링 후 저장 (첫 행과 첫 열은 좌표이므로 무시)
        for i in height_df.index[1:]:  # 첫 번째 행 무시
            for j in height_df.columns[1:]:  # 첫 번째 열 무시
                height = height_df.at[i, j]
                if pd.notna(height) and isinstance(height, (int, float)):  # 빈 셀 및 숫자가 아닌 셀 무시
                    filtered_df.at[i, j] = filter_height(height, ranges)
                else:
                    filtered_df.at[i, j] = height  # 원본 데이터를 그대로 유지 (NaN 포함)

        # 첫 행과 첫 열의 값을 원본에서 복사하여 유지
        filtered_df.iloc[0, :] = height_df.iloc[0, :]
        filtered_df.iloc[:, 0] = height_df.iloc[:, 0]


        ##저장되는 데이터 결과값은 0,1,2,3

        # 필터링된 결과를 엑셀 파일로 저장
        # filtered_output_file_path = r'C:\Users\Win\Desktop\진단학회_Impact-eco\2_실습 데이터 및 Python 코드\result_H_fil.xlsx'  ### 저장할 엑셀 파일 경로
        with pd.ExcelWriter(filtered_output_file_path, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, header=False, sheet_name='Filtered_Heights')

        print("Filtered data saved successfully to", filtered_output_file_path)

    except Exception as e:
        print("An error occurred:", e)
