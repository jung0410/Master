import pandas as pd
import openpyxl

def calculate_height(frequency, vp):
    # Impact echo 식: h = 0.96 * Vp / (2 * f)
    h = 0.96 * vp / (2 * frequency)
    return h  # 높이 mm 단위

def main_calculate_H(file_path,output_file_path):
    try:
        # 속도 Vp 입력 (m/s)
        vp = 4000  # 예시로 설정된 속도 (m/s)

        # 엑셀 파일에서 주파수 데이터를 불러옴
        # file_path = 'C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ.xlsx'  ### 엑셀 파일 경로를 입력
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
        # output_file_path = 'C:/Users/Win/Desktop/data_result/DATA_xlsx/result_H.xlsx'   ### 저장할 엑셀 파일 경로를 입력
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            height_df.to_excel(writer, index=False, header=False, sheet_name='Calculated_Heights')

        print("Height calculations saved successfully to", output_file_path)

    except Exception as e:
        print("An error occurred:", e)

