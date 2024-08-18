import pandas as pd
import numpy as np

# 무작위 데이터 생성
# 데이터 생성
data = []
for x in range(1, 19):  # X: 1 ~ 18
    for y in range(1, 15):  # Y: 1 ~ 14
        if x % 2 == 1 and y % 2 == 0:  # x와 y가 모두 홀수일 경우
            h = 0
        elif x % 2 == 0 and y % 2 == 1:  # x와 y가 모두 홀수일 경우
            h = 0
        elif 3 <= x <= 6 and 3 <= y <= 6:
            h = np.random.uniform(3, 4)  # 2 ~ 3 사이의 무작위 실수
        elif 9 <= x <= 12 and 3 <= y <= 6:
            h = np.random.uniform(5, 6)
        elif 9 <= x <= 12 and 9 <= y <= 12:
            h = np.random.uniform(9, 10)
        elif 3 <= x <= 6 and 9 <= y <= 12:
            h = np.random.uniform(11, 12)
        else:
            h = np.random.uniform(1, 2)
        data.append([x, y, h])


# DataFrame 생성
df = pd.DataFrame(data, columns=['X', 'Y', 'H'])

# DataFrame을 txt 파일로 저장
file_path = 'C:/Users/Win/Desktop/detect/grid_data_2.txt'
df.to_csv(file_path, sep='\t', index=False)

print(f"Data saved to {file_path}")



# 무작위 데이터 생성
# 데이터 생성
data = []
for x in range(1, 19):  # X: 1 ~ 18
    for y in range(1, 15):  # Y: 1 ~ 14
        if 3 <= x <= 6 and 3 <= y <= 6:
            h = np.random.uniform(3, 4)  # 2 ~ 3 사이의 무작위 실수
        elif 9 <= x <= 12 and 3 <= y <= 6:
            h = np.random.uniform(5, 6)
        elif 9 <= x <= 12 and 9 <= y <= 12:
            h = np.random.uniform(9, 10)
        elif 3 <= x <= 6 and 9 <= y <= 12:
            h = np.random.uniform(11, 12)
        elif x % 2 == 1 and y % 2 == 0:  # x와 y가 모두 홀수일 경우
            h = 0
        elif x % 2 == 0 and y % 2 == 1:  # x와 y가 모두 홀수일 경우
            h = 0
        else:
            h = np.random.uniform(1, 2)
        data.append([x, y, h])


# DataFrame 생성
df = pd.DataFrame(data, columns=['X', 'Y', 'H'])

# DataFrame을 txt 파일로 저장
file_path = 'C:/Users/Win/Desktop/detect/grid_data_3.txt'
df.to_csv(file_path, sep='\t', index=False)

print(f"Data saved to {file_path}")
