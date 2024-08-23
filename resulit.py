import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import griddata


# 파일 경로 및 파일 이름 지정.
file_path = 'C:/Users/Win/Desktop/detect/E4.txt'
df = pd.read_csv(file_path, delimiter='\t', names=['X', 'Y', 'H'])
# 데이터 로드
data = np.loadtxt(file_path, delimiter='\t')

# Scatter Plot 생성
plt.figure(figsize=(10, 8))

# 그래프 1: Scatter Plot 생성 (cmap='YlGnBu'로 설정)
scatter = plt.scatter(df['X'], df['Y'], c=df['H'], cmap='YlGnBu', s=100, edgecolor='black')

# 색상 바 추가
cbar = plt.colorbar(scatter)
cbar.set_label('Sample Sequence', fontsize=12)

# X축과 Y축 레이블 설정
plt.xticks(np.arange(1, 19))
plt.yticks(ticks=np.arange(1, 15), labels=range(15, 1,-1))

# Y축 방향을 반대로 설정 (이미지와 동일하게)
plt.gca().invert_yaxis()

# 타이틀 설정
plt.title('Training Samples', fontsize=16)


# X 자 마커로 데이터가 없는 위치 표시
missing_data = df[df['H'] == 0]  # H 값이 0인 위치를 데이터가 없는 것으로 간주
plt.scatter(missing_data['X'], missing_data['Y'], color='red', marker='x', s=100, label='No Data')

# 범례 추가
plt.legend()

# 그래프 출력
plt.grid(True)
plt.show()



df = pd.read_csv('C:/Users/Win/Desktop/detect/grid_data_3.txt',sep='\t')


# 데이터 확인 (열 이름 확인)
print(df.columns)  # ['X', 'Y', 'H']가 출력되어야 함
print(df)







# Scatter Plot 생성
plt.figure(figsize=(10, 8))

# 그래프 1: Scatter Plot 생성 (cmap='YlGnBu'로 설정)
scatter = plt.scatter(df['X'], df['Y'], c=df['H'], cmap='YlGnBu', s=100, edgecolor='black')

# 색상 바 추가
cbar = plt.colorbar(scatter)
cbar.set_label('Sample Sequence', fontsize=12)

# X축과 Y축 레이블 설정
plt.xticks(np.arange(1, 19))
plt.yticks(ticks=np.arange(1, 15), labels=range(15, 1,-1))

# Y축 방향을 반대로 설정 (이미지와 동일하게)
plt.gca().invert_yaxis()

# 타이틀 설정
plt.title('Training Samples', fontsize=16)


# X 자 마커로 데이터가 없는 위치 표시
missing_data = df[df['H'] == 0]  # H 값이 0인 위치를 데이터가 없는 것으로 간주
plt.scatter(missing_data['X'], missing_data['Y'], color='red', marker='x', s=100, label='No Data')

# 범례 추가
plt.legend()

# 그래프 출력
plt.grid(True)
plt.show()


# 피벗 테이블로 데이터 변환 (X, Y 좌표에 따른 H 값)
grid_data = df.pivot(index="Y", columns="X", values="H")

# 그리드 설정
plt.figure(figsize=(10, 8))

sns.heatmap(grid_data, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Frequency (kHz)'}, vmin=0, vmax=12)

# X축과 Y축 레이블 설정
plt.xticks(ticks=np.arange(1, 19), labels=range(1, 19))
# plt.yticks(ticks=np.arange(1, 15), labels=range(15, 1,-1))
plt.gca().invert_yaxis()

# 축 레이블 설정
plt.title('Grid Interpolation (X, Y, H)')
plt.xlabel('X-grid')
plt.ylabel('Y-grid')

# 그래프 출력
plt.show()






###보간 그래프 작성

# 보간을 위한 그리드 생성
grid_x, grid_y = np.mgrid[1:19:100j, 1:15:100j]  # 100x100 그리드로 보간

# 보간 수행 (양선형 보간법 사용)
grid_z = griddata((df['X'], df['Y']), df['H'], (grid_x, grid_y), method='linear')

# 보간 결과 시각화
plt.figure(figsize=(10, 8))
plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='YlGnBu')
plt.colorbar(label='H Value')


# 축 설정
plt.xticks(np.arange(0, 20))
plt.yticks(np.arange(0, 16))
# plt.gca().invert_yaxis()
plt.title('Interpolated Grid with Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()