import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
import os



###N은 배수
def make_graph_interpolation(n, Nodata,file_path,directory1,directory2):
    # 데이터 프레임 로드
    df = pd.read_excel(file_path, index_col=0)  # 첫 번째 열을 인덱스로 사용
    print(df.head())  # 데이터 프레임의 처음 몇 줄을 출력


    # 원본 데이터 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap="viridis", cbar=True, annot=True, fmt=".2f", annot_kws={'size': 8}, square=True,
                mask=pd.isna(df))
    plt.title('kHz of Excel Data')
    plt.grid(True)
    plt.yticks(rotation=360)
    plt.savefig(directory1)
    plt.show()
    # 해당 디렉터리가 존재하지 않는 경우 생성
    # if not os.path.exists(directory1):
    #     os.makedirs(directory1)



    # NaN을 0으로 채우기
    df_filled = df.fillna(Nodata)

    df = pd.read_excel(file_path, index_col=0)  # 첫 번째 열을 인덱스로 사용

    # 좌표 및 값 추출
    x = np.arange(df_filled.shape[1])
    y = np.arange(df_filled.shape[0])
    X, Y = np.meshgrid(x, y)
    values = df_filled.values.flatten()

    # 그리드 데이터 생성 (30배 보간)
    grid_x, grid_y = np.mgrid[0:df_filled.shape[0]:complex(0, df_filled.shape[0] * 30),
                     0:df_filled.shape[1]:complex(0, df_filled.shape[1] * 30)]

    # 보간 수행
    grid_z = griddata((Y.ravel(), X.ravel()), values, (grid_x, grid_y), method='cubic')

    # 보간 결과 시각화
    plt.figure(figsize=(10, 8))

    sns.heatmap(grid_z, cmap="viridis", cbar=True, annot=False, fmt=".2f", annot_kws={'size': 8}, square=True)
    plt.title('Interpolated Excel Data Using Griddata')
    plt.grid(True)
    # n=30
    n_x = n * 14 / 15
    n_y = n * 17 / 18

    # x축 눈금 위치와 레이블 설정 (보정된 위치 설정)
    # 열 레이블에 빈 문자열 추가
    columns_with_blanks_x = [''] + list(df.columns) + ['']
    xticks_positions = np.arange(len(columns_with_blanks_x)) * n_x  # 열 개수에 맞춰 위치 설정, +1은 맨 마지막 간격을 위함
    xticks_positions = xticks_positions - 0.5 * n_x  # 모든 위치를 0.5n만큼 이동

    plt.xticks(xticks_positions, columns_with_blanks_x)  # x축 눈금 위치와 레이블 설정

    columns_with_blanks_y = [''] + list(df.index) + ['']
    yticks_positions = np.arange(len(columns_with_blanks_y)) * n_y  # 열 개수에 맞춰 위치 설정, +1은 맨 마지막 간격을 위함
    yticks_positions = yticks_positions - 0.5 * n_y  # 모든 위치를 0.5n만큼 이동

    plt.yticks(yticks_positions, columns_with_blanks_y)  # x축 눈금 위치와 레이블 설정

    # 해당 디렉터리가 존재하지 않는 경우 생성
    # if not os.path.exists(directory2):
    #     os.makedirs(directory2)

    plt.xticks(rotation=360)

    plt.savefig(directory2)
    plt.show()
