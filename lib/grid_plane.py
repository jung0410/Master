import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###데이터의 표시 방식을 연구해주는 코드입니다.
def create_and_transpose_df(num_rows, num_cols):
    # 행 이름 생성: 'A', 'B', ..., 'R' (알파벳 18개)
    row_names = [chr(i) for i in range(ord('A'), ord('A') + num_rows)]

    # 열 이름 생성: '1', '2', ..., '13'
    col_names = [str(i) for i in range(1, num_cols + 1)]



    # 모든 값이 0인 데이터 프레임 생성
    # df = pd.DataFrame(np.zeros((num_rows, num_cols)), index=row_names, columns=col_names)


    ## 행열 바꾸려면
    df = pd.DataFrame(np.nan, index=col_names, columns=row_names)

    # 데이터 프레임 출력
    print(df)
    return(df)

###그래프를 그려줌
def create_graph_2D(df):
    # 행 이름 생성: 'A', 'B', ..., 'R' (알파벳 18개)
    row_names = [chr(i) for i in range(ord('A'), ord('A') + num_rows)]
    # 열 이름 생성: '1', '2', ..., '13'
    col_names = [str(i) for i in range(1, num_cols + 1)]
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.viridis  # 색상 맵 설정
    cmap.set_bad('white', 1.)  # NaN 값은 흰색으로 표시
    # 축에 레이블 추가
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns)
    plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)
    plt.imshow(df, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.colorbar()  # 색상 막대 추가
    plt.grid(True)
    plt.title('Data Frame Visualization')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

# _avg를 제거하고 알파벳과 숫자를 분리하는 함수
def split_name(column_name):
    clean_name = column_name.replace('_avg', '')  # _avg 제거
    alpha_part = ''.join(filter(str.isalpha, clean_name))  # 알파벳만 추출
    num_part = ''.join(filter(str.isdigit, clean_name))  # 숫자만 추출
    return alpha_part, num_part



num_rows = 19
num_cols = 15
df=create_and_transpose_df(num_rows, num_cols)

df.loc['13', 'A'] = 50
df.loc['13', 'D'] = 30
# create_graph_2D(df)