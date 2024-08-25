import pandas as pd
import matplotlib.pyplot as plt
from lib.FFT_DATA import process_and_find_peak,process_and_find_peak
from lib.Reading_Excel_Making_DataFrame  import create_custom_df,plot_data,find_peaks_in_dataframe
from lib.grid_plane import create_and_transpose_df,create_graph_2D,split_name


##데이터 읽어옴
file_path = "C:/Users/Win/Desktop/data/example_data.xlsx"
custom_df = create_custom_df(file_path)

### peak 구해주고 dataframe으로 만들어줌
peaks_df=find_peaks_in_dataframe(custom_df, 300.0, 30000.0, 2)

peaks_df.to_excel('peaks_data.xlsx')

# 'avg'로 끝나는 열 이름만 필터링
avg_columns = [col for col in peaks_df.columns if col.endswith('avg')]
filtered_df = peaks_df[avg_columns]

##빈 DF 생성
num_rows = 18
num_cols = 13
result_KZ=create_and_transpose_df(num_rows, num_cols)
print(result_KZ)

# peaks_df의 피크 값 할당
for column in filtered_df.columns:
    alpha, num = split_name(column)
    if num in result_KZ.index and alpha in result_KZ.columns:
        result_KZ.loc[num, alpha] = peaks_df.loc['Peak', column]

print(result_KZ)
create_graph_2D(result_KZ)


###
print(custom_df)
print(peaks_df)
print(filtered_df)