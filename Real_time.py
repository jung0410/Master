import pandas as pd
import matplotlib.pyplot as plt
from lib.FFT_DATA import process_and_find_peak,process_and_find_peak
from lib.Reading_Excel_Making_DataFrame  import create_custom_df,plot_data,find_peaks_in_dataframe
from lib.grid_plane import create_and_transpose_df,create_graph_2D,split_name
from lib.Reading_Excel_Making_DataFrame import ensure_directory_exists

###데이터 읽어오고 해주는애고

###데이터 불러옴

##데이터 읽어옴
file_path = "C:/Users/Win/Desktop/data/example_data.xlsx"
custom_df = create_custom_df(file_path)

### peak 구해주고 dataframe으로 만들어줌
peaks_df,peaks_sep_df=find_peaks_in_dataframe(custom_df, 300.0, 30000.0, 2)

peaks_df.to_excel('peaks_data.xlsx')


####일반 데이터 알림
# 'avg'로 끝나는 열 이름만 필터링
avg_columns_normal = [col for col in peaks_df.columns if col.endswith('avg')]
filtered_df = peaks_df[avg_columns_normal]
print(filtered_df)

##분리된 데이터 필터링
avg_columns_sep = [col for col in peaks_sep_df.columns if col.endswith('avg')]
filtered_df_sep = peaks_sep_df[avg_columns_normal]
print(filtered_df_sep)

##빈 DF 생성
num_rows = 19
num_cols = 15
result_KZ=create_and_transpose_df(num_rows, num_cols)
print(result_KZ)
result_KZ_sep=result_KZ

# peaks_df의 피크 값 할당(norm)
for column in filtered_df.columns:
    alpha, num = split_name(column)
    if num in result_KZ.index and alpha in result_KZ.columns:
        result_KZ.loc[num, alpha] = peaks_df.loc['Peak', column]

# peaks_df의 피크 값 할당(sep)
for column in filtered_df_sep.columns:
    alpha, num = split_name(column)
    if num in result_KZ_sep.index and alpha in result_KZ_sep.columns:
        result_KZ_sep.loc[num, alpha] = peaks_df.loc['Peak', column]


print(result_KZ)
print(result_KZ_sep)
create_graph_2D(result_KZ)


###데이터 불러옴
