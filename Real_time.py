import pandas as pd
import matplotlib.pyplot as plt
from lib.FFT_DATA import process_and_find_peak,process_and_find_peak
from lib.Reading_Excel_Making_DataFrame  import create_custom_df,plot_data,find_peaks_in_dataframe
from lib.grid_plane import create_and_transpose_df,create_graph_2D,split_name
from lib.Reading_Excel_Making_DataFrame import ensure_directory_exists
import pandas as pd
from lib.kHZtoH import main_calculate_H
from lib.draw_graph_Interpolation import make_graph_interpolation
from lib.Deep_shallow import deep_shallow

###데이터 읽어오고 해주는애고

###데이터 불러옴

##데이터 읽어옴
# file_path = "C:/Users/Win/Desktop/data/example_data.xlsx"
# custom_df = create_custom_df(file_path)
#
# ### peak 구해주고 dataframe으로 만들어줌
# peaks_df,peaks_sep_df=find_peaks_in_dataframe(custom_df, 300.0, 30000.0, 2)
#
# peaks_df.to_excel('peaks_data.xlsx')
#
#
# ####일반 데이터 알림
# # 'avg'로 끝나는 열 이름만 필터링
# avg_columns_normal = [col for col in peaks_df.columns if col.endswith('avg')]
# filtered_df = peaks_df[avg_columns_normal]
# print(filtered_df)
#
#
# ##빈 DF 생성
# num_rows = 19
# num_cols = 15
# result_KZ=create_and_transpose_df(num_rows, num_cols)
# print(result_KZ)
#
#
# # peaks_df의 피크 값 할당(norm)
# for column in filtered_df.columns:
#     alpha, num = split_name(column)
#     if num in result_KZ.index and alpha in result_KZ.columns:
#         result_KZ.loc[num, alpha] = peaks_df.loc['Peak', column]
#
#
# ##분리된 데이터 필터링
# avg_columns_sep = [col for col in peaks_sep_df.columns if col.endswith('avg')]
# filtered_df_sep = peaks_sep_df[avg_columns_sep]
# print(filtered_df_sep)
#
# ###새 데이터 만듬
# result_KZ_sep=create_and_transpose_df(num_rows, num_cols)
#
# ###여기까지는 잘됨
# # peaks_df의 피크 값 할당(sep)
# for column in filtered_df_sep.columns:
#     alpha, num = split_name(column)
#     if num in result_KZ_sep.index and alpha in result_KZ_sep.columns:
#         result_KZ_sep.loc[num, alpha] = peaks_sep_df.loc['Peak', column]
#
#
# print(result_KZ)
# print(result_KZ_sep)
#
# # CSV 파일로 저장
# result_KZ.to_csv('C:/Users/Win/Desktop/data_result/DATA_CSV/result_KZ.csv', index=True)  # index=False는 인덱스를 파일에 포함하지 않겠다는 옵션입니다.
# result_KZ_sep.to_csv('C:/Users/Win/Desktop/data_result/DATA_CSV/result_KZ_sep.csv', index=True)  # index=False는 인덱스를 파일에 포함하지 않겠다는 옵션입니다.
#
# # Excel 파일로 저장
# result_KZ.to_excel('C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ.xlsx', index=True, engine='openpyxl')  # engine='openpyxl'은 Excel 파일을 생성하기 위해 필요합니다.
# result_KZ_sep.to_excel('C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ_sep.xlsx', index=True, engine='openpyxl')  # engine='openpyxl'은 Excel 파일을 생성하기 위해 필요합니다.
# ### 아왜 미친듯이 출력됨?
# # create_graph_2D(result_KZ)
# # create_graph_2D(result_KZ_sep)
#
# main_calculate_H('C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ.xlsx','C:/Users/Win/Desktop/data_result/DATA_xlsx/result_H.xlsx')
# main_calculate_H('C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ_sep.xlsx','C:/Users/Win/Desktop/data_result/DATA_xlsx/result_H_sep.xlsx')
# ###데이터 불러옴

###우선 frez 그리기
###N=몇배 ,Nodata=특정값(아무것도없을때의 값)
# filepath_frz='C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ.xlsx'
# directory1='C:/Users/Win/Desktop/data_result/graph_dat/KZ_jpg'
# directory2='C:/Users/Win/Desktop/data_result/graph_dat/KZ_interpolation_jpg'
# make_graph_interpolation(30, 10,filepath_frz,directory1,directory2)


filepath_frz='C:/Users/Win/Desktop/data_result/DATA_xlsx/result_KZ.xlsx'
filtered_output_file_path='C:/Users/Win/Desktop/data_result/DATA_xlsx/result_Where.xlsx'
deep_shallow(filepath_frz,filtered_output_file_path)

directory1='C:/Users/Win/Desktop/data_result/graph_dat/where_jpg'
directory2='C:/Users/Win/Desktop/data_result/graph_dat/where_interpolation_jpg'
###N=몇배 ,Nodata=특정값(아무것도없을때의 값) 여기서 정상 구분은 0
make_graph_interpolation(30, 0,filtered_output_file_path,directory1,directory2)

### 구분
