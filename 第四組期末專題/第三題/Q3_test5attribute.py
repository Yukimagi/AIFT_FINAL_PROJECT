#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path='top200_training.xls'
#記得pip install xlrd
df = pd.read_excel(csv_file_path)
#df.head()


# In[ ]:


from itertools import product
from itertools import combinations


# 將年月轉換為日期型數據
df['年月'] = pd.to_datetime(df['年月'], format='%Y%m')

# 獲取所有不同的年份
all_years = ['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008']

# 開始的欄位
start_column = '1997'

# 開始欄位的索引
start_index = all_years.index(start_column)

# 創建一個字典來保存結果
year_combinations_dict = {}

# 生成所有可能的欄位組合
for i in range(1, len(all_years) - start_index + 1):
    for subset in combinations(all_years[start_index:], i):
        selected_columns = list(subset)
        # 將結果存入字典
        year_combinations_dict[len(year_combinations_dict) + 1] = selected_columns

# 打開一個檔案來保存結果
with open('year_combinations.txt', 'w') as file:
    # 遍歷字典，將結果寫入檔案
    for key, value in year_combinations_dict.items():
        file.write(f"Key: {key}, Selected year: {value}\n")
        # 在這裡進行你的分析或模型擬合等操作
        # ...
        file.write("\n" + "="*40 + "\n")

# 提示保存成功
print("Results saved to 'year_combinations.txt'")


# In[ ]:


from itertools import combinations
from sklearn.model_selection import train_test_split
import json
columns_comb1=['利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率']
columns_comb2=['本益比', '股價營收比', '利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率']
columns_comb3=['本益比', '股價營收比', '負債/淨值比', 'M流動比率', 'M營業利益成長率', 'M稅後淨利成長率']
columns_comb4=['股價營收比', '資產報酬率ROA', '負債/淨值比', 'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率']
columns_comb5=['本益比', 'M淨值報酬率─稅後', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率']

column_combinations_dict = {}
columns_combinations = [columns_comb1, columns_comb2, columns_comb3, columns_comb4, columns_comb5]

for i, columns_comb in enumerate(columns_combinations, start=1):
    column_combinations_dict[i] = columns_comb
print(column_combinations_dict)
# 將字典保存到.txt文件
file_path = 'column_combinations_GaussianNB_train.txt'

with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(column_combinations_dict, file, ensure_ascii=False)
print('Results saved to column_combinations_GaussianNB_train.txt')


# In[7]:


max_ret = 0
ans = {}
list_ans = []

# 選擇特定欄位分析
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes
from itertools import islice

selected_range = islice(year_combinations_dict.items(), 4094) #4094是所有可用來訓練的年份組合
cols=0

with open('GaussianNB_train_test_result.txt', 'a') as file:
    for key, selected_years in selected_range:
        
        # 選擇訓練集年分
        train_years = [int(year) for year in selected_years]
        # 選擇測試集年分
        test_years = [int(year) for year in all_years if year not in selected_years]
        print(f'總共有test year:{len(test_years)}')
        # 根據年份選擇訓練集
        train_data = df[df['年月'].dt.year.isin(train_years)]
        print(f"Number of rows in train_data: {len(train_data)}")
        
        for col in range (1,6):#取top5組合
            cols=cols+1
            list_ans = []
            # 指定要選擇的欄位
            selected_columns = column_combinations_dict[col]
            list_ans.append(selected_columns)

            # 選擇指定的欄位
            features_to_scale_train = train_data[selected_columns]

            scaler = StandardScaler()
            scaled_features_train = scaler.fit_transform(features_to_scale_train)

            # 將標準化後的特徵資料轉換為 DataFrame
            df_feat_train = pd.DataFrame(scaled_features_train, columns=features_to_scale_train.columns)
            
            print(df_feat_train.head())
            
            #到這邊會把"column_combinations.txt"從key1印到key16383(標準化後的資料) head->只有xlsx的前5列
        # --------------------------------------------------------------------------------------------------
            from sklearn.model_selection import train_test_split
            # 將資料分成訓練組及測試組
            X_train = df_feat_train
            print(X_train) #印出key1所選的屬性的數值(全部 0-199)
            y_train = train_data['ReturnMean_year_Label']
            print(y_train) #印出1997全部'ReturnMean_year_Label'

            X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.3, random_state=101) #200*0.3=60(test_size)
            print(X_train, y_train) #train_size:200-60=140 被挑為train的X(屬性的數值) y(ReturnMean_year_Label)
        
        #----------------------------------------------------------------------------------------------------
            # 使用高斯朴素貝葉斯演算法
            clf = GaussianNB()  # Gaussian Naive Bayes model
            clf.fit(X_train, y_train)
        #----------------------------------------------------------------------------------------------------
            predict_pre5 = {}
        #----------------------------------------------------------------------------------------------------
            for year in test_years:

                # 選擇特定年份的 test_data
                test_data = df[df['年月'].dt.year == year]
                print(year)
                print(f"Number of rows in test_data: {len(test_data)}")

                # 選擇指定的欄位
                features_to_scale_test = test_data[selected_columns]
                
                # 使用之前訓練好的標準化物件進行標準化
                scaled_features_test = scaler.transform(features_to_scale_test)

                # 將標準化後的特徵資料轉換為 DataFrame
                df_feat_test = pd.DataFrame(scaled_features_test, columns=features_to_scale_test.columns)
                
                # 使用已經訓練好的模型進行預測 Gaussian Naive Bayes model
                predictions_new = clf.predict(df_feat_test)

                #----------------------------------------------------------------------------------------------------
                # 測試高斯朴素貝葉斯演算法的好壞
                from sklearn.metrics import classification_report, confusion_matrix

                # 將實際類別分為真正例（True Positive）、真負例（True Negative）、偽正例（False Positive）和偽負例（False Negative）
                print('confusion_matrix:')
                print(confusion_matrix(test_data['ReturnMean_year_Label'], predictions_new))

                # 模型的精確度、召回率、F1分數和支持數等指標，用來評估模型對於每個類別的預測性能。
                print('classification_report')
                print(classification_report(test_data['ReturnMean_year_Label'], predictions_new))

                # 比較預測結果
                accuracy_new = clf.score(df_feat_test, test_data['ReturnMean_year_Label'])
                print(f'新數據的預測準確率: {accuracy_new}')
                #----------------------------------------------------------------------------------------------------

                # Assuming clf is your trained Gaussian Naive Bayes classifier

                # Get the probability estimates for the positive class (1)
                probability_positive_class = clf.predict_proba(df_feat_test)[:, 1]

                # Sort the instances based on the probability in descending order
                sorted_indices = np.argsort(probability_positive_class)[::-1]

                # Select the top 5 instances where the predicted label is 1
                selected_indices = [index for index in sorted_indices if predictions_new[index] == 1][:5]

                #----------------------------------------------------------------------------------------------------

                # 將選取的索引轉換成列表
                selected_indices_list = list(selected_indices)

                # 打印預測值為1的前5筆索引
                print("預測值為1的前5筆索引:")
                print(selected_indices_list)
                select = test_data
                
                # 獲取股票名稱
                selects_stock = select.iloc[selected_indices_list]['簡稱']
                print("獲取股票名稱:")
                print(selects_stock)
                
                # 獲取股票return
                selects_ret = select.iloc[selected_indices_list]['Return']
                print("獲取股票return:")
                print(selects_ret)
                
                # 將獲取的 股票名稱 和 股票return 寫進檔案
                with open(f'selected_stock_GaussianNB_train/{col}.txt', 'a') as stockfile: #col=key
                    stockfile.write(selects_stock.to_string())
                    stockfile.write('\n')
                    stockfile.write(selects_ret.to_string())
                # 計算return
                stock_returns = (selects_ret.mean() / 100) + 1 #.mean()計算數據的平均值
                portfolio_returns = stock_returns
                print(portfolio_returns)
                predict_pre5[year] = portfolio_returns
                # print(stock_returns)

            sum2 = 1

            #-----------------------------------------------------------predict_pre5
            print(predict_pre5)
            for k in predict_pre5.keys():  # 修改這裡以處理所有年度
                # 條件語句確保只有在預測回報率不為0的情況下才進行計算
                if not (predict_pre5[k]==0):
                    # 計算每年的複利，將每年的預測回報率乘以上一年的總複利，並將結果賦值給 sum2
                    sum2=predict_pre5[k]*sum2
            print(f'年均化複利為:{sum2/int(len(test_years))}')
            #----------------------------------------------------
            print(f'ori sum2:{sum2}')
            print(f'test years have:{int(len(test_years))}')
            sum2=sum2/int(len(test_years))
            print(f'after sum2:{sum2}')
            
            list_ans.append(sum2) #list_ans=[key1選的屬性(本益比)+sum2]
            ans[col] = list_ans #ans[key]
            file.write(f'{col}:{list_ans}\n\n') #1:本益比 sum2
            file.flush()  # 強制將緩衝區內容寫入檔案
            if sum2 > max_ret:
                max_ret = sum2
                with open('max_GaussianNB_train.txt', 'w') as maxfile:
                    maxfile.write(f'max:train_year:{train_years},test_year:{test_years}:{cols}:{list_ans}\n\n')
                    maxfile.flush()  # 強制將緩衝區內容寫入檔案


# In[ ]:




