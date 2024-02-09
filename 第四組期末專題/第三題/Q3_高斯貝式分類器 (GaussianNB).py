#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path='top200_1997.xlsx'

df=pd.read_excel(csv_file_path)
df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

columns_to_exclude=['證券代碼','簡稱','年月','ReturnMean_year_Label']

#移除沒有要用到的欄位
features_to_scale=df.drop(columns=columns_to_exclude)
#創建了一個標準化的實例
scaler=StandardScaler()
#擬合（fit）標準化器，計算特徵的均值和標準差->對均值和標準差transform，實現標準化->將標準化後的特徵存到scaled_features
scaled_features=scaler.fit_transform(features_to_scale)

# 將標準化後的特徵資料(scaled_features)轉換為 DataFrame
df_feat = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
df_feat.head()


# In[ ]:


from itertools import combinations
# 所有的欄位
all_columns = ['市值(百萬元)', '收盤價(元)_年', '本益比', '股價淨值比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA', '營業利益率OPM', '利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率', 'M存貨週轉率 (次)', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率']

# 開始的欄位
start_column = '本益比'

# 開始欄位的索引
start_index = all_columns.index(start_column)

# 創建一個字典來保存結果
column_combinations_dict = {}

# 生成所有可能的欄位組合
for i in range(1, len(all_columns) - start_index + 1):
    for subset in combinations(all_columns[start_index:], i):
        selected_columns = list(subset)
        # 將結果存入字典
        column_combinations_dict[len(column_combinations_dict) + 1] = selected_columns

# 打開一個檔案來保存結果
with open('column_combinations.txt', 'w') as file:
    # 遍歷字典，將結果寫入檔案
    for key, value in column_combinations_dict.items():
        file.write(f"Key: {key}, Selected Columns: {value}\n")
        # 在這裡進行你的分析或模型擬合等操作
        # ...
        file.write("\n" + "="*40 + "\n")

# 提示保存成功
print("Results saved to 'column_combinations.txt'")


# In[ ]:


max_ret = 0
ans = {}
list_ans = []

# 選擇特定欄位分析
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes

#選5隻股票，max報酬率:9903:[['利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率', 'M應收帳款週轉次', 'M營業利益成長率', 'M稅後淨利成長率'], 6.816019081885784]
#選10隻股票，
with open('test1.txt', 'a') as file:
    for col in range(1, 16384): #key1到key16383 
        list_ans = []
        # 指定要選擇的欄位
        selected_columns = column_combinations_dict[col]
        list_ans.append(selected_columns)

        # 選擇指定的欄位
        features_to_scale = df[selected_columns]

        # scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_to_scale)

        # 將標準化後的特徵資料轉換為 DataFrame
        df_feat = pd.DataFrame(scaled_features, columns=selected_columns)
        print(df_feat.head())
        #到這邊會把"column_combinations.txt"從key1印到key16383(標準化後的資料) head->只有xlsx的前5列
        # --------------------------------------------------------------------------------------------------

        # 將資料分成訓練組及測試組
        from sklearn.model_selection import train_test_split

        X = df_feat.dropna() #drop Not a Number
        print(X) #印出key1所選的屬性的數值(全部 0-199)
        y = df.loc[X.index, 'ReturnMean_year_Label']
        print(y) #印出1997全部'ReturnMean_year_Label'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #200*0.3=60(test_size)
        print(X_train, y_train) #train_size:200-60=140 被挑為train的X(屬性的數值) y(ReturnMean_year_Label)
        
        #----------------------------------------------------------------------------------------------------
        # 使用高斯朴素貝葉斯演算法
        clf = GaussianNB()  # Gaussian Naive Bayes model
        clf.fit(X_train, y_train)
        # --------------------------------------------------------------------------------------------------
        predict_pre5 = {}
        # --------------------------------------------------------------------------------------------------
        for year in range(1998, 2010):
        #for year in range(1998, 1999):
            csv_file_path = f'top200_{year}.xlsx'
            new = pd.read_excel(csv_file_path)

            # 排除指定的欄位
            df_temp = new.drop(columns=columns_to_exclude) #columns_to_exclude=['證券代碼','簡稱','年月','ReturnMean_year_Label']
            # 選擇指定的欄位
            features_to_scale_new = df_temp[selected_columns]

            # 使用之前訓練好的標準化物件進行標準化
            scaled_features_new = scaler.fit_transform(features_to_scale_new)

            # 將標準化後的特徵資料轉換為 DataFrame
            df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)

            # 使用已經訓練好的模型進行預測 Gaussian Naive Bayes model
            predictions_new = clf.predict(df_feat_new)

            # ------------------------------
            # 測試高斯朴素貝葉斯演算法的好壞
            from sklearn.metrics import classification_report, confusion_matrix

            # 將實際類別分為真正例（True Positive）、真負例（True Negative）、偽正例（False Positive）和偽負例（False Negative）
            print('confusion_matrix:')
            print(confusion_matrix(new['ReturnMean_year_Label'], predictions_new))

            # 模型的精確度、召回率、F1分數和支持數等指標，用來評估模型對於每個類別的預測性能。
            print('classification_report')
            print(classification_report(new['ReturnMean_year_Label'], predictions_new))

            # 比較預測結果
            accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
            print(f'新數據的預測準確率: {accuracy_new}')
            # --------下面有改---------------------------
            # print(predictions_new)

            # Assuming clf is your trained Gaussian Naive Bayes classifier

            # Get the probability estimates for the positive class (1)
            probability_positive_class = clf.predict_proba(df_feat_new)[:, 1]

            # Sort the instances based on the probability in descending order
            sorted_indices = np.argsort(probability_positive_class)[::-1]

            # Select the top 5 instances where the predicted label is 1
            selected_indices = [index for index in sorted_indices if predictions_new[index] == 1][:5]

            # Get the corresponding stock names and returns
            selected_stock_names = new.loc[selected_indices, '簡稱']
            selected_stock_returns = new.loc[selected_indices, 'Return']

            # Print the selected stock names and returns
            print("Selected Stock Names:")
            print(selected_stock_names)
            print("\nSelected Stock Returns:")
            print(selected_stock_returns)
            # ------------------------------

            # 將選取的索引轉換成列表
            selected_indices_list = list(selected_indices)

            # 打印預測值為1的前5筆索引
            print("預測值為1的前5筆索引:")
            print(selected_indices_list)
            select = pd.read_excel(csv_file_path) #top200_1998.xlsx-top200_2009.xlsx
            # 獲取股票名稱
            selects_stock = select.loc[selected_indices_list, '簡稱']
            print("獲取股票名稱:")
            print(selects_stock)
            # 獲取股票return
            selects_ret = select.loc[selected_indices_list, 'Return']
            print("獲取股票return:")
            print(selects_ret)
            # 將獲取的 股票名稱 和 股票return 寫進檔案
            with open(f'selected_stock/{col}.txt', 'a') as stockfile: #col=key
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
        print(f'複利為:{sum2}')
        #----------------------------------------------------
        list_ans.append(sum2) #list_ans=[key1選的屬性(本益比)+sum2]
        ans[col] = list_ans #ans[key]
        file.write(f'{col}:{list_ans}\n\n') #1:本益比 sum2
        file.flush()  # 強制將緩衝區內容寫入檔案
        if sum2 > max_ret:
            max_ret = sum2
            with open('max.txt', 'w') as maxfile:
                maxfile.write(f'max:{col}:{list_ans}\n\n')
                maxfile.flush()  # 強制將緩衝區內容寫入檔案

