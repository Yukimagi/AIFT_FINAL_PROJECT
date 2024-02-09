#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path='top200_1997.xlsx'

df = pd.read_excel(csv_file_path)
df.head()


# In[23]:


from sklearn.preprocessing import StandardScaler

columns_to_exclude = ['簡稱', '證券代碼', '年月','ReturnMean_year_Label']

# 排除指定的欄位
features_to_scale = df.drop(columns=columns_to_exclude)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_to_scale)

# 將標準化後的特徵資料轉換為 DataFrame
df_feat = pd.DataFrame(scaled_features, columns=features_to_scale.columns)
df_feat.head()


# In[24]:


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


max_ret=0
ans={}
list_ans=[]
#選擇特定欄位分析
from sklearn.preprocessing import StandardScaler
#16384
with open('test1.txt', 'a') as file:
    
    for col in range (1,16384):
        list_ans=[]
        # 指定要選擇的欄位
        selected_columns = column_combinations_dict[col]
        list_ans.append(selected_columns)
        # 選擇指定的欄位
        features_to_scale = df[selected_columns]

        #scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_to_scale)

        # 將標準化後的特徵資料轉換為 DataFrame
        df_feat = pd.DataFrame(scaled_features, columns=selected_columns)
        print(df_feat.head())
    #--------------------------------------------------------------------------------------------------

        #將資料分成訓練組及測試組
        from sklearn.model_selection import train_test_split

        X = df_feat.dropna()
        print(X)
        y = df.loc[X.index, 'ReturnMean_year_Label']
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
        print(X_train, y_train)
    #--------------------------------------------------------------------------------------------------
        error_rate = []
        from sklearn.neighbors import KNeighborsClassifier
        for i in range(1, 100):
            knn = KNeighborsClassifier(n_neighbors=i, p=2, weights='distance', algorithm='brute')
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))

        # 將k=1~60的錯誤率製圖畫出。k=23之後，錯誤率就在5-6%之間震盪。
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 100), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()
    #--------------------------------------------------------------------------------------------------
        min_error = min(error_rate)
        optimal_k = error_rate.index(min_error) + 1  # Adding 1 because Python indexing starts from 0

        print(f"Optimal k value for minimum error rate: {optimal_k}")
        print(f"Minimum Error Rate: {min_error}")
        list_ans.append(optimal_k)
    #--------------------------------------------------------------------------------------------------
        #使用KNN演算法
        clf=KNeighborsClassifier(n_neighbors=optimal_k,p=2,weights='distance',algorithm='brute')
        clf.fit(X_train,y_train)
    #--------------------------------------------------------------------------------------------------
        predict_label1={}
        predict_k_pre5={}
    #--------------------------------------------------------------------------------------------------
        for year in range(1998,2010):
            csv_file_path = f'top200_{year}.xlsx'
            new = pd.read_excel(csv_file_path)

            # 排除指定的欄位
            df_temp = new.drop(columns=columns_to_exclude)
            # 選擇指定的欄位
            features_to_scale_new=df_temp[selected_columns]

            # 使用之前訓練好的標準化物件進行標準化
            scaled_features_new = scaler.fit_transform(features_to_scale_new)

            # 將標準化後的特徵資料轉換為 DataFrame
            df_feat_new = pd.DataFrame(scaled_features_new, columns=features_to_scale_new.columns)        

            # 使用已經訓練好的模型進行預測
            predictions_new = clf.predict(df_feat_new)

            #------------------------------
            #測試KNN演算法的好壞
            from sklearn.metrics import classification_report,confusion_matrix
            #將實際類別分為真正例（True Positive）、真負例（True Negative）、偽正例（False Positive）和偽負例（False Negative）
            print('confusion_matrix:')
            print(confusion_matrix(new['ReturnMean_year_Label'],predictions_new))

            #模型的精確度、召回率、F1分數和支持數等指標，用來評估模型對於每個類別的預測性能。
            print('classification_report')
            print(classification_report(new['ReturnMean_year_Label'],predictions_new))

            # 比較預測結果
            accuracy_new = clf.score(df_feat_new, new['ReturnMean_year_Label'])
            print(f'新數據的預測準確率: {accuracy_new}')
            #---------------------------------------
            #print(predictions_new)

            predicted_positive_indices = (predictions_new == 1)

            # 獲取股票名稱
            predicted_positive_stock_names = new.loc[predicted_positive_indices, '簡稱']

            # 設定檔案名稱
            output_file_name = 'selected_stocks_1998.csv'

            # 匯出成 CSV 檔案
            predicted_positive_stock_names.to_csv(output_file_name, index=True)

            # 預測要投資的股票名稱
            print("選擇股票:")
            print(predicted_positive_stock_names)


            # 選擇預測為1的股票
            selected_stocks = new[predictions_new == 1]

            # 計算return
            stock_returns = selected_stocks['Return']

            portfolio_returns = (stock_returns.mean()/100)+1

            print(portfolio_returns)
            predict_label1[year]=portfolio_returns

            # 使用 kneighbors 方法取得最近鄰居的索引和距離
            distances, indices = clf.kneighbors(df_feat_new, n_neighbors=5)

            # 合併所有測試樣本的最近鄰居索引
            all_indices = np.concatenate(indices)
            all_distances = np.concatenate(distances)

            # 將索引和距離組合成一個 2D 陣列，方便排序
            combined_data = np.column_stack((all_indices, all_distances))

            # 按照距離重新排序
            sorted_combined_data = combined_data[np.argsort(combined_data[:, 1])]

            # 選取最近鄰居中預測值為1的前5筆，且不重複
            selected_indices = set()
            i = 0
            while len(selected_indices) < 5 and i < len(sorted_combined_data):
                index = int(sorted_combined_data[i, 0])
                prediction = predictions_new[index]
                if prediction == 1 and index not in selected_indices:
                    selected_indices.add(index)
                i += 1

            # 將選取的索引轉換成列表
            selected_indices_list = list(selected_indices)

            # 打印最近鄰居中預測值為1的前5筆索引
            print("最近鄰居中預測值為1的前5筆索引:")
            print(selected_indices_list)
            select = pd.read_excel(csv_file_path)
            # 獲取股票名稱
            selects_stock = select.loc[selected_indices_list,'簡稱']
            print(selects_stock)
            # 獲取股票return
            selects_ret = select.loc[selected_indices_list,'Return']
            print(selects_ret)
            with open(f'selected_stock/{col}.txt', 'a') as stockfile:
                stockfile.write(selects_stock.to_string())
                stockfile.write('\n')
                stockfile.write(selects_ret.to_string())
            # 計算return
            stock_returns = (selects_ret.mean() / 100)+1
            portfolio_returns = stock_returns
            print(portfolio_returns)
            predict_k_pre5[year]=portfolio_returns
            #print(stock_returns)
        sum1=1
        sum2=1
        print(predict_label1)
        for k in predict_label1.keys():
            if not (predict_label1[k]==0):
                sum1=predict_label1[k]*sum1
        print(f'複利為:{sum1}')
        print('\n')
        print(predict_k_pre5)
        for k in predict_k_pre5.keys():
            if not (predict_k_pre5[k]==0):
                sum2=predict_k_pre5[k]*sum2
        print(f'複利為:{sum2}')
        list_ans.append(sum2)
        ans[col]=list_ans
        file.write(f'{col}:{list_ans}\n\n')
        file.flush()  # 強制將緩衝區內容寫入檔案
        if(sum2>max_ret):
            max_ret=sum2
            with open('max.txt', 'w') as maxfile:
                maxfile.write(f'max:{col}:{list_ans}\n\n')
                maxfile.flush()  # 強制將緩衝區內容寫入檔案


# In[17]:


ans

