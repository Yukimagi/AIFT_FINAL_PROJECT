#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

csv_file_path='top200_training.xls'
#記得pip install xlrd
df = pd.read_excel(csv_file_path)
#df.head()


# In[15]:


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


# In[16]:


from itertools import combinations
from sklearn.model_selection import train_test_split
# 所有的欄位
all_columns = ['股價淨值比', '股價營收比' ,'M淨值報酬率─稅後', 'M營業利益成長率', 'M稅後淨利成長率']

# 開始的欄位
start_column = '股價淨值比'

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
with open('column_combinations_KNN_train.txt', 'w') as file:
    # 遍歷字典，將結果寫入檔案
    for key, value in column_combinations_dict.items():
        file.write(f"Key: {key}, Selected Columns: {value}\n")
        # 在這裡進行你的分析或模型擬合等操作
        # ...
        file.write("\n" + "="*40 + "\n")

# 提示保存成功
print("Results saved to 'column_combinations_KNN_train.txt'")


# In[ ]:


max_ret=0
ans={}
list_ans=[]
#選擇特定欄位分析
from sklearn.preprocessing import StandardScaler
from itertools import islice
cols=0
start_index = 1  # 第3001筆資料的索引
end_index = 4093  # 最後一筆資料的索引

selected_range = islice(year_combinations_dict.items(), start_index, end_index + 1)
#selected_range = islice(year_combinations_dict.items(), 4094) #4094是所有可用來訓練的年份組合

#16384
with open('KNN_train_test_result.txt', 'a') as file:
    for key, selected_years in selected_range:
        
        # 選擇訓練集年分
        train_years = [int(year) for year in selected_years]
        #print(train_years)

        #print("---------------------")
        # 選擇測試集年分
        test_years = [int(year) for year in all_years if year not in selected_years]
        print(f'總共有test year:{len(test_years)}')
        
        # 根據年份選擇訓練集
        train_data = df[df['年月'].dt.year.isin(train_years)]
        print(f"Number of rows in train_data: {len(train_data)}")
        #print(train_data)
        
        # 排除指定列，並標準化
        #columns_to_exclude = ['簡稱', '證券代碼', '年月', 'ReturnMean_year_Label', 'Return']
        #features_to_scale_train = train_data.drop(columns=columns_to_exclude)
        #features_to_scale_test = test_data.drop(columns=columns_to_exclude)
    
        for col in range (1,32):#取top10組合的前幾大frquency的屬性組合測試
            cols=cols+1
            list_ans=[]
            # 指定要選擇的欄位
            selected_columns = column_combinations_dict[col]
            list_ans.append(selected_columns)
            
            # 選擇指定的欄位
            features_to_scale_train = train_data[selected_columns]

            scaler = StandardScaler()
            scaled_features_train = scaler.fit_transform(features_to_scale_train)
            
            # 轉換為dataframe
            df_feat_train = pd.DataFrame(scaled_features_train, columns=features_to_scale_train.columns)
            
        #---------------------------------------------------------------------------------------------
            # 分割訓練集和測試集
            X_train = df_feat_train
            y_train = train_data['ReturnMean_year_Label']
            
            X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.3,random_state=101)
  
        #--------------------------------------------------------------------------------------------------
            error_rate = []
            from sklearn.neighbors import KNeighborsClassifier
            for i in range(1, 100):
                knn = KNeighborsClassifier(n_neighbors=i, p=2, weights='distance', algorithm='brute')
                knn.fit(X_train, y_train)
                pred_i = knn.predict(X_test)
                error_rate.append(np.mean(pred_i != y_test))

            # 將k=1~60的錯誤率製圖畫出。k=23之後，錯誤率就在5-6%之間震盪。
            #plt.figure(figsize=(10, 6))
            #plt.plot(range(1, 100), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
            #plt.title('Error Rate vs. K Value')
            #plt.xlabel('K')
            #plt.ylabel('Error Rate')
            #plt.show()
        #--------------------------------------------------------------------------------------------------
            min_error = min(error_rate)
            optimal_k = error_rate.index(min_error) + 1  # Adding 1 because Python indexing starts from 0

            print(f"Optimal k value for minimum error rate: {optimal_k}")
            #print(f"Minimum Error Rate: {min_error}")
            list_ans.append(optimal_k)
        #--------------------------------------------------------------------------------------------------
            #使用KNN演算法
            clf=KNeighborsClassifier(n_neighbors=optimal_k,p=2,weights='distance',algorithm='brute')
            clf.fit(X_train,y_train)
        #--------------------------------------------------------------------------------------------------
            predict_label1={}
            predict_k_pre5={}
        #--------------------------------------------------------------------------------------------------

            for year in test_years:
                
                # 選擇特定年份的 test_data
                test_data = df[df['年月'].dt.year == year]
                print(year)
                #print(test_data)
                print(f"Number of rows in test_data: {len(test_data)}")
                
                # 選擇指定的欄位
                features_to_scale_test = test_data[selected_columns]
                # 使用之前訓練好的標準化物件進行標準化
                scaled_features_test = scaler.transform(features_to_scale_test)
                # 將標準化後的特徵資料轉換為 DataFrame
                df_feat_test = pd.DataFrame(scaled_features_test, columns=features_to_scale_test.columns)

                
                # 使用已經訓練好的模型進行預測
                predictions_new = clf.predict(df_feat_test)

                #------------------------------
                #測試KNN演算法的好壞
                from sklearn.metrics import classification_report,confusion_matrix
                #將實際類別分為真正例（True Positive）、真負例（True Negative）、偽正例（False Positive）和偽負例（False Negative）
                print('confusion_matrix:')
                print(confusion_matrix(test_data['ReturnMean_year_Label'],predictions_new))

                #模型的精確度、召回率、F1分數和支持數等指標，用來評估模型對於每個類別的預測性能。
                print('classification_report')
                print(classification_report(test_data['ReturnMean_year_Label'],predictions_new))

                # 比較預測結果
                accuracy_new = clf.score(df_feat_test, test_data['ReturnMean_year_Label'])
                print(f'新數據的預測準確率: {accuracy_new}')
                #---------------------------------------
                #print(predictions_new)

                #predicted_positive_indices = (predictions_new == 1)

                # 獲取股票名稱
                #predicted_positive_stock_names = new.loc[predicted_positive_indices, '簡稱']

                # 設定檔案名稱
                #output_file_name = 'selected_stocks_1998.csv'

                # 匯出成 CSV 檔案
                #predicted_positive_stock_names.to_csv(output_file_name, index=True)

                # 預測要投資的股票名稱
                #print("選擇股票:")
                #print(predicted_positive_stock_names)


                # 選擇預測為1的股票
                selected_stocks = test_data[predictions_new == 1]

                # 計算return
                stock_returns = selected_stocks['Return']

                portfolio_returns = (stock_returns.mean()/100)+1

                print(portfolio_returns)
                predict_label1[year]=portfolio_returns

                # 使用 kneighbors 方法取得最近鄰居的索引和距離
                distances, indices = clf.kneighbors(df_feat_test, n_neighbors=5)

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
                    if(index>=200 and index<400):
                        index=index-200
                    elif(index>=400 and index<600):
                        index=index-400
                    elif(index>=600 and index<800):
                        index=index-600
                    elif(index>=800 and index<1000):
                        index=index-800
                    elif(index>=1000 and index<1200):
                        index=index-1000
                    elif(index>=1200 and index<1400):
                        index=index-1200
                    elif(index>=1400 and index<1600):
                        index=index-1400
                    elif(index>=1600 and index<1800):
                        index=index-1600
                    elif(index>=1800 and index<2000):
                        index=index-1800
                    elif(index>=2000 and index<2200):
                        index=index-2000
                    elif(index>=2200 and index<2400):
                        index=index-2200
                    elif(index>=2400 and index<2600):
                        index=index-2400
                    elif(index>=2600 and index<2800):
                        index=index-2600
                    elif(index>=2800 and index<3000):
                        index=index-2800
                    elif(index>=3000 and index<3200):
                        index=index-3000
                    prediction = predictions_new[index]
                    if prediction == 1 and index not in selected_indices:
                        selected_indices.add(index)
                    i += 1

                # 將選取的索引轉換成列表
                selected_indices_list = list(selected_indices)

                # 打印最近鄰居中預測值為1的前5筆索引
                print("最近鄰居中預測值為1的前5筆索引:")
                print(selected_indices_list)
                select = test_data
                # 獲取股票名稱
                selects_stock = select.iloc[selected_indices_list]['簡稱']
                print(selects_stock)
                # 獲取股票return
                selects_ret = select.iloc[selected_indices_list]['Return']
                print(selects_ret)
                with open(f'selected_stock_KNN_train/{cols}.txt', 'a') as stockfile:
                    stockfile.write(selects_stock.to_string())
                    stockfile.write('\n')
                    stockfile.write(selects_ret.to_string())
                # 計算return
                stock_returns = (selects_ret.mean() / 100)+1
                portfolio_returns = stock_returns
                print(portfolio_returns)
                predict_k_pre5[year] = portfolio_returns
                #print(stock_returns)
            sum1=1
            sum2=1
            print(predict_label1)
            for k in predict_label1.keys():
                if not (predict_label1[k]==0):
                    sum1=predict_label1[k]*sum1
            print(f'年均化複利為:{sum1/int(len(test_years))}')
            print('\n')
            print(predict_k_pre5)
            for k in predict_k_pre5.keys():
                if not (predict_k_pre5[k]==0):
                    sum2=predict_k_pre5[k]*sum2
            print(f'年均化複利為:{sum2/int(len(test_years))}')
            #list_ans.append(sum2/len(test_data)/200)
            print(f'ori sum2:{sum2}')
            print(f'test years have:{int(len(test_years))}')
            sum2=sum2/int(len(test_years))
            print(f'after sum2:{sum2}')
            list_ans.append(sum2)
            ans[cols]=list_ans
            file.write(f'{cols}:{list_ans}\n\n')
            file.flush()  # 強制將緩衝區內容寫入檔案
            if(sum2>max_ret):
                max_ret=sum2
                with open('max_KNN_train.txt', 'w') as maxfile:
                    maxfile.write(f'max:train_year:{train_years},test_year:{test_years}:{cols}:{list_ans}\n\n')
                    maxfile.flush()  # 強制將緩衝區內容寫入檔案


# In[ ]:




