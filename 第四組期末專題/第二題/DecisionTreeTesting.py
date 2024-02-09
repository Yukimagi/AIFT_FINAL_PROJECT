import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#請直接在這裡並一您要操作的檔案名稱與年份(請依照以下格是撰寫)

csv_file_path='top200_training.xls'
#測試資料檔案名稱:
#test_csv_file_path='top200_testing.xls'
test_csv_file_path='top200_testing.xls'

#測試年分:(請一定要輸入int格式的!!!)
#test_years = [2010,2011,2012,2013,2014,2015]
# 使用 input() 函數接收測試年分的輸入
test_years_input = input("請輸入測試年分，以逗號分隔（例如：2010,2011,2012）: ")

# 將輸入的年分轉換為整數格式
test_years = [int(year) for year in test_years_input.split(',')]

#記得pip install xlrd
df = pd.read_excel(csv_file_path)
df_test = pd.read_excel(test_csv_file_path)
df.head()

from itertools import product
from itertools import combinations


# 將年月轉換為日期型數據
df['年月'] = pd.to_datetime(df['年月'], format='%Y%m')

df_test['年月'] = pd.to_datetime(df_test['年月'], format='%Y%m')


max_ret=0
ans={}
list_ans=[]
#選擇特定欄位分析
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import islice

from sklearn.tree import DecisionTreeClassifier

#selected_range = islice(year_combinations_dict.items(), 1) #4094是所有可用來訓練的年份組合
#selected_range = year_combinations_dict.items()
isopen=0
with open('Decision_train_test_result.txt', 'a') as file:
    file.truncate(0)
    # 選擇訓練集年分
    train_years = [1997,1999,2002,2004,2007]#要寫int數字非字串(很重要)
    #print(train_years)

    #print("---------------------")
    print(f'總共有test year:{len(test_years)}')
        
    # 根據年份選擇訓練集
    train_data = df[df['年月'].dt.year.isin(train_years)]
    print(f"Number of rows in train_data: {len(train_data)}")
    print(train_data)
        
    # 排除指定列，並標準化
    #columns_to_exclude = ['簡稱', '證券代碼', '年月', 'ReturnMean_year_Label', 'Return']
    #features_to_scale_train = train_data.drop(columns=columns_to_exclude)
    #features_to_scale_test = test_data.drop(columns=columns_to_exclude)
    
    list_ans=[]
    # 指定要選擇的欄位
    selected_columns = ['股價營收比', '利潤邊際NPM', 'M速動比率']
    list_ans.append(selected_columns)
    
    # 檢查 train_data 是否包含任何資料
    if len(train_data) == 0:
        print("警告：train_data 中沒有任何資料。")
    else:
        # 檢查每個欄位是否都存在於 train_data
        missing_columns = [col for col in selected_columns if col not in train_data.columns]

        if not missing_columns:
            # 如果所有列都存在
            
            # 選擇指定的欄位
            features_to_scale_train = train_data[selected_columns]

            scaler = StandardScaler()
            scaled_features_train = scaler.fit_transform(features_to_scale_train)

            # 將標準化後的特徵資料轉換為 DataFrame
            df_feat_train = pd.DataFrame(scaled_features_train, columns=features_to_scale_train.columns)
            
            print(df_feat_train.head())
        else:
            print(f"警告：train_data 中缺少以下列： {missing_columns}")
        

    #---------------------------------------------------------------------------------------------
    # 分割訓練集和測試集
    X_train = df_feat_train
    y_train = train_data['ReturnMean_year_Label']

    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.3,random_state=101)

    model=DecisionTreeClassifier(max_depth=2,criterion='entropy', ccp_alpha=0.025)
    model.fit(X_train,y_train)
    sum=0.0
    predict_label1={}
    predict_k_pre5={}
    
    for year in test_years:

        # 選擇特定年份的 test_data
        test_data = df_test[df_test['年月'].dt.year == year]
        print(year)
        
        print(test_data)
        print(f"Number of rows in test_data: {len(test_data)}")

        # 選擇指定的欄位
        features_to_scale_test = test_data[selected_columns]
        # 使用之前訓練好的標準化物件進行標準化
        scaled_features_test = scaler.transform(features_to_scale_test)
        # 將標準化後的特徵資料轉換為 DataFrame
        df_feat_test = pd.DataFrame(scaled_features_test, columns=features_to_scale_test.columns)


        # 使用已經訓練好的模型進行預測
        predictions_new = model.predict(df_feat_test)

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
        accuracy_new = model.score(df_feat_test, test_data['ReturnMean_year_Label'])
        print(f'新數據的預測準確率: {accuracy_new}')
        #---------------------------------------


        # Get the probability estimates for the positive class (1)
        probability_positive_class = model.predict_proba(df_feat_test)[:, 1]

        # Sort the instances based on the probability in descending order
        sorted_indices = np.argsort(probability_positive_class)[::-1]

        # Set a custom threshold (e.g., 0.5)
        custom_threshold = 0.5

        # Update predictions based on the custom threshold
        predictions_new = (probability_positive_class > custom_threshold).astype(int)

        # Select the top 5 instances where the predicted label is 1
        selected_indices = np.argsort(probability_positive_class)[::-1][:5]

        # 將選取的索引轉換成列表
        selected_indices_list = list(selected_indices)

        # 打印預測值為1的前5筆索引
        print("單一年份預測值為1的前5筆索引:")
        print(selected_indices_list)
        select = test_data
        #select = test_data_cleaned
                
        # 獲取股票名稱
        selects_stock = select.iloc[selected_indices_list]['簡稱']#這裡用iloc按位置選擇行(這也是為甚麼前面會需要減掉1000(代表從1000排)(才會對應到真正的相對位置區域)
        #print("獲取股票名稱:")
        print(selects_stock)
                
        # 獲取股票return
        selects_ret = select.iloc[selected_indices_list]['Return']
        #print("獲取股票return:")
        print(selects_ret)

        with open(f'selected_stock_DecisionTree_train/attr.txt', 'a') as stockfile:
            if (isopen==0):
                stockfile.truncate(0)
                isopen=1
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
    ans=list_ans
    file.write(f'{list_ans}\n\n')
    file.flush()  # 強制將緩衝區內容寫入檔案
    if(sum2>max_ret):
        max_ret=sum2
        with open('max.txt', 'w') as maxfile:
            maxfile.write(f'max:train_year:{train_years},test_year:{test_years}:{list_ans}\n\n')
            maxfile.flush()  # 強制將緩衝區內容寫入檔案