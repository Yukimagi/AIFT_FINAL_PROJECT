# https://www.datacamp.com/tutorial/decision-tree-classification-python
# https://tree.rocks/decision-tree-graphviz-contour-with-pandas-gen-train-test-dataset-for-beginner-9137b7c8416a
# https://medium.com/jackys-blog/machine-learning%E4%B8%8B%E7%9A%84decision-tree%E5%AF%A6%E4%BD%9C%E5%92%8Crandom-forest-%E8%A7%80%E5%BF%B5-%E4%BD%BF%E7%94%A8python-3a94cef2ce6f
# 2023/12/20 Sherry pan

# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
## Common imports
import sys
import sklearn # scikit-learn
import os
import scipy
## plot
import matplotlib as mpl
import matplotlib.pyplot as plt
## 分割資料
from sklearn import datasets
from sklearn.model_selection import train_test_split

## modelbuilding 模型套件
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree #圖
from sklearn import tree


col_names = ['證券代碼', '簡稱', '年月', '市值(百萬元)', 
             '收盤價(元)_年', '本益比', '股價淨值比', '股價營收比', 
             'M淨值報酬率─稅後', '資產報酬率ROA', '營業利益率OPM', 
             '利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率', 
             'M存貨週轉率 (次)', 'M應收帳款週轉次', 'M營業利益成長率', 
             'M稅後淨利成長率', 'Return', 'ReturnMean_year_Label']
# load dataset
file_path = r"C:\Users\sherr\python\人工智慧與金融科技\top200_year\top200_1997.xlsx"
df = pd.read_excel(file_path,names=col_names)

feature_cols = ['股價營收比', '利潤邊際NPM', 'M速動比率']
# feature_cols = ['市值(百萬元)', '收盤價(元)_年', '本益比', '股價營收比', 'M淨值報酬率─稅後', '資產報酬率ROA', '利潤邊際NPM', 'M速動比率']
# feature_cols=['本益比', '股價營收比', 'M淨值報酬率─稅後', '利潤邊際NPM', '負債/淨值比', 'M流動比率', 'M速動比率','M存貨週轉率 (次)']
target = ['ReturnMean_year_Label']

x_train=df[feature_cols]
y_train=df[target]

#criterion使用entropy來計算訊息增益 
#max_depth是指樹狀圖的最大深度

model=DecisionTreeClassifier(max_depth=2,criterion='entropy', ccp_alpha=0.0)
model.fit(x_train,y_train)

tree.plot_tree(model)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(feature_cols)
sum=0.0
for i in range(1998, 2010):
    # Load the new dataset
    file_path = r"C:\\Users\sherr\\python\\人工智慧與金融科技\\top200_year\top200_"+str(i)+".xlsx"
    df = pd.read_excel(file_path, names=col_names)

    # Extract features and target variable
    x_test = df[feature_cols]
    y_test = df.ReturnMean_year_Label

    # Predict using the trained model
    y_pred = model.predict(x_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    sum=sum+accuracy
    print("Accuracy on top200_"+str(i)+" dataset:", accuracy)
print("Average: ", sum/12)



import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Create empty lists to store results
all_簡稱_array = []
all_Return_array = []
predict_label1={}
predict_k_pre5={}


sum=0.0
# Iterate through datasets
for i in range(1998, 2010):
    # Load the new dataset
    file_path = r"C:\\Users\sherr\\python\\人工智慧與金融科技\\top200_year\\top200_"+str(i)+".xlsx"
    df = pd.read_excel(file_path, names=col_names)

    # Extract features and target variable
    X = df[feature_cols]
    y = df.ReturnMean_year_Label
    # Predict using the trained model
    y_pred = model.predict(X)

    # Identify rows where the prediction is 1
    indices = [index for index, prediction in enumerate(y_pred) if prediction == 1]

    # Create arrays to store results for each dataset
    簡稱_array = [df.at[index, '簡稱'] for index in indices]
    Return_array = [df.at[index, 'Return'] for index in indices]

    # Append values to the overall lists
    all_簡稱_array.extend(簡稱_array)
    all_Return_array.extend(Return_array)
    
     # 選擇預測為1的股票
    selected_stocks = df[y_pred == 1]

    # 計算return
    stock_returns = selected_stocks['Return']

    portfolio_returns = (stock_returns.mean()/100)+1
    
    predict_label1[i]=portfolio_returns

    # Get predicted probabilities on the test set\n",
    probabilities = model.predict_proba(x_test)
    # Find the indices of the top five instances with the highest predicted probabilities for the positive class\n",
    top_five_indices = np.argsort(probabilities[:, 1])[::-1][:5]
    # Extract the corresponding features and actual labels for the top five predictions\n",
    top_five_instances_features = x_test.iloc[top_five_indices]
    actual_labels = y_test.iloc[top_five_indices]
    selects_stock = df.loc[top_five_indices,'簡稱']
    print(selects_stock)
    # 獲取股票return\n",
    selects_ret = df.loc[top_five_indices,'Return']
    print(selects_ret)
    # 計算return\n",
    stock_returns = (selects_ret.mean() / 100)+1
    portfolio_returns = stock_returns
    print(portfolio_returns)
    predict_k_pre5[i]=portfolio_returns


# Create a DataFrame from the combined arrays
result_df = pd.DataFrame({'簡稱': all_簡稱_array, 'Return': all_Return_array})

# Group by '簡稱' and calculate the average 'Return'
result_df = result_df.groupby('簡稱').mean().reset_index()

# Sort the DataFrame by average 'Return' in descending order
result_df = result_df.sort_values(by='Return', ascending=False)

# Display the combined, averaged, and sorted DataFrame
print(result_df.head(20))


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

