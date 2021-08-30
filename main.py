import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("ボストン市の住宅価格の重回帰分析")

dataset = load_boston()
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
Features = df.columns
df["PRICES"] = dataset['target']

if st.checkbox("テーブルデータ形式でデータセットを表示"):
    st.dataframe(df)

if st.checkbox(' 説明変数のカラム名とその説明を表示'):
    st.markdown(
         r"""
          ### 説明変数のカラム名とその説明 
           #### CRIM: 町ごとの単位人口あたりの犯罪発生率 
           #### ZN: 25,000平方フィート面積を有する住宅の割合 
           #### INDUS:町ごとの非小売の土地面積の割合 
           #### CHAS:チャールズ川沿いの立地か否か：0は近郊、1は非近郊 
           #### NOX: 窒素化合物濃度 
           #### RM: 住宅あたりの平均部屋数 
           #### AGE: 1940年以前に建てられた建物の割合 
           #### DIS: 5つの雇用センターからの加重距離 
           #### RAD: 高速道路へのアクセスし易さの指標 
           #### TAX: 10万ドルあたりの税率 
           #### PTRATIO: 町あたりの生徒と教師の割合 
           #### B: 黒人割合に関連する指標 
           #### LSTAT: 下層階級の人口割合
          ###
          """)


if st.checkbox("目的変数と説明変数の相関を可視化"):
    checked_variable = st.selectbox("説明変数を1つ選択してください:",
    df.drop(columns="PRICES").columns)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x = df[checked_variable], y = df["PRICES"])
    plt.xlabel(checked_variable)
    plt.ylabel("PRICES")

    st.pyplot(fig)


"""
## 前処理
"""

Features_NotUsed = st.multiselect(
    "学習時に使用しない変数を選択してください",
    Features
)

df = df.drop(columns = Features_NotUsed)

left_column, right_column = st.columns(2)
bool_log = left_column.radio(
    "対数変換を行いますか？",
    ("No","Yes")
)

df_log, Log_Features = df.copy(), []
if bool_log == "Yes":
    Log_Features == right_column.multiselect(
        "対数変換を適用する目的変数もしくは説明変数を選択してください",
        df.columns
    )
    df_log[Log_Features] = np.log(df_log[Log_Features])

left_column, right_column = st.columns(2)
bool_std = left_column.radio(
    "標準化を行いますか？",
    ("Yes", "No")
)

df_std = df_log.copy()
if bool_std == "Yes":
 Std_Features_NotUsed = right_column.multiselect(
     "標準化を適用しない変数を選択してください(例えば質的変数)", 
     df_log.drop(columns=["PRICES"]).columns
 )

Std_Features_chosen = []
for name in df_log.drop(columns=["PRICES"]).columns:
    if name in Std_Features_NotUsed:
        continue
    else:
        Std_Features_chosen.append(name)
ssccaler = preprocessing.StandardScaler()
ssccaler.fit(df_std[Std_Features_chosen])
df_std[Std_Features_chosen] = ssccaler.transform(df_std[Std_Features_chosen])




left_column, right_column = st.columns(2)
test_size = left_column.number_input(
    "検証用データのサイズ(比率:0.0-1.0):",
    min_value = 0.0, max_value =1.0,value=0.2,step=0.1
)
random_seed = right_column.number_input(
    "ランダムシードの設定(0以上の整数):",
    value=0, step=1, min_value=0)

X_train, X_val, Y_train, Y_val = train_test_split(
    df_std.drop(columns=["PRICES"]),
    df_std["PRICES"],
    test_size=test_size,
    random_state=random_seed
)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)

if "PRICES" in Log_Features:
    Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
    Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)

"""
## 結果の表示
"""


"""
### モデルの精度
"""

R2 = r2_score(Y_val, Y_pred_val)
st.write(f"R2値 : {R2:.2f}")

"""
### グラフの描写
"""

left_column, right_column = st.columns(2)
show_train = left_column.radio(
    "訓練データの結果を表示:",
    ("Yes", "No")
)
show_val = right_column.radio(
    "検証データの結果を表示:",
    ("Yes", "No")
)

y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val]))

left_column, right_column = st.columns(2)
x_min = left_column.number_input('x軸の最小値:', value = 0, step = 1) 
x_max = right_column.number_input('x軸の最大値:', value = y_max, step = 1) 
left_column, right_column = st.columns(2) 
y_min = left_column.number_input('y軸の最小値:', value = 0, step = 1) 
y_max = right_column.number_input('y軸の最大値:', value = y_max, step = 1)


fig = plt.figure(figsize=(3, 3))
if show_train == "Yes":
    plt.scatter(Y_train, Y_pred_train, lw=0.1, color="r", label='training data')
if show_val == "Yes":
    plt.scatter(Y_val, Y_pred_val, lw=0.1, color="b", label='validation data')

plt.xlabel('PRICES', fontsize=8)
plt.ylabel('Prediction of PRICES', fontsize = 8)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)

st.pyplot(fig)

