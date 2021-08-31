import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_wine 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA 
import plotly.graph_objects as go 

# タイトル の 表示 
st.title(' PCA applied to Wine dataset') 
# ワイン 分類 の データセット を 読み込む 
dataset = load_wine() 
# 説明 変数 を pandas 　 DataFrame 型 で 格納 する 
df = pd.DataFrame(dataset.data) 
# 説明 変数 名 を DataFrame の 列 名 へ 割り当てる 
df.columns = dataset.feature_names 
# 目的 変数（ ワイン の ラベル） を target の 列 名 で 割り当てる 
df["target"] = dataset.target 
# チェック ボックス が ON の 時 のみ データセット を 表示 する 
if st.checkbox(' Show the dataset as table data'): 
     st.dataframe(df)


X = df.drop(columns=["target"])
y = df["target"]


ssccaler = StandardScaler()
x_std = ssccaler.fit_transform(X)

st.sidebar.markdown(
    r"""
    ### Select the number of principal components to include in the result 
    Note: The number is nonnegative integer.
    """
)

num_pca = st.sidebar.number_input(
            "The minumum value is an integer of 3 or more.",
            value = 3,
            step = 1, 
            min_value = 3
)



pca = PCA(n_components=num_pca)
x_pca = pca.fit_transform(x_std)

st.sidebar.markdown(
        r"""
        ### Select the principal components to plot ex.Choose "1" for PCA 1
        """
)

idx_x_pca = st.sidebar.selectbox("x axis is the principal componen of ", np.arange(1, num_pca+1), 0)
idx_y_pca = st. sidebar. selectbox(" y axis is the principal component of ", np. arange( 1, num_pca + 1), 1)
idx_z_pca = st.sidebar.selectbox(" z axis is the principal component of ", np. arange( 1, num_pca + 1), 2) 
# 軸 ラベル 
x_lbl, y_lbl, z_lbl = f"PCA {idx_x_pca}", f" PCA {idx_y_pca}", f" PCA {idx_z_pca}" 
# 各 軸 の データ を 用意 
x_plot, y_plot, z_plot = x_pca[:, idx_x_pca-1], x_pca[:, idx_y_pca-1], x_pca[:, idx_z_pca-1] 

# グラフ オブジェクト の 作成 
trace1 = go. Scatter3d( 
    x = x_plot, y = y_plot, z = z_plot, 
    mode =' markers', 
    marker = dict( size = 5, color = y, ) 
    ) 
# グラフ レイ アウト の オブジェクト を 作成 
fig = go. Figure( data =[trace1]) 
fig. update_layout(scene = dict(
                xaxis_title = x_lbl, 
                yaxis_title = y_lbl, 
                zaxis_title = z_lbl), 
                width = 700, 
                margin = dict(r = 20, b = 10, l = 10, t = 10), 
                ) 
"""### 3 d plot of the PCA result by plotly""" 
# streamlit で Web アプリ 上 に 表示 
st. plotly_chart( fig, use_container_width = True)
