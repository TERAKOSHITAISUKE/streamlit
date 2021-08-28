import streamlit as st
# import numpy as np
# import pandas as pd
# from PIL import Image

st.title("Streamlit 超入門")

st.write("Date frame")

# df = pd.DataFrame({
#         "1列目" : [1, 2, 3, 4],
#         "2列目" : [10, 20 , 30 , 40]
# })

# st.dataframe(df.style.highlight_max(axis=0))

# st.table(df)


"""
# 章
## 節
### 項

```python
import streamlit as st
import numpy as np
import pandas as pd

```


"""


# df = pd.DataFrame(
#     np.random.rand(100, 2)/[50, 50] + [35.69, 139.70],
#     columns=['lat','lon']
# )


# # st.line_chart(df)

# # st.area_chart(df)

# st.map(df)


st.write("Display image")

option = st.selectbox(
    "あなたが好きな数字を教えてください",
    list(range(1, 10))
)

'あなたの好きな数字は、',option, 'です' 

st.write("Interavtive Widgets")


left_column, right_column = st.beta_columns(2)
button = left_column.button("右カラムに文字を表示")
if button:
    right_column.write("This is right cloumn")

expander = st.beta_expander("問合せ")
expander.write("問い合わせを書く")
expander1 = st.beta_expander("問合せ1")
expander1.write("問い合わせを書く")
expander2 = st.beta_expander("問合せ2")
expander2.write("問い合わせを書く")


import time

st.write("プログレスバーの表示")
"Start!!!"

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Iteration {i +1}')
    bar.progress( i + 1)
    time.sleep(0.1)

"Done!!"

# option2 = st.text_input('あなたの趣味を教えてください')
# 'あなたの趣味は', option2, 'です'

# # option2 = st.sidebar.text_input('あなたの趣味を教えてください')
# # 'あなたの趣味は', option2, 'です'

# condition = st.slider('あなたの今の調子は。',0 , 100, 50)
# 'コンディション', condition



# if st.checkbox("Show Image"):

#   imag = Image.open("sample.png")
#   st.image(imag, caption=False, use_column_width=True)



