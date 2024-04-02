import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px


path = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'

@st.cache_data
def load_data():
    tips = pd.read_csv(path)
    return tips

tips=load_data()

st.title('My First Streamlit App')
st.header('Исследование по чаевым')

st.sidebar.title("About")
st.sidebar.info(
    """
    Данное приложение визуализирует исследование по чаевым
    """
)


st.subheader("Гистограмма общей суммы чеков:")
plot = sns.histplot(data=tips, x="total_bill", kde=True)
st.pyplot(plot.get_figure())
plt.close()


st.subheader("График, связывающий сумму счета, чаевые и день недели")
total_bill_range = st.slider('Диапазон суммы счета', float(tips['total_bill'].min()), float(tips['total_bill'].max()), (float(tips['total_bill'].min()), float(tips['total_bill'].max())))
filtered_tips = tips[(tips['total_bill'] >= total_bill_range[0]) & (tips['total_bill'] <= total_bill_range[1])]
plot1 = sns.relplot(data=filtered_tips, x="total_bill", y="tip", hue="day")
plot1.set_axis_labels(x_var='Сумма счета', y_var='Сумма чаевых')
st.pyplot(plot1)
plt.close()


st.subheader("График, связывающий сумму счета, чаевые и размер заказа")
size_range = st.slider('Диапазон размера заказа', int(tips['size'].min()), int(tips['size'].max()), (int(tips['size'].min()), int(tips['size'].max())))
filtered_size = tips[(tips['size'] >= size_range[0]) & (tips['size'] <= size_range[1])]
plot2 = sns.relplot(data=filtered_size, x="total_bill", y="tip", hue="size")
plot2.set_axis_labels(x_var='Сумма счета', y_var='Сумма чаевых')
st.pyplot(plot2)
plt.close()



st.subheader("Связь суммы счета и дня недели")
selected_days = st.multiselect('Выберите дни недели', tips['day'].unique(), default=tips['day'].unique())
filtered_tips = tips[tips['day'].isin(selected_days)]
fig = px.scatter(filtered_tips, x='total_bill', y='tip', color='day')
st.plotly_chart(fig)
# plot3 = sns.stripplot(data=filtered_tips, x="total_bill", y="day", hue="day", dodge=True)
# plot3.set(xlabel='Сумма счета', ylabel='День недели')
# st.pyplot(plt.gcf())
# plt.close()


st.subheader("Связь суммы чаевых, дня недели и пола клиента")
selected_sex = st.multiselect('Выберите пол клиента', tips['sex'].unique(), default=tips['sex'].unique())
filtered_sex= tips[tips['sex'].isin(selected_sex)]
plot4 = sns.stripplot(data=filtered_sex, x="tip", y="day", hue="sex", dodge=True)
plot4.set(xlabel='Сумма чаевых', ylabel='День недели')
st.pyplot(plt.gcf())
plt.close()

if st.button('Показать датафрейм tips'):
    st.write(tips)