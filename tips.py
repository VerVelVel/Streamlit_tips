# import numpy as np
import pandas as pd

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns
import streamlit as st
import plotly.express as px
# import plotly.graph_objects as go

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
fig1 = px.histogram(tips, x="total_bill")
fig1.update_xaxes(title_text="Сумма счета ($)")
fig1.update_yaxes(title_text="Частота")  
st.plotly_chart(fig1)
# plot = sns.histplot(data=tips, x="total_bill", kde=True)
# st.pyplot(plot.get_figure())
# plt.close()


st.subheader("График, связывающий сумму счета, чаевые и день недели")
total_bill_range = st.slider('Диапазон суммы счета', float(tips['total_bill'].min()), float(tips['total_bill'].max()), (float(tips['total_bill'].min()), float(tips['total_bill'].max())))
filtered_tips = tips[(tips['total_bill'] >= total_bill_range[0]) & (tips['total_bill'] <= total_bill_range[1])]
plot1 = px.scatter(filtered_tips, x="total_bill", y="tip", color="day")
plot1.update_xaxes(title_text="Сумма счета ($)")
plot1.update_yaxes(title_text="Сумма чаевых ($)")  
st.plotly_chart(plot1)
# plot1 = sns.relplot(data=filtered_tips, x="total_bill", y="tip", hue="day")
# plot1.set_axis_labels(x_var='Сумма счета', y_var='Сумма чаевых')
# st.pyplot(plot1)
# plt.close()


st.subheader("График, связывающий сумму счета, чаевые и размер заказа")
size_range = st.slider('Диапазон размера заказа', int(tips['size'].min()), int(tips['size'].max()), (int(tips['size'].min()), int(tips['size'].max())))
filtered_size = tips[(tips['size'] >= size_range[0]) & (tips['size'] <= size_range[1])]
plot2 = px.scatter(filtered_size, x="total_bill", y="tip", color="size")
plot2.update_xaxes(title_text="Сумма счета ($)")
plot2.update_yaxes(title_text="Сумма чаевых ($)")  
st.plotly_chart(plot2)
# plot2 = sns.relplot(data=filtered_size, x="total_bill", y="tip", hue="size")
# plot2.set_axis_labels(x_var='Сумма счета', y_var='Сумма чаевых')
# st.pyplot(plot2)
# plt.close()



st.subheader("Связь суммы счета и дня недели")
selected_days = st.multiselect('Выберите дни недели', tips['day'].unique(), default=tips['day'].unique())
filtered_tips1 = tips[tips['day'].isin(selected_days)]
fig2 = px.scatter(filtered_tips1, x='total_bill', y='tip', color='day')
fig2.update_xaxes(title_text="Сумма счета ($)")
fig2.update_yaxes(title_text="Сумма чаевых ($)")  
st.plotly_chart(fig2)


st.subheader("Распределение чаевых во времени")
fig3 = px.histogram(tips, x="tip", facet_col="day", color="time", 
                   histnorm='probability density', marginal="rug")
st.plotly_chart(fig3)

days = tips['day'].unique().tolist()
selected_day = st.selectbox("Выберите день:", days)

for day in days:
    filtered_tips = tips[tips['day'] == day]

    if day == selected_day:

        fig4 = px.histogram(filtered_tips, x="tip", color="time",
                           histnorm='probability density', marginal="rug",
                           height=350, width=500)

        fig4.update_layout(
            title=f"График зависимости чаевых от времени для дня {day}",
            xaxis_title="Чаевые ($)",
            yaxis_title="Плотность вероятности"
        )

        st.plotly_chart(fig4)



# plot3 = sns.stripplot(data=filtered_tips, x="total_bill", y="day", hue="day", dodge=True)
# plot3.set(xlabel='Сумма счета', ylabel='День недели')
# st.pyplot(plt.gcf())
# plt.close()


st.subheader("Связь суммы чаевых, дня недели и пола клиента")
selected_sex = st.multiselect('Выберите пол клиента', tips['sex'].unique(), default=tips['sex'].unique())
filtered_sex= tips[tips['sex'].isin(selected_sex)]
plot4 = px.strip(filtered_sex, x="tip", y="day", color="sex", color_discrete_map={"Male": "blue", "Female": "pink"})
plot4.update_xaxes(title_text="Сумма счета ($)")
plot4.update_yaxes(title_text="День недели")  
st.plotly_chart(plot4)

# plot4 = sns.stripplot(data=filtered_sex, x="tip", y="day", hue="sex", dodge=True)
# plot4.set(xlabel='Сумма чаевых', ylabel='День недели')
# st.pyplot(plt.gcf())
# plt.close()

st.subheader("Связь суммы чаевых, суммы счета пола клиента и его привычки курить")
plot5 = px.scatter(tips, x="total_bill", y="tip", color="smoker", facet_col="sex", color_discrete_map={"Yes": "blue", "No": "yellow"})
plot5.update_xaxes(title_text="Сумма счета ($)")
plot5.update_yaxes(title_text="Сумма чаевых ($)")  
st.plotly_chart(plot5)

# sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker", col="sex")


st.subheader("Тепловая карта зависимостей")

dct1 = {'Male': 0, 'Female': 1}
dct2 = {'No': 0, 'Yes': 1}
dct3 = {'Dinner': 0, 'Lunch': 1}
tips1 = tips.copy()
tips1['sex_code'] = tips['sex'].map(dct1)
tips1['smoker_code'] = tips['smoker'].map(dct2)
tips1['time_code'] = tips['time'].map(dct3)

def day_code(df, feature):
    for i in df[feature].unique():
        df[i] = (df[feature] == i).astype(int)

day_code(tips1, 'day')

fig5 = px.imshow(tips1.corr(numeric_only=True),
                labels=dict(x="Признаки", y="Признаки", color="Корреляция"),
                x=tips1.corr(numeric_only=True).columns,
                y=tips1.corr(numeric_only=True).columns,
                color_continuous_scale='RdBu', text_auto=True,
                zmin=-1, zmax=1)

fig5.update_layout(width=800, height=600)
st.plotly_chart(fig5)

if st.button('Показать датафрейм tips'):
    st.write(tips)