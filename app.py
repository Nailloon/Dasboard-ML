import pickle
import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    model_KNN = load_model("model_classifier1.h5")
    model_Bagging = load_model("model_bagging2.h5")
    model_Neuro = load_model_tfkeras("model_classification_neuro.h5")
    test_data = load_test_data("card_transdata.csv")
    data1 = load_data("card_transdata.csv")
    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Визуализация зависимостей" ,"Запрос к моделям"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""Набор данных представляет собой транзакции, совершенные кредитными картами, необходимо определить является ли операция мошеннической или нет""")

        st.write(data1.iloc[:10, :])

        st.header("Описание данных")
        st.markdown("""
Предоставленные бинарные признаки:
* fraud - совершена транзакция мошенником(1) или обычным человеком(0),
* repeat_retailer - это транзакция совершена у того же продавца(1) или впервые(0).
* used_chip - это транзакция прошла с помощью чипа (кредитной карты)(1) или без использования чипа(0).
* used_pin_number - для этой транзакции был введен PIN-код(1) или без PIN-кода(0).
* online_order - является транзакция онлайн-заказом(1) или офлайн-заказом(0).

Числовые признаки:

* distance_from_home - расстояние от дома до места, где произошла транзакция.
* distance_from_last_transaction - расстояние с места совершения последней транзакции до места совершения текущей транзакции.
* ratio_to_median_purchase_price - отношение транзакции по цене покупки к средней цене покупки.

""")

    elif page == "Запрос к моделям":
        st.title("Запрос к моделям")
        st.write("Выберите страницу слева")

        distance_from_home = st.slider("Задайте расстояние от дома до места совершения последней транзакции, выберите 1, если операция совершена дома: ", min_value=1, max_value=1000, value=1, step=1)
        distance_from_home = float(distance_from_home)

        distance_from_last_transaction = st.slider("Задайте расстояние от места совершения последней транзакции: ", min_value=1, max_value=1000, value=1, step=1)
        distance_from_last_transaction = float(distance_from_last_transaction)

        ratio_to_median_purchase_price = st.number_input("Задайте отношение транзакции по цене покупки к средней цене покупки", 0, 267, 2)
        ratio_to_median_purchase_price = float(ratio_to_median_purchase_price)

        repeat_retailer  = st.selectbox("Эта транзакция совершена у того же продавца? 1 - если да, иначе - 0", [0, 1])
        repeat_retailer  = float(repeat_retailer)

        used_chip  = st.selectbox("Эта транзакция прошла с помощью чипа(кредитной карты)? 1 - если да, иначе - 0", [0, 1])
        used_chip  = float(used_chip)

        used_pin_number  = st.selectbox("Для этой транзакции был введен PIN-код? 1 - если да, иначе - 0", [0, 1])
        used_pin_number  = float(used_pin_number)

        online_order  = st.selectbox("Это онлайн-заказ? 1 - если да, иначе - 0", [0, 1])
        online_order  = float(online_order)

        if st.button('Получить предсказание'):
            data = [distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price, repeat_retailer, used_chip, used_pin_number, online_order]
            data = np.array(data).reshape((1, -1))
            data_df = pd.DataFrame(data)
            pred1 = model_KNN.predict(data_df)
            st.write(f"Значение, предсказанное с помощью KNN: {pred1[0]:.2f}, точность: 0.97")
            pred2 = model_Bagging.predict(data_df)
            st.write(f"Значение, предсказанное с помощью бэггинга: {pred2[0]:.2f}, точность: 0.98")
            pred3 = np.around(model_Neuro.predict(data_df, verbose=None))
            st.write(f"Значение, предсказанное с помощью полносвязной нейронной сети: {pred3[0][0]:.2f}, точность: 0.99, macroavg 0,97")
        else:
            pass
    elif page == "Визуализация зависимостей":
        st.title("Визуализация данных и зависимостей")
        st.write("Выберите страницу слева")
        request = st.selectbox("Выберите запрос",["Оценки плотности вероятности","Диаграмма корреляции", "Гистограмма", "Диаграмма рассеивания", "Круговая диаграмма"])
        if request == "Диаграмма корреляции":
            fig, ax = plt.subplots(figsize=(25,25), dpi= 60)
            sns.heatmap(data1.corr(), ax=ax, annot = True)
            st.write(fig)
        elif request == "Гистограмма":
            fig, ax = plt.subplots(figsize=(25,25), dpi= 60)
            x_title = st.selectbox("Выберите признак", data1.columns)
            if x_title in {"distance_from_home", "distance_from_last_transaction"}:
                plt.hist(data1[x_title], bins = 5000, range=(0, 400), color='green')
            elif x_title in {"repeat_retailer", "fraud", "used_chip", "used_pin_number", "online_order"}:
                plt.hist(data1[x_title], bins = 2, range=(0, 1), color='red')
            elif x_title == "ratio_to_median_purchase_price":
                 plt.hist(data1[x_title], bins = 20, range=(0, 30), color='blue')
            plt.title(x_title)
            st.write(fig)
        elif request == "Диаграмма рассеивания":
            x_axis = st.selectbox("Выберите столбец для x-axis", data1.columns)
            y_axis = st.selectbox("Выберите столбец для y-axis", data1.columns)
            visualize_data(data1, x_axis, y_axis)
        elif request == "Круговая диаграмма":
            labelss = st.selectbox("Выберите столбец", datacolumns("binary"))
            visual(data1, labelss)
        elif request == "Оценки плотности вероятности":
            st.write("kde (kernel density estimation) - это метод оценки плотности вероятности, который используется в статистике для оценки плотности распределения вероятности переменной.")
            st.write("В контексте визуализации графиков, использование параметра kde=True в графиках Seaborn означает, что график будет отображать линию плотности вероятности для каждого значения переменной x, на основе выборки данных.")
            st.write("Например, если вы построите график распределения расстояний от дома до работы для двух групп - мошеннических и не-мошеннических транзакций, и установите параметр kde=True, то график будет содержать две кривые плотности вероятности - одну для каждой группы.")
            st.write("Визуализация линии плотности вероятности может помочь наглядно сравнить форму распределения между различными группами, а также оценить, как она изменяется в зависимости от других переменных.")
            x_title = st.selectbox("Выберите признак", datacolumns("numeric"))
            g = sns.displot(data=data1, x=x_title, hue='fraud', log_scale=True, kde=True)
            chart = g.fig
            st.pyplot(chart)


def datacolumns(typee):
    if typee == "binary":
        return ["repeat_retailer", "fraud", "used_chip", "used_pin_number", "online_order"]
    if typee == "numeric":
        return ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
def visual(data1, labelss):
    fig, ax = plt.subplots()
    data1.groupby(labelss).size().plot(kind='pie')
    plt.show()
    st.write(fig)
def visualize_data(data1, x_axis, y_axis):
    fig, ax = plt.subplots(figsize=(16,14)) 
    ax.scatter(x = data1[x_axis], y = data1[y_axis])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
    st.write(fig)
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = joblib.load(model_file)
    return model

def load_model_tfkeras(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = tf.keras.models.load_model(model_file.name)
    return model
@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file)
    df = df.drop(labels=['fraud'], axis=1)
    return df
def load_data(path_to_file):
    df = pd.read_csv(path_to_file)
    return df
if __name__ == "__main__":
    main()
