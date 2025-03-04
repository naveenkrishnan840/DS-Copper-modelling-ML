import streamlit as st
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from mlflow.sklearn import load_model
import json
st.markdown("<h2 style='font-family: san-serif; color: red; text-align: center;'>"
            "Model Prediction</h2>", unsafe_allow_html=True)


task = st.selectbox(label="Select ML Task for prediction", options=["Select", "Regression", "Classification"])

if task != "Select":
    if task == "Regression":
        with st.container():
            cols = st.columns(5)
            with cols[0]:
                item_type = st.selectbox(label="Item Type", options=["W", "WI", "S", "PL", "IPL", "SLAWR", "Others"])
            with cols[1]:
                application = st.selectbox(label="Application", options=["W", "WI", "S", "PL", "IPL", "SLAWR", "Others"])
            with cols[2]:
                thickness = st.number_input(label="thickness")
            with cols[3]:
                width = st.number_input(label="width")
            with cols[4]:
                delivery = st.date_input(label="delivery Date")
    elif task == "Classification":
        st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                    "text-align:center; cursor:pointer; line-height: 3rem; ' > "
                    "Based On Model Training We should take classification Task Model Version 9. Version 9 is very "
                    "good accuracy for both training & testing</p>",
                    unsafe_allow_html=True)
        cols = st.columns(5)
        with cols[0]:
            with open("../country_encoder.json") as file:
                country = json.loads(file)
            item_type = st.selectbox(label="Country", options=list(country.key()))
        with cols[1]:
            with open("../customer_encoder.json") as file:
                customer = json.loads(file)
            application = st.selectbox(label="Customer", options=list(customer.key()))
        with cols[3]:
            width = st.number_input(label="Quantity Tons")
        with cols[4]:
            delivery = st.date_input(label="Item Date")

        @st.cache_data
        def load_model():
            return load_model(model_uri="models:/Classfication Task/9")

        classification_pipeline = make_pipeline(StandardScaler(), load_model())

