import os
import pandas as pd
import streamlit
import streamlit as st
import plotly.express as px
import seaborn as sns
import mlflow
import json
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models import infer_signature

st.set_page_config(layout="wide", page_title="Copper Set Modelling Machine Learning Task")
# st.container()

st.markdown("<h2 style='font-family: san-serif; color: red; text-align: center;'>"
            "Copper Set Modelling Machine Learning Task</h2>", unsafe_allow_html=True)

task = st.selectbox(label="Select ML Task", options=["Select", "Regression", "Classification"])
# st.markdown("", unsafe_allow_html=True)
if task != "Select":
    pass
    before_modeling_training = True
    modeling_training = False
    st.divider()


    @st.cache_data
    def load_excel():
        return pd.read_excel("./Copper_Set.xlsx")


    data = load_excel()
    # def classification_data():
    #     return pd.read_csv("./classification_data.csv", low_memory=False)
    st.markdown("<style>p:hover{background-color: grey;}</style>", unsafe_allow_html=True)
    if task == "Classification":
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Data Ingestion </p>",
                        unsafe_allow_html=True)
            classification_data = data.loc[:, ["id", "item_date", "quantity tons", "customer", "country", "status"]]
            classification_data = classification_data[classification_data["status"].isin(["Won", "Lost"])]
            # classification_data.reset_index(inplace=True)
            # if os.path.exists("./classification_data.csv"):
            #     os.remove("./classification_data.csv")
            # classification_data.to_csv("./classification_data.csv", index=False)
            st.table(classification_data.head(10))

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Show Missing & Nan Count </p>",
                        unsafe_allow_html=True)
            st.table(classification_data.isnull().sum())

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Handle Missing Values </p>",
                        unsafe_allow_html=True)
            classification_data["item_date"] = classification_data["item_date"].fillna(
                classification_data["item_date"].mode()[0])
            classification_data["customer"] = classification_data["customer"].fillna(
                classification_data["customer"].mode()[0])
            classification_data["country"] = classification_data["country"].fillna(
                classification_data["country"].mode()[0])
            classification_data["status"] = classification_data["status"].fillna(
                classification_data["status"].mode()[0])
            classification_data["item_year"] = classification_data["item_date"].astype(str).str.slice(0, 4)
            classification_data["item_month"] = classification_data["item_date"].astype(str).str.slice(4, 6)
            classification_data["item_day"] = classification_data["item_date"].astype(str).str.slice(6, 8)
            classification_data = classification_data[
                ~(classification_data["item_month"] == "00") | (classification_data["item_day"] == "00")]
            classification_data = classification_data[~classification_data.loc[:, "quantity tons"].apply(
                lambda x: x == "e")]
            classification_data.drop(labels=["item_date", "id"], axis=1, inplace=True)
            # if os.path.exists("./classification_data.csv"):
            #     os.remove("./classification_data.csv")
            classification_data["customer"] = classification_data["customer"].astype(int).astype(str)
            classification_data["country"] = classification_data["country"].astype(int).astype(str)
            classification_data["quantity tons"] = classification_data["quantity tons"].astype(float)
            classification_data.to_csv("./classification_data.csv", index=False)
            st.markdown("<h5 style='color: red;'>After Filling Values</h%>", unsafe_allow_html=True)
            st.table(classification_data.isnull().sum())
            st.markdown("<h5 style='color: red;'>Remove Unnecessary Features</h%>", unsafe_allow_html=True)
            st.table(classification_data.head(10))

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Visualize the feature as Charts </p>",
                        unsafe_allow_html=True)
            st.markdown("<h5 style='color: red;'>target Feature as Bar Chart</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(classification_data, x="status"), use_container_width=True, theme=None)
            # plt = plt.figure(figsize=(10, 10))
            # sns.boxplot(classification_data.sample(1000), x="status", y="quantity tons")
            # st.pyplot(plt)
            st.markdown("<h5 style='color: red;'>Compare Quantity Tons & status Feature as Box Chart</h5>",
                        unsafe_allow_html=True)
            st.plotly_chart(px.box(classification_data.sample(500), x="status", y="quantity tons"),
                            use_container_width=True, theme=None)
            st.markdown("<h5 style='color: red;'>Country Feature as Bar Chart</h5>",
                        unsafe_allow_html=True)
            st.plotly_chart(px.bar(classification_data["country"].value_counts()), use_container_width=True, theme=None)

            # refresh = st.button("scatter plot refresh")
            # if refresh:
            #     st.plotly_chart(px.scatter(classification_data.sample(30000), x="customer", y="quantity tons",
            #                                hover_data="status"))
            # else:
            st.markdown("<h5 style='color: red;'>Compare Quantity Tons & status Feature as Scatter Chart</h5>",
                        unsafe_allow_html=True)
            st.plotly_chart(px.scatter(classification_data.sample(30000), x="customer", y="quantity tons",
                                       hover_data="status"), theme=None)
            CLT_data = pd.Series([classification_data.loc[:, "quantity tons"].sample(n=100, replace=True).mean()
                                  for i in range(0, 500)])
            st.markdown("<h4 style='color: red;' > Kernel Density DistPlot </h4>", unsafe_allow_html=True)
            st.plotly_chart(ff.create_distplot([CLT_data], group_labels=["quantity tons"]), theme=None)
            st.text("Positive Skewness is asymmetrical distribution (mean > median > mode)")
            st.text(f"check the condition for positive or right skewness (mean > median > mode) is {CLT_data.mean()} > "
                    f"{CLT_data.median()} > {stat.mode(CLT_data).mode}")

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Handling Outlier </p>",
                        unsafe_allow_html=True)
            upper_limit_of_z_score = classification_data["quantity tons"].mean() + (3 * classification_data[
                "quantity tons"].std())
            lower_limit_of_z_score = classification_data["quantity tons"].mean() - (3 * classification_data[
                "quantity tons"].std())
            st.markdown("<h5 style='color: red;'> Showing Outlier using Z-Score mean is 0 & std is 1 </h5>",
                        unsafe_allow_html=True)
            st.table(classification_data[(classification_data["quantity tons"] > upper_limit_of_z_score) |
                                         (classification_data["quantity tons"] < lower_limit_of_z_score)].head(10))

            classification_data["quantity tons"] = np.where(classification_data["quantity tons"] >
                                                            upper_limit_of_z_score, upper_limit_of_z_score,
                                                            np.where(classification_data["quantity tons"] <
                                                                     lower_limit_of_z_score, lower_limit_of_z_score,
                                                                     classification_data["quantity tons"]))

            st.markdown("<h5 style='color: red;'>After Handling Outlier Compare Quantity Tons & status Feature as "
                        "Box Chart</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.box(classification_data.sample(500), x="status", y="quantity tons"),
                            use_container_width=True, theme=None)
            # if os.path.exists("./classification_data.csv"):
            #     os.remove("./classification_data.csv")
            # classification_data.to_csv("./classification_data.csv", index=False)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > "
                        "Handle Feature to Number by using Label Encoder </p>",
                        unsafe_allow_html=True)
            # Status
            label_encoder = LabelEncoder()
            encoder_values = label_encoder.fit_transform(classification_data.status.drop_duplicates())
            status_class_encoder_dict = {}
            for i in range(len(label_encoder.classes_)):
                status_class_encoder_dict[label_encoder.classes_[i]] = int(encoder_values[i])
            classification_data["status"] = classification_data["status"].apply(lambda x: status_class_encoder_dict[x])
            if os.path.exists("./status_encoder.json"):
                os.remove("./status_encoder.json")
            with open("./status_encoder.json", "w") as file:
                file.write(json.dumps(status_class_encoder_dict))

            # Customer
            cust_class_encoder_dict = {j: i + 1 for i, j in classification_data[
                "customer"].drop_duplicates().reset_index(drop=["index"]).items()}
            classification_data["customer"] = classification_data["customer"].apply(
                lambda x: cust_class_encoder_dict[x])
            if os.path.exists("./customer_encoder.json"):
                os.remove("./customer_encoder.json")
            with open("./customer_encoder.json", "w") as file:
                file.write(json.dumps(cust_class_encoder_dict))

            # Country
            country_class_encoder_dict = {j: i + 1 for i, j in classification_data[
                "country"].drop_duplicates().reset_index(drop=["index"]).items()}
            classification_data["country"] = classification_data["country"].apply(lambda x:
                                                                                  country_class_encoder_dict[x])
            if os.path.exists("./country_encoder.json"):
                os.remove("./country_encoder.json")
            with open("./country_encoder.json", "w") as file:
                file.write(json.dumps(country_class_encoder_dict))

            # if os.path.exists("./classification_data.csv"):
            #     os.remove("./classification_data.csv")
            # classification_data.to_csv("./classification_data.csv", index=False)
            st.table(classification_data.head(10))

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > "
                        "Apply Log Transformation </p>",
                        unsafe_allow_html=True)
            log_transformation = FunctionTransformer(func=np.log1p)
            classification_data["quantity tons"] = classification_data["quantity tons"].apply(
                lambda x: x if x > 0 else float(classification_data["quantity tons"].median()))
            classification_data["quantity tons"] = log_transformation.fit_transform(
                classification_data["quantity tons"])
            # if os.path.exists("./classification_data.csv"):
            #     os.remove("./classification_data.csv")
            # classification_data.to_csv("./classification_data.csv", index=False)
            st.plotly_chart(px.scatter(classification_data.sample(3000), x="customer", y="quantity tons",
                                       hover_data="status"), theme=None)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > "
                        " Scaling & Data Splitting </p>",
                        unsafe_allow_html=True)
            X = classification_data.drop(labels=["status"], axis=1)
            y = classification_data.loc[:, "status"]
            std_scaler = MinMaxScaler()
            X.loc[:, ["customer", "country", "quantity tons", "item_year", "item_month",
                      "item_day"]] = std_scaler.fit_transform(X[["customer", "country", "quantity tons",
                                                                 "item_year", "item_month", "item_day"]])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            X_train["status"] = y_train
            X_test["status"] = y_test
            st.table(X)
            if not os.path.exists("./classification"):
                os.makedirs("classification")
            if os.path.exists("./classification/training_data.csv"):
                os.remove("./classification/training_data.csv")
            if os.path.exists("./classification/training_data.csv"):
                os.remove("./classification/testing_data.csv")
            X_train.to_csv("./classification/training_data.csv", index=False)
            X_test.to_csv("./classification/testing_data.csv", index=False)

        with st.container():
            model_training = st.button("Model Training With Hyperparameter Tuning (Cross Validation)",
                                       disabled=modeling_training)
            models = {
                "LogisticRegression": LogisticRegression(),
                "RandomForestClassifier": RandomForestClassifier(random_state=42),
                "AdaBoostClassifier": AdaBoostClassifier(),
            }
            if model_training:
                # model_uri = 'runs:/cdedb0f2d5f845abbaad4ac0ffcb4401/best_estimator'
                model_name = "LogisticRegression"
                model_version = "1"
                model_uri = f"models:/{model_name}/{model_version}"
                logistic_regression = mlflow.sklearn.load_model(model_uri)
                if not logistic_regression:
                    with mlflow.start_run(experiment_id="901191393770180809", run_name="LogisticRegression") as run_1:
                        mlflow.autolog()
                        mlflow.doctor()
                        cv = KFold(n_splits=20, shuffle=True, random_state=None)
                        LogisticRegressionParams = {
                            "penalty": ["l1", "l2", "elasticnet"],
                            "max_iter": list(range(100, 200)),
                            "random_state": list(range(1, 50)),
                            "n_jobs": list(range(5, 10)),
                            "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                            "class_weight": ["balanced"],
                            "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        }
                        logistic_tuning_model = RandomizedSearchCV(estimator=LogisticRegression(),
                                                                   param_distributions=LogisticRegressionParams,
                                                                   n_jobs=2,
                                                                   cv=cv, scoring="accuracy", verbose=50, n_iter=10)
                        logistic_tuning_model.fit(X_train, y_train)
                        mlflow.log_metric("testing_accuracy",
                                          accuracy_score(y_test, logistic_tuning_model.predict(X_test)))
                else:
                    test_prediction = logistic_regression.predict(X_test)
                    confusion_matrix(y_test, test_prediction)
                # model_uri = 'runs:/2663c5173397461db6a2be5ea747813e/best_estimator'
                model_name = "RandomForestClassifier"
                model_version = "1"
                model_uri = f"models:/{model_name}/{model_version}"
                random_forest = mlflow.sklearn.load_model(model_uri)
                if not random_forest:
                    with mlflow.start_run(experiment_id="901191393770180809",
                                          run_name="RandomForestClassifier") as run_2:
                        mlflow.autolog()
                        mlflow.doctor()
                        cv = KFold(n_splits=20, shuffle=True, random_state=None)
                        random_forest_params = {
                            "criterion": ["gini"],
                            # "splitter": ["best", "random"],
                            "max_depth": list(range(1, 50)),
                            "min_samples_split": list(range(1, 121)),
                            "max_features": ["sqrt", "log2"],
                            "n_estimators": list(range(1, 200)),
                            "n_jobs": list(range(1, 10)),
                            "random_state": list(range(1, 50))
                        }
                        random_forest_tuning_model = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                                                        param_distributions=random_forest_params,
                                                                        n_jobs=2,
                                                                        cv=cv,
                                                                        scoring="accuracy", verbose=3, n_iter=10)
                        random_forest_tuning_model.fit(X_train, y_train)
                        mlflow.log_metric("testing_accuracy", accuracy_score(y_test,
                                                                             random_forest_tuning_model.predict(
                                                                                 X_test)))
                else:
                    test_prediction = random_forest.predict(X_test)
                    confusion_matrix(y_test, test_prediction)
                # model_uri = 'runs:/70d0f8755d9d4cca9983335ccf557f79/best_estimator'
                model_name = "AdaBoostClassifier"
                model_version = "1"
                model_uri = f"models:/{model_name}/{model_version}"
                ada_boost = mlflow.sklearn.load_model(model_uri)
                if not ada_boost:
                    with mlflow.start_run(experiment_id="901191393770180809", run_name="AdaBoostClassifier") as run_3:
                        mlflow.autolog()
                        mlflow.doctor()
                        cv = KFold(n_splits=20, shuffle=True, random_state=None)
                        adaboost_param = {
                            "n_estimators": list(range(50, 100)),
                            "learning_rate": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
                            "random_state": list(range(10, 50))
                        }
                        adaboost_random_tuning_model = RandomizedSearchCV(estimator=AdaBoostClassifier(
                            estimator=DecisionTreeClassifier(criterion="gini", max_depth=10, max_features="log2",
                                                             random_state=44)),
                            param_distributions=adaboost_param, n_jobs=9, cv=cv, scoring="accuracy", verbose=3,
                            n_iter=10)
                        adaboost_random_tuning_model.fit(X_train, y_train)
                        y_pred = adaboost_random_tuning_model.predict(X_test)
                        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
                        mlflow.log_metric("testing_accuracy", accuracy)
                else:
                    test_prediction = ada_boost.predict(X_test)
                    confusion_matrix(y_test, test_prediction)
    elif task == "Regression":
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Data Ingestion </p>",
                        unsafe_allow_html=True)
            regression_data = data.loc[:, ["item type", "application", "thickness", "width", "material_ref",
                                           "product_ref", "delivery date", "selling_price"]]
            st.table(regression_data.head(10))
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Show Missing & Nan Count </p>",
                        unsafe_allow_html=True)
            regression_data.drop_duplicates(inplace=True, keep="first")
            st.table(regression_data.isnull().sum())
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Handle Missing Values </p>",
                        unsafe_allow_html=True)
            regression_data["delivery date"] = regression_data["delivery date"].fillna(
                regression_data["delivery date"].mode()[0])
            regression_data["selling_price"] = regression_data["selling_price"].fillna(
                regression_data["selling_price"].median())
            regression_data["thickness"] = regression_data["thickness"].fillna(regression_data["thickness"].median())
            regression_data["application"] = regression_data["application"].fillna(
                regression_data["application"].mode()[0])
            regression_data["delivery_date_year"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                0, 4).astype(int)
            regression_data["delivery_date_month"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                4, 6).astype(int)
            regression_data["delivery_date_day"] = regression_data["delivery date"].astype(int).astype(str).str.slice(
                6, 8).astype(int)
            regression_data = regression_data[~(regression_data["delivery_date_month"] == 22)]
            regression_data.drop(labels=["product_ref", "material_ref", "delivery date"], axis=1, inplace=True)
            st.table(regression_data.head(10))
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' >Visualize the categorical features</p>",
                        unsafe_allow_html=True)
            cate_item = pd.DataFrame(regression_data["item type"].value_counts())
            st.markdown("<h5 style='color: red;'>Item Type feature</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(cate_item, x=cate_item.index, y="count"), theme=None)
            application = pd.DataFrame(regression_data["application"].value_counts())
            st.markdown("<h5 style='color: red;'>Application feature</h5>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(application, x=application.index, y="count"), theme=None)

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' >Handle Categorical feature</p>",
                        unsafe_allow_html=True)
            item_type_class_encoder_dict = {j: i + 1 for i, j in regression_data[
                "item type"].drop_duplicates().reset_index(drop=["index"]).items()}
            regression_data["item type"] = regression_data["item type"].apply(
                lambda x: item_type_class_encoder_dict[x])
            if os.path.exists("./item_type.json"):
                os.remove("./item_type.json")
            with open("./item_type.json", "w") as file:
                file.write(json.dumps(item_type_class_encoder_dict))
            st.table(regression_data.head(10))

        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Identify Skewness as distplot </p>",
                        unsafe_allow_html=True)
            CLT_width_data = pd.Series([regression_data.loc[:, "width"].sample(n=100, replace=True).mean()
                                        for i in range(0, 500)])
            st.text(f"skewness Of Width is {str(CLT_width_data.skew())}")
            st.plotly_chart(ff.create_distplot([CLT_width_data], group_labels=["Width"]), theme=None)
            st.text("Negative Skewness is asymmetrical distribution (median > mean  > mode)")
            st.text(f"check the condition for negative or left skewness (median > mean > mode) is "
                    f"{CLT_width_data.median()} > {CLT_width_data.mean()} > "
                    f"{stat.mode(CLT_width_data).mode}")

            CLT_thickness_data = pd.Series([regression_data.loc[:, "thickness"].sample(n=100, replace=True).mean()
                                            for i in range(0, 1000)])
            st.text(f"skewness Of Thickness is {str(CLT_thickness_data.skew())}")
            st.plotly_chart(ff.create_distplot([CLT_thickness_data], group_labels=["Thickness"]), theme=None)
            st.text("Positive Skewness is asymmetrical distribution (mean > median > mode)")
            st.markdown(f"<h5 style='color: red;'>check the condition for positive or right skewness (mean > median > mode) is "
                    f"{CLT_thickness_data.mean()} > {CLT_thickness_data.median()} > "
                    f"{stat.mode(CLT_thickness_data).mode}</h5", unsafe_allow_html=True)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem;' > "
                        "showing outlier & handling outlier </p>", unsafe_allow_html=True)
            st.markdown("<h4> Before Handling Outlier </h4>", unsafe_allow_html=True)
            st.markdown("<h5> Violin Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.violin(regression_data, x="width", box=True), theme=None)
            st.plotly_chart(px.violin(regression_data, x="thickness", box=True), theme=None)
            st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.box(regression_data, x="width"), theme=None)
            st.plotly_chart(px.box(regression_data, x="thickness"), theme=None)
            lower_width_limit = regression_data["width"].mean() - 3 * regression_data["width"].std()
            upper_width_limit = regression_data["width"].mean() + 3 * regression_data["width"].std()

            # Count of Width Outlier
            count_of_outlier = regression_data[
                (regression_data["width"] < lower_width_limit) |
                (regression_data["width"] > upper_width_limit)].shape[0]
            st.text(f"Count Of Width Outlier is {str(count_of_outlier)}")
            # Handling Width Outlier
            regression_data["width"] = np.where(regression_data["width"] < lower_width_limit,
                                                lower_width_limit,
                                                np.where(regression_data["width"] > upper_width_limit,
                                                         upper_width_limit, regression_data["width"]))

            lower_thickness_limit = regression_data["thickness"].mean() - 3 * regression_data["thickness"].std()
            upper_thickness_limit = regression_data["thickness"].mean() + 3 * regression_data["thickness"].std()
            # Count of thickness Outlier
            count_of_outlier = regression_data[
                (regression_data["thickness"] < lower_thickness_limit) |
                (regression_data["thickness"] > upper_thickness_limit)].shape[0]
            st.text(f"Count Of thickness Outlier is {str(count_of_outlier)}")
            # Handling thickness Outlier
            regression_data["thickness"] = np.where(regression_data["thickness"] < lower_thickness_limit,
                                                    lower_thickness_limit,
                                                    np.where(regression_data["thickness"] > upper_thickness_limit,
                                                             upper_thickness_limit, regression_data["thickness"]))
            st.markdown("<h4> After Handling Outlier </h4>", unsafe_allow_html=True)
            st.markdown("<h5> Violin Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.violin(regression_data, x="width", box=True), theme=None)
            st.plotly_chart(px.violin(regression_data, x="thickness", box=True), theme=None)
            st.markdown("<h5> Box Plot </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.box(regression_data, x="width"), theme=None)
            st.plotly_chart(px.box(regression_data, x="thickness"), theme=None)
            st.markdown("<h5> Correlation Chart </h5>", unsafe_allow_html=True)
            st.plotly_chart(px.imshow((regression_data.corr()), text_auto=True, height=1000), theme=None)
        with st.container():
            st.markdown("<p style='border: 1px solid red; height: 40px; border-radius:10px; "
                        "text-align:center; cursor:pointer; line-height: 3rem; ' > Applying Transformation & "
                        "Scaling & Split The Data</p>", unsafe_allow_html=True)
            regression_data[["width", "thickness"]] = PowerTransformer().fit_transform(
                regression_data[["width", "thickness"]])
            X = regression_data.drop(labels=["selling_price"], axis=1)
            y = regression_data["selling_price"]
            # st.text(",".join(X.columns))
            X[["item type", "application", "thickness", "width", "delivery_date_year", "delivery_date_month",
               "delivery_date_day"]] = MinMaxScaler().fit_transform(X)
            st.table(pd.concat([X, y], axis=1).head(10))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            X_train["selling_price"] = y_train
            X_test["selling_price"] = y_test
            if not os.path.exists("./regression"):
                os.makedirs("regression")
            if os.path.exists("./regression/training_data.csv"):
                os.remove("./regression/training_data.csv")
            if os.path.exists("./regression/testing_data.csv"):
                os.remove("./regression/testing_data.csv")
            X_train.to_csv("./regression/training_data.csv", index=False)
            X_test.to_csv("./regression/testing_data.csv", index=False)




# st.plotly_chart(px.imshow(regression_data.corr(), text_auto=True))
# # st.markdown("<h5> Showing HeatMap Chart </h5>", unsafe_allow_html=True)
# #             st.plotly_chart(px.density_heatmap(classification_data.corr(), text_auto=True))
