import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ---------------- Title ----------------
st.title("🌸 Iris Flower Species Prediction")
st.caption("Enter flower measurements to predict the species")
st.divider()

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Iris_dataset.csv")

dataset = load_data()

X = dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
Y = dataset["Species"]

# ---------------- Train-Test Split ----------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, random_state=42
)

# ---------------- Train Model ----------------
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    return model

model = train_model()

# ---------------- Sidebar Inputs (Number) ----------------
st.sidebar.header("🔢 Input Measurements (cm)")

sepal_length = st.sidebar.number_input(
    "Sepal Length", min_value=4.0, max_value=8.0, value=5.0, step=0.1
)
sepal_width = st.sidebar.number_input(
    "Sepal Width", min_value=2.0, max_value=4.5, value=3.3, step=0.1
)
petal_length = st.sidebar.number_input(
    "Petal Length", min_value=1.0, max_value=7.0, value=1.4, step=0.1
)
petal_width = st.sidebar.number_input(
    "Petal Width", min_value=0.1, max_value=2.5, value=0.2, step=0.1
)

# ---------------- Prediction ----------------
st.subheader("🔮 Prediction Result")

if st.sidebar.button("🚀 Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data).max()

    col1, col2 = st.columns(2)
    col1.metric("Predicted Species", prediction)
    col2.metric("Confidence", f"{confidence:.2%}")
else:
    st.info("Enter numeric values in the sidebar and click **Predict Species**.")

st.divider()

# ---------------- Dataset View ----------------
with st.expander("📂 View Dataset"):
    st.dataframe(dataset, use_container_width=True)
