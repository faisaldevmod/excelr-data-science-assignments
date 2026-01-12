import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data():
    df = pd.read_csv("Titanic_train.csv")
    return df


def preprocess_data(df):
  
    df = df.drop(columns=["Cabin", "PassengerId", "Ticket", "Name"])  # Create a new DataFrame without mutating the original
   
    df = df.dropna()
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    return df

def train_model(df):
    X = df[['Pclass', 'Sex', 'Embarked', 'Age']]
    y = df['Survived']
    model = LogisticRegression()
    model.fit(X, y)
    return model

def predict_survival(model, passenger):
    prediction = model.predict(passenger)
    return prediction

def main():
    st.title("Titanic Survival Prediction")

    df = load_data()

    df = preprocess_data(df)

    model = train_model(df)

    st.sidebar.header("Enter Passenger Information")
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])
    age = st.sidebar.slider("Age", min(df['Age']), max(df['Age']), min(df['Age']))

    le_sex = LabelEncoder()
    sex_encoded = le_sex.fit_transform([sex])[0]

    le_embarked = LabelEncoder()
    embarked_encoded = le_embarked.fit_transform([embarked])[0]

    passenger = [[pclass, sex_encoded, embarked_encoded, age]]
    prediction = predict_survival(model, passenger)

    if prediction[0] == 0:
        st.write("The passenger is predicted not to survive.")
    else:
        st.write("The passenger is predicted to survive.")

if __name__ == "__main__":
    main()