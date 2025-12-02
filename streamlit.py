import streamlit as st
from empLinearRegression import empLinearRegression as emplr
import numpy as np


def main():
    x, y, model = train_model( filename="Salary.csv", link="https://raw.githubusercontent.com/Vishal-Singh27/Salary-Predictor/refs/heads/main/Salary.csv")

    start_site(model=model, fig=model.visualize(x, y))

    
@st.cache_resource
def train_model(filename, link=None):
    model = emplr(filename=filename, link=link)
    x = np.array(model.data["YearsExperience"])
    y = np.array(model.data["Salary"])
    model.train(x, y, alpha=0.01)
    return x, y, model



def start_site(fig, model):
    st.markdown("# CIA 1 - Linear Regression: Simple & Multiple")
    st.markdown("### Name: Vishal Singh | Class: 2MCA | Reg. No.: 25225028")
    exp = st.slider("Years of experience", 1, 100)

    if exp:
        st.markdown("#### Prediction is: " + str(model.predict(np.float64(exp))))
        st.header("Graph of the predictions")
        st.pyplot(fig)

if __name__ == "__main__":
    main()