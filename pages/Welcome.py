import streamlit as st

st.title(":red[Mushroom] Classifier 🍄")


st.write("It uses the UCI Mushroom  [dataset](https://archive.ics.uci.edu/dataset/73/mushroom)")
st.write("and a **Logistic Regression** model to classify mushrooms as edible or poisonous.")

if st.button("Go to Guide"):
    st.switch_page("pages/guide.py") 

if st.button("Test the App"):
    st.switch_page("pages/Classifier.py")  