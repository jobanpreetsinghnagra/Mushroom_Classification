import streamlit as st


# --- PAGE SETUP ---
welcome_page = st.Page(
    "pages/Welcome.py",
    title="Welcome",
    icon=":material/cottage:",
    default=True,
)
project_1_page = st.Page(
    "pages/Guide.py",
    title="Guide",
    icon=":material/article:",
)
project_2_page = st.Page(
    "pages/Classifier.py",
    title="App",
    icon=":material/robot:",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
pg = st.navigation(pages=[welcome_page, project_1_page, project_2_page])

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 30px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SHARED ON ALL PAGES ---
st.sidebar.markdown("Made with ❤️ by [Joban](https://x.com/j0bannagra)")


# --- RUN NAVIGATION ---
pg.run()