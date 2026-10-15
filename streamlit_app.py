"""Entrypoint: page config, then navigation built from the registry."""

import streamlit as st

from lab.nav import build_navigation

st.set_page_config(page_title="Ayden's Lab", page_icon=":material/science:", layout="wide")
st.navigation(build_navigation(), expanded=True).run()
