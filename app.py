import warnings warnings.filterwarnings("ignore")
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib matplotlib,use("Agg")
import matplotlib.patches as mpatches
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="New Ridge Sales Interface",
    page_icon="📦",
    layout="wide",
)

LEAD_TIME_DAYS     = 14
SAFETY_BUFFER_DAYS = 7
REORDER_POINT_DAYS = LEAD_TIME_DAYS + SAFETY_BUFFER_DAYS

MONTH_COLS = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sept","Oct","Nov","Dec"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"]  = ["Times New Roman", "DejaVu Serif"]
TITLE_SIZE = 14
LABEL_SIZE = 11
COLOR_CRIT = "#d62728"
COLOR_WARN = "#ff7f0e"
COLOR_OK   = "#2ca02c"
PALETTE    = {"CRITICAL": COLOR_CRIT, "WARNING": COLOR_WARN,
              "OK": COLOR_OK, "NO_MOVEMENT": "#aaaaaa"}