import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload, visualize_influenza, visualize_mers, visualize_measles  # import your pages here

# Create an instance of the app 
app = MultiPage()


# Add all your application here
app.add_page("Creating the model", data_upload.app)
app.add_page("Forecast Influenza", visualize_influenza.app)
app.add_page("Forecast Mers", visualize_mers.app)
app.add_page("Forecast Measles", visualize_measles.app)
#app.add_page("Forecast New Disease", forecast.app)

# The main app
app.run()