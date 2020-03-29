#!/usr/bin/env python
# coding: utf-8

# # Analysis of similarity of measured data
# 
# Computational notebook 06 for **Morphological tessellation as a way of partitioning space: Improving consistency in urban morphology at the plot scale**.
# 
# 
# Fleischmann, M., Feliciotti, A., Romice, O. and Porta, S. (2020) _‘Morphological tessellation as a way of partitioning space: Improving consistency in urban morphology at the plot scale’_, Computers, Environment and Urban Systems, 80, p. 101441. doi: [10.1016/j.compenvurbsys.2019.101441](http://doi.org/10.1016/j.compenvurbsys.2019.101441).
# 
# Contact: martin@martinfleischmann.net
# 
# Date: 29/03/2020
# 
# Note: notebook has been cleaned and released retroactively. It is likely that different versions of packages were initially used, but we made sure that the results remained unaltered.
# 
# ---
# **Description**
# 
# This notebook generates figure 14, the illustration of tessellation-based contiguity matrix.
# 
# ---
# **Data**
# 
# 
# The source of the data used wihtin the research is the Amtliche Vermessung dataset accessible from the Zurich municipal GIS open data portal (https://maps.zh.ch). From it can be extracted the cadastral layer (`Liegenschaften_Liegenschaft_Area`) and the layer of buildings (all features named `Gebäude`). All data are licensed under CC-BY 4.0.
# 
# Source data: Vektor-Übersichtsplan des Kantons Zürich, 13.03.2018, Amt für Raumentwicklung Geoinformation / GIS-Produkte, Kanton Zürich, https://opendata.swiss/de/dataset/vektor-ubersichtsplan1
# 
# --
# 
# Data structure:
# 
# ```
# data/
#     contiguity_diagram.gpkg - samples to be used in diagram
#         blg_s
#         tess_s
#         blg_c
#         tess_c
# ```

# In[1]:


import geopandas as gpd
import libpysal
from splot.libpysal import plot_spatial_weights
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


path = (
    "data/contiguity_diagram.gpkg"
)

blg_s = gpd.read_file(path, layer="blg_s")
tess_s = gpd.read_file(path, layer="tess_s")
blg_c = gpd.read_file(path, layer="blg_c")
tess_c = gpd.read_file(path, layer="tess_c")

blg = pd.concat([blg_s, blg_c])
tess = pd.concat([tess_s, tess_c])


blg = blg.sort_values("uID")
blg.reset_index(inplace=True)
tess = tess.loc[tess["uID"].isin(blg["uID"])]

tess = tess.sort_values("uID")
tess.reset_index(inplace=True)

weights = libpysal.weights.contiguity.Queen.from_dataframe(tess)

f, ax = plt.subplots(figsize=(20, 10))
tess.plot(ax=ax)
plot_spatial_weights(weights, blg, ax=ax)
#plt.savefig(
#    "contiguity_diagram.svg",
#    dpi=300,
#    bbox_inches="tight",
#)


# In[ ]:




