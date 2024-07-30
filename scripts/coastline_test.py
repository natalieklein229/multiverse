
from quality_of_life.my_base_utils import find_root_dir_of_repo, my_warn
from quality_of_life.my_plotly_utils import vector_viz
from bnns.utils import load_coast_coords
from matplotlib import pyplot as plt
from plotly import graph_objects as go
import os
os.chdir( os.path.join(find_root_dir_of_repo(), "bnns", "data" ) )

#
# ~~~ Plot coastline
try:
    os.chdir("ne_10m_coastline")
    c = load_coast_coords("ne_10m_coastline.shp")
    x, y = c[:,0], c[:,1]
    plt.scatter(x,y)
    plt.show()
    os.chdir("..")
except FileNotFoundError:
    my_warn("In order to plot the coastline, go to https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/ and click the `Download coastline` button. Unzip the folder, and move the unzipped folder called `ne_10m_coastline` into the folder bnns/bnns/data")


#
# ~~~ Plot a heatmap
from slosh_70_15_15 import out_np, coords_np
vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0] )
vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0], graph_object=go.Heatmap )

