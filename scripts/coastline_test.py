
from quality_of_life.my_base_utils import find_root_dir_of_repo, my_warn
from quality_of_life.my_scipy_utils import extend_to_grid
from bnns.utils import load_coast_coords
from matplotlib import pyplot as plt
# from plotly import graph_objects as go
import numpy as np
import os
os.chdir( os.path.join(find_root_dir_of_repo(), "bnns", "data" ) )

#
# ~~~ Plot coastline
try:
    os.chdir("ne_10m_coastline")
    c = load_coast_coords("ne_10m_coastline.shp")
    coast_x, coast_y = c[:,0], c[:,1]
    plt.scatter(coast_x,coast_y)
    plt.show()
    os.chdir("..")
except FileNotFoundError:
    my_warn("In order to plot the coastline, go to https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/ and click the `Download coastline` button. Unzip the folder, and move the unzipped folder called `ne_10m_coastline` into the folder bnns/bnns/data")


#
# ~~~ Get the data
from bnns.data.slosh_70_15_15 import out_np, coords_np
# vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0] )
x = coords_np[:,0]
y = coords_np[:,1]
z = out_np[10]

#
# ~~~ Plot a heatmap using interpolation
X,Y,Z = extend_to_grid( x, y, z, res=501, method="linear" )
Z = np.nan_to_num(Z,nan=0.)
Z = Z*(Z>0)
plt.figure(figsize=(6,5))
plt.contourf( X, Y, Z, cmap="viridis" )
plt.colorbar(label="Storm Surge Heights")
plt.plot( coast_x, coast_y, color="black", linewidth=1, label="Coastline" )
plt.xlim(X.min(),X.max())
plt.ylim(Y.min(),Y.max())
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heightmap")
plt.legend()
plt.tight_layout()
plt.show()


#
# ~~~ Plot a heatmap as a scatterplot without interpolation
plt.figure(figsize=(6,5))
plt.scatter( x, y, c=z, cmap="viridis" )
plt.colorbar(label="Storm Surge Heights")
plt.plot( coast_x, coast_y, color="black", linewidth=1, label="Coastline" )
plt.xlim(X.min(),X.max())
plt.ylim(Y.min(),Y.max())
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heightmap")
plt.legend()
plt.tight_layout()
plt.show()
