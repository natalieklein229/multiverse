
from quality_of_life.my_base_utils import find_root_dir_of_repo
from quality_of_life.my_plotly_utils import vector_viz
from bnns.utils import load_coast_coords
from matplotlib import pyplot as plt
from plotly import graph_objects as go
import os
os.chdir( os.path.join(find_root_dir_of_repo(), "bnns", "data", "ne_10m_coastline") )

#
# ~~~ Plot coastline
c = load_coast_coords("ne_10m_coastline.shp")
x, y = c[:,0], c[:,1]
plt.scatter(x,y)
plt.show()

#
# ~~~ Plot a heatmap
os.chdir("..")
from slosh_70_15_15 import out_np, coords_np
vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0] )
vector_viz( x=coords_np[:,0], y=coords_np[:,1], z=out_np[0], graph_object=go.Heatmap )

