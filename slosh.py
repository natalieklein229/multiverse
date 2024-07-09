
import os
import pyreadr                  # ~~~ from https://stackoverflow.com/a/61699417

#
# ~~~ Set path to the .rda file
ans = input("    Is the path 'slosh_dat_nj.rda' correct?\n    Enter 'y' for yes, any other key for no.\n")
if ans.lower()=="y":
    PATH = 'slosh_dat_nj.rda'
else:
    PATH = input(".   Please type the path without quotes and press enter:\n") # ~~~ e.g., /Users/winckelman/Downloads/slosh_dat_nj.rda

#
# ~~~ Extract the data as numpy arrays
DATA = pyreadr.read_r(path)     # ~~~ from https://stackoverflow.com/a/61699417
coords_np  =  DATA["coords"].to_numpy()
inputs_np  =  DATA["inputs"].to_numpy()
out_np     =     DATA["out"].to_numpy()
