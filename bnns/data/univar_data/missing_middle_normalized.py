
from bnns.data.missing_middle import x_train, y_train, x_test, y_test

#
# ~~~ Scale down the data
scale = 12
y_train /= scale
y_test  /= scale
ground_truth = lambda x: f(x)/scale