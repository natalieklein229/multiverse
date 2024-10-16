
from bnns.data.missing_middle import x_train, y_train, x_test, y_test, x_val, y_val
from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset

#
# ~~~ Scale down the data
scale = 12
y_train /= scale
y_test /= scale
v_val /= scale
ground_truth = lambda x: f(x)/scale

#
# ~~~ Finally, package as objects of class torch.utils.data.Dataset
D_train = convert_Tensors_to_Dataset(x_train,y_train)
D_test = convert_Tensors_to_Dataset(x_test,y_test)
D_val = convert_Tensors_to_Dataset(x_val,y_val)
