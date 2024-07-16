
import torch
from quality_of_life.my_base_utils import dict_to_json

hyperparameters = {
    "DEVICE" : "cpu",
    "seed" : 2024,
    "lr" : 0.0005,
    "batch_size" : 64,
    "n_epochs" : 200,
    "make_gif" : True,
    "how_often" : 10,                   # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
    "initial_frame_repetitions" : 24,   # ~~~ for how many frames should the state of initialization be rendered
    "final_frame_repetitions" : 48,     # ~~~ for how many frames should the state after training be rendered
    "data" : "univar_missing_middle",
    "model" : "univar_NN"
}


dict_to_json( hyperparameters, "new_trial.json", override=True )