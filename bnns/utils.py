
class TrainingConfig:
    DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

#
# ~~~ Regarding the training method
functional = True
Optimizer = torch.optim.Adam
batch_size = 64
lr = 0.0005
n_epochs = 2000
n_posterior_samples = 100   # ~~~ posterior distributions are approximated as empirical dist.'s of this many samples
n_MC_samples = 20           # ~~~ expectations are estimated as an average of this many Monte-Carlo samples
project = True              # ~~~ if True, use projected gradient descent; else use the weird thing from the paper
projection_tol = 1e-6       # ~~~ for numerical reasons, project onto [projection_tol,Inf), rather than onto [0,Inft)
conditional_std = 0.9       # ~~~ what Natalie was explaining to me on Tuesday

#
# ~~~ Regarding the SSGE
M = 50          # ~~~ M in SSGE
J = 10          # ~~~ J in SSGE
eta = 0.0001    # ~~~ stability term added to the SSGE's RBF kernel

#
# ~~~ Regarding Stein GD
n_Stein_particles = n_posterior_samples
n_Stein_iterations = n_epochs

#
# ~~~ Regarding visualizaing of training
make_gif = True         # ~~~ if true, aa .gif is made (even if false, the function is still plotted)
how_often = 10          # ~~~ how many snap shots in total should be taken throughout training (each snap-shot being a frame in the .gif)
initial_frame_repetitions = 24  # ~~~ for how many frames should the state of initialization be rendered
final_frame_repetitions = 48    # ~~~ for how many frames should the state after training be rendered
plot_indivitual_NNs = False     # ~~~ if True, do *not* plot confidence intervals and, instead, plot only a few sampled nets
extra_std = False               # ~~~ if True, add the conditional std. when plotting the +/- 2 standard deviation bars

