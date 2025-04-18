source activate multiverse

seed=($(seq 1 10))

for s in ${seed[@]}; do
    python chemcam_fit_siamuq.py --seed $s
done