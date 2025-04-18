source activate multiverse

seed=(11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

for s in ${seed[@]}; do
    python chemcam_fit_siamuq.py --seed $s
done