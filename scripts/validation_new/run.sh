source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh

python -m scripts.Grid_Agglom_optimiser \
    --input /vols/cms/mm1221/Independent/Data/Electron_2/test/ \
    --max_events 1000 \
    --cluster_events 1000 \
    --model_path /vols/cms/mm1221/Independent/Transformer/runs/Electron_hd64_nl4_k36/best_model.pt \
    --n_coarse 10 \
    --n_fine 10 \
    --min_threshold 0.1 \
    --max_threshold 1.0 \
    --fine_window 0.2 \
    --output threshold_sweep_test.csv
