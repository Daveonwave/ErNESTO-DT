python ernesto.py \
    --battery_model first_order_thevenin \
    --thermal_model r2c_thermal \
    adaptive --config_files ./data/config/sim_adaptive.yaml \
    --alpha 0.18 \
    --batch_size 10000 \
    --alg L-BFGS-B \
    --n_restarts 2
