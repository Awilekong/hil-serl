export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=charge_first_run \
    --demo_path=/home/dexfranka/ws_zpw/hil-serl/examples/demo_data/ram_insertion_40_demos_2025-11-27_12-06-07.pkl \
    --demo_path=/home/dexfranka/ws_zpw/hil-serl/examples/demo_data/ram_insertion_20_demos_2025-11-27_11-54-51.pkl \
    --demo_path=/home/dexfranka/ws_zpw/hil-serl/examples/demo_data/ram_insertion_13_demos_2025-11-27_13-06-50.pkl \
    --learner \