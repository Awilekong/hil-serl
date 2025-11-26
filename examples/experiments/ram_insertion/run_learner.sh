export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=/home/dexfranka/ws_zpw/hil-serl/examples/demo_data/ram_insertion_49_demos_6d.pkl \
    --learner \