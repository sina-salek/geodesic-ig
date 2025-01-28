#!/bin/bash

processes=${processes:-1}
device=${device:-cpu}

while [ $# -gt 0 ]
do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi
  shift
done

trap ctrl_c INT

function ctrl_c() {
    echo " Stopping running processes..."
    kill -- -$$
}

for seed in $(seq 12 12 60)
do
  python main.py \
    --device "$device" \
    --seed "$seed" \
    --n_steps 50 \
    --explainers geodesic_integrated_gradients input_x_gradient kernel_shap svi_integrated_gradients guided_integrated_gradients integrated_gradients gradient_shap &

  # Support lower versions
  if ((BASH_VERSINFO[0] >= 4)) && ((BASH_VERSINFO[1] >= 3))
  then
    if [[ $(jobs -r -p | wc -l) -ge $processes ]]
    then
      wait -n
    fi
  else
    while [[ $(jobs -r -p | wc -l) -ge $processes ]]
    do
      sleep 1
    done
  fi
done

wait