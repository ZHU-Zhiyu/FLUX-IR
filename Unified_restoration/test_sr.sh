CUDA_VISIBLE_DEVICES="0" python main.py \
 --prompt 'high-resolution, ultra-sharp, detailed' \
 --images_path input/super-resolution \
 --local_path checkpoints/FluxIR.bin \
 --use_controlnet \
 --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 5 \
 --num_steps 21 --true_gs 4 --guidance 4 \
 --control_weight 0.8 \
 --save_path results/super-resolution