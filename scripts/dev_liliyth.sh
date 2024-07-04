#export PYTHONPATH=.
#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="stable-diffusion/stable-diffusion-v1-5"
export INSTANCE_DIR="data/training/liliyth/liliyth_clean"
export OUTPUT_DIR="output/mist/sd1-5/liliyth_clean"
export CLASS_DIR="data/class/liliyth_clean"
 
accelerate launch attacks/mist.py \
 --cuda --low_vram_mode --resize \
 --pretrained_model_name_or_path=$MODEL_NAME  \
 --instance_data_dir=$INSTANCE_DIR \
 --output_dir=$OUTPUT_DIR \
 --class_data_dir=$CLASS_DIR\
 --instance_prompt "an animated girl" \
 --class_prompt "a oil painting of a girl, high quality, master piece" \
 --mixed_precision fp16 \
 --max_train_steps 5 \
 --checkpointing_iterations 1 \
 --prior_loss_weight 0.1 \
 --pgd_alpha 0.005 \
 --pgd_eps 0.04 \
 --max_adv_train_steps 30 \
 --max_f_train_steps 10