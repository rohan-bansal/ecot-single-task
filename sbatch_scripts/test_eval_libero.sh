# # Original OpenVLA eval, working
# python experiments/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True

source /coc/flash7/zhenyang/miniconda3/etc/profile.d/conda.sh 
conda activate openvla

ckpt="/coc/flash7/rbansal66/embodied-CoT/runs/prism-dinosiglip-224px+mx-libero-90+n1+b16+x7/checkpoints/step-000200-epoch-00-loss=0.0597.pt"

ckpt="/coc/flash7/zhenyang/data/openvla_ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt"
# ckpt="/coc/flash7/zhenyang/data/openvla_ckpts/openvla-7b-prismatic"
ckpt="openvla/openvla-7b-finetuned-libero-spatial"

# ECoT eval
python experiments/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $ckpt \
  --task_suite_name libero_90 \
  --center_crop True
