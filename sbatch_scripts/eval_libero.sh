# # Original OpenVLA eval, working
# python experiments/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True

cpkt="/coc/flash7/rbansal66/embodied-CoT/runs/prism-dinosiglip-224px+mx-libero-90+n1+b16+x7/checkpoints/step-000200-epoch-00-loss=0.0597.pt"

# ECoT eval
python experiments/libero/run_libero_eval.py \
  --model_family ecot \
  --pretrained_checkpoint $cpkt \
  --task_suite_name libero_90 \
  --center_crop True

# online ckpt Embodied-CoT/ecot-openvla-7b-oxe