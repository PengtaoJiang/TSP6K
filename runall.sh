bash tools/dist_train.sh \
    configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py 8 --auto-resume  

# bash tools/dist_test.sh \
#     configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
#     ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/latest.pth \
#     1 --out ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/results.pkl --aug-test --eval mIoU  

# bash tools/dist_test.sh \
#     configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
#     ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/latest.pth \
#     1 --out ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/results.pkl --aug-test --eval cityscapes 