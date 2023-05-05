hfai python tools/get_flops.py \
    configs/tsp6k/segnext_large_1024x1024_160k_tsp6k_msaspp_ra_mod.py \
    --shape 512 \
    -- -n 1 -i ubuntu2004-cu113-ext --name segnext_large_1024x1024_160k_tsp6k_msaspp_ra_mod_flops
