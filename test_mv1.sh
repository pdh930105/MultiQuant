python mv1_test.py \
--data /dataset/imagenet/ \
--visible_gpus '0,1,2,3' \
--multiprocessing_distributed True \
--dist_url 'tcp://127.0.0.1:23456' \
--workers 20  \
--arch 'mv1_quant' \
--batch_size 256  \
--epochs 90 \
--lr_m 0.1 \
--lr_q 0.0001 \
--log_dir "./results/mv1" \
--bit_list 2468 \
--resume './mv1-2468/checkpoint.pth.tar'