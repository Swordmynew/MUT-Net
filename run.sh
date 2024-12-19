export CUDA_VISIBLE_DEVICES=2,3
source activate DGNet
python MyTrain.py --gpu_id 0,1  --save_path /data1/kangkejun/SAE-bi/snapshot/Exp-DGNet/ --batchsize 4 --model PVTv2-B4 --train_root /data1/kangkejun/TrainDataset/ --val_root /data1/kangkejun/TestDataset/CHAMELEON/ --trainsize 352
# --load /data1/kangkejun/SAE-bi/snapshot/Exp-DGNet/Net_epoch_200_EF-B4.pth --gpu_id 0,1
