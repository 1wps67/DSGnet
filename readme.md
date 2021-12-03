conda activate deepc
cd **\DSGnet-MainCode\DSGnetTotal\DEGnetNetwork\tools 
DSGnet train：python train.py --cfg ../experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
DSGnet test：python val_net_res.py --cfg ../experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml