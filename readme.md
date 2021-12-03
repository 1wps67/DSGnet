# Requirement
conda create -n deepc --file deepc.yaml

conda activate deepc




# Train
```
cd **\DSGnet-MainCode\DSGnetTotal\DEGnetNetwork\tools 
DSGnet train：python train.py --cfg ../experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```
# Test
```
cd **\DSGnet-MainCode\DSGnetTotal\DEGnetNetwork\tools 
DSGnet test：python val_net_res.py --cfg ../experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml
```
