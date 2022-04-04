# kidney_histo
Detecting kidney cancer from histological slides

command for final model:

sbatch -x mirtti-gpu-2 ~/kidney_histo/scripts/submit1x4.sh main.py --project_name "Cancer_histo" --run_name "HARD_e300_LR25-5_sd-3A500_CB_GB5_drop-1_dpath-1_SMOOTH_pre3" --epochs 300 --lr 0.00025 --sample True --train_spot_dir "/data/atte/data/resnet_pred_edges_high_conf.csv" --train_wsi_spot_dir "/data/atte/data/resnet_99percent_train_wsi_noclinic.csv" --sample_val True --num_cancer_wsi 0 --num_benign_wsi 0  --weight_decay 0 --prob_gaussian 0.05 --spectral True --sd_lambda 0.001 --drop 0.1 --drop_path 0.1 --class_balance True --sd_anneal 500 --label_smooth True --pre_train True
