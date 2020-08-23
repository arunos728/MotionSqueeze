
##################################################################### 
# Train on local machine
if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
    cd $PBS_O_WORKDIR
fi

##################################################################### 
# Parameters!
mainFolder="net_runs"
subFolder="TSM_MS_R18_kinetics_run1"
snap_pref="MotionSqueeze"

train_path="data/Kinetics400_rgb_train.txt"
val_path="data/Kinetics400_rgb_val.txt"


#############################################
#--- training hyperparams ---
dataset_name="kinetics"
netType="MS"
batch_size=8
learning_rate=0.02 # lr follows root of bs
num_segments=8
mode=0
dropout=0.5
iter_size=1
num_workers=16

##################################################################### 
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training
mkdir -p ${mainFolder}/${subFolder}/validation

echo "Current network folder: "
echo ${mainFolder}/${subFolder}


##################################################################### 
# Find the latest checkpoint of network 
checkpointIter="$(ls ${mainFolder}/${subFolder}/*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
##################################################################### 


echo "${checkpointIter}" 
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"    
    echo "Continuing from checkpoint ${lastCheckpoint}"

#####################################################################

python3 -u main_kinetics.py ${dataset_name} RGB ${train_path} ${val_path}  --arch ${netType} --num_segments ${num_segments} --mode ${mode} --gd 20 --wd 1e-4 --lr ${learning_rate} --lr_steps 30 60 --epochs 75 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/${snap_pref} --consensus_type avg --eval-freq 1 --rgb_prefix img_ --pretrained_parts finetune --no_partialbn -e -p 20 --val_output_folder ${mainFolder}/${subFolder}/validation --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt    

fi