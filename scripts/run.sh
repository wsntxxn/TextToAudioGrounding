#!/bin/bash
#SBATCH --job-name audio-grounding
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --exclude=gqxx-01-016
#SBATCH --output=logs/audio-grounding-%j.log
#SBATCH --error=logs/audio-grounding-%j.err

if [ $# -lt 5 ]; then
    echo -e "Usage: $0 <run-script> <run-config> <audio-feature> <query-feature> <eval-label-file> [<seed>]"
    exit 1
fi


run_script=$1
run_config=$2
audio_feature=$3
query_feature=$4
eval_label_file=$5

seed=1
if [ $# -eq 6 ]; then
    seed=$6
fi

# stage 1, train the audio caption model
if [ ! $experiment_path ]; then
    experiment_path=`python ${run_script} \
                            train \
                            ${run_config} \
                            --seed $seed`
fi

# stage 2, evaluate by several metrics

if [ ! $experiment_path ]; then
    echo "invalid experiment path, maybe the training ended abnormally"
    exit 1
fi

python ${run_script} \
       evaluate \
       ${experiment_path} \
       ${audio_feature} \
       ${query_feature} \
       ${eval_label_file}
