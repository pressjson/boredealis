#!/usr/bin/env bash

# Uncomment the part that you want to use

trap "Exiting . . ." SIGINT

# test a model and a video over multiple iterations

# test_model="/home/pressjson/Documents/Boredealis/vgg_128_models/checkpoint_best.pth"
# test_video="/home/pressjson/Documents/Boredealis_Media/test_for_dr_fasel/04122019_044905.avi"
# output_dir="/home/pressjson/Documents/Boredealis_Media/test_for_dr_fasel/iterations/"

# mkdir -p $output_dir

# for iter in {2..14..2}; do
#     file_base="${test_model##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}_iter_${iter}.mp4"
#     echo "./run.sh -i=$test_video -o=$save_path -c=$test_model -I=$iter"
#     ./run.sh -i=$test_video -o=$save_path -c=$test_model -I=$iter
# done

# test multiple videos across multiple models

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
# test_models_dir=(/home/pressjson/Documents/Boredealis/testing_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/split_vgg_various_stages/"

# mkdir -p $output_dir

# for model in ${test_models_dir[@]}; do
#     model_base="${model##*/}"
#     model_base_noext="${model_base%.*}"
#     for file in ${test_videos_dir[@]}; do
#         file_base="${file##*/}"
#         file_base_noext="${file_base%.*}"
#         mkdir -p ${output_dir}${model_base_noext}
#         save_path="${output_dir}${model_base_noext}/${file_base_noext}_${model_base_noext}.mp4"
#         echo "./run.sh -i=$file -o=$save_path -c=$model"
#         ./run.sh -i=$file -o=$save_path -c=$model
#     done
# done

# test multiple images across multiple models

test_models_dir=(/home/pressjson/Documents/Boredealis/testing_models/*)
output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_various_stages/"
test_images_dir=(/home/pressjson/Documents/Boredealis_Media/test_images/*)
# test_image="/home/pressjson/Documents/Boredealis/readme_images/output_0258.png"
mkdir -p $output_dir

for test_image in ${test_images_dir[@]}; do
    test_image_base="${test_image##*/}"
    test_image_base_noext="${test_image_base%.*}"
    output_subdir="${output_dir}${test_image_base_noext}"
    echo $output_subdir
    mkdir -p "$output_subdir"

    for file in ${test_models_dir[@]}; do
        file_base="${file##*/}"
        file_base_noext="${file_base%.*}"
        save_path="${output_subdir}/${file_base_noext}.png"
        command="import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
        echo $command
        python -c "$command"
        echo
    done
done

# test multiple videos across a single model

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_for_dr_fasel/034811-244905_iter_5/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/test_for_dr_fasel/that_special_time_iter_10/"
# model_path="/home/pressjson/Documents/Boredealis/vgg_128_models/checkpoint_best.pth"
# mkdir -p $output_dir

# for file in "${test_videos_dir[@]}"; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     model_base="${model_path##*/}"
#     model_base_noext="${model_base%.*}"
#     save_path="${output_dir}${file_base_noext}_${model_base_noext}.mp4"
#     command="-i=$file -o=$save_path -c=$model_path -I=5"
#     echo "./run.sh $command"
#     ./run.sh $command
# done


# test a single image across multiple models

# test_models_dir=(/home/pressjson/Documents/Boredealis/vgg_128_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/randthree_128_various_stages/"
# test_image="/home/pressjson/Documents/Boredealis_Media/test/clouds_output/output_3591.png"
# # test_image="/home/pressjson/Documents/Boredealis/readme_images/output_0258.png"

# source venv/bin/activate
# for file in ${test_models_dir[@]}; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}.png"
#     command="import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
#     echo $command
#     python -c "$command"
#     echo
# done
# deactivate

# test a single video across multiple models

# test_models_dir=(/home/pressjson/Documents/Boredealis/split_videos_128_filter_models_take_two/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/split_videos_128_filter_models_take_two/"
# test_video="/home/pressjson/Documents/Boredealis_Media/test_videos/06012016_071857.avi"

# mkdir -p $output_dir

# for file in ${test_models_dir[@]}; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}.mp4"
#     echo "./run.sh -i=$test_video -o=$save_path -c=$file"
#     ./run.sh -i=$test_video -o=$save_path -c=$file
# done
