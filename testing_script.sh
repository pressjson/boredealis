#!/usr/bin/env bash

# Uncomment the part that you want to use

trap "Exiting . . ." SIGINT

# test a model and a video over multiple iterations

# test_model="$HOME/Documents/Boredealis/vgg_128_models/checkpoint_best.pth"
# test_video="$HOME/Documents/Boredealis_Media/test_for_dr_fasel/04122019_044905.avi"
# output_dir="$HOME/Documents/Boredealis_Media/test_for_dr_fasel/iterations/"

# mkdir -p $output_dir

# for iter in {2..14..2}; do
#     file_base="${test_model##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}_iter_${iter}.mp4"
#     echo "./run.sh -i=$test_video -o=$save_path -c=$test_model -I=$iter"
#     ./run.sh -i=$test_video -o=$save_path -c=$test_model -I=$iter
# done

# test multiple videos across multiple models

# test_videos_dir=($HOME/Documents/Boredealis_Media/test_videos/*)
# test_models_dir=($HOME/Documents/Boredealis/testing_models/*)
# output_dir="$HOME/Documents/Boredealis_Media/split_vgg_various_stages/"

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

# test_models_dir=($HOME/Documents/Boredealis/testing_models/*)
# output_dir="$HOME/Documents/Boredealis_Media/randiv_various_stages/"
# test_images_dir=($HOME/Documents/Boredealis_Media/test_images/*)
# # test_image="$HOME/Documents/Boredealis/readme_images/output_0258.png"
# mkdir -p $output_dir

# for test_image in ${test_images_dir[@]}; do
#     test_image_base="${test_image##*/}"
#     test_image_base_noext="${test_image_base%.*}"
#     output_subdir="${output_dir}${test_image_base_noext}"
#     echo $output_subdir
#     mkdir -p "$output_subdir"

#     for file in ${test_models_dir[@]}; do
#         file_base="${file##*/}"
#         file_base_noext="${file_base%.*}"
#         save_path="${output_dir}${file_base_noext}/${test_image_base}"
#         mkdir -p "${output_dir}${file_base_noext}"
#         command="import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
#         echo $command
#         python -c "$command"
#         echo
#     done
# done

# test multiple images across a single model

# model_path="$HOME/Documents/Boredealis/testing_models/randiv_64_checkpoint_best.pth"
# output_dir="$HOME/Documents/Boredealis_Media/noisy_randiv_64_models/"
# test_images_dir=($HOME/Documents/Boredealis_Media/test_images/*)
# mkdir -p $output_dir

# for test_image in "${test_images_dir[@]}"; do
#     test_image_base="${test_image##*/}"
#     test_image_base_noext="${test_image_base%.*}"
#     model_base="${model_path##*/}"
#     model_base_noext="${model_base%.*}"
#     save_path="${output_dir}${test_image_base_noext}_${model_base_noext}.png"
#     command="import test; test.save_test(image_path='$test_image', model_load_path='$model_path', save_path='$save_path')"
#     echo $command
#     # python -c "$command"
#     echo
# done

# # test multiple videos across a single model

# # model_path="$HOME/Documents/Boredealis/testing_models/randiv_128_epoch_23.pth"
# # output_dir="$HOME/Documents/Boredealis_Media/randiv_128_models/"
# test_videos_dir=($HOME/Documents/Boredealis_Media/test_videos/*)
# mkdir -p $output_dir

# for file in "${test_videos_dir[@]}"; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     model_base="${model_path##*/}"
#     model_base_noext="${model_base%.*}"
#     save_path="${output_dir}${file_base_noext}_${model_base_noext}.mp4"
#     command="-i=$file -o=$save_path -c=$model_path"
#     echo "./run.sh $command"
#     # ./run.sh $command
# done

# test a single image across multiple models

# test_models_dir=($HOME/Documents/Boredealis/vgg_128_models/*)
# output_dir="$HOME/Documents/Boredealis_Media/randthree_128_various_stages/"
# test_image="$HOME/Documents/Boredealis_Media/test/clouds_output/output_3591.png"
# # test_image="$HOME/Documents/Boredealis/readme_images/output_0258.png"

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

# test_models_dir=($HOME/Documents/Boredealis/split_videos_128_filter_models_take_two/*)
# output_dir="$HOME/Documents/Boredealis_Media/split_videos_128_filter_models_take_two/"
# test_video="$HOME/Documents/Boredealis_Media/test_videos/06012016_071857.avi"

# mkdir -p $output_dir

# for file in ${test_models_dir[@]}; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}.mp4"
#     echo "./run.sh -i=$test_video -o=$save_path -c=$file"
#     ./run.sh -i=$test_video -o=$save_path -c=$file
# done

# do the gauntlet with multiple iterations of multiple models

for BLEND_STRENGTH in {0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0}; do
    # test multiple images across a single model

    model_path="$HOME/Documents/Boredealis/testing_models/randiv_96_checkpoint_best.pth"
    output_dir="$HOME/Documents/Boredealis_Media/noisy_randiv_96_models/blend_strength_${BLEND_STRENGTH}/"
    test_images_dir=($HOME/Documents/Boredealis_Media/test_images/*)
    mkdir -p $output_dir

    for test_image in "${test_images_dir[@]}"; do
        test_image_base="${test_image##*/}"
        test_image_base_noext="${test_image_base%.*}"
        model_base="${model_path##*/}"
        model_base_noext="${model_base%.*}"
        save_path="${output_dir}${test_image_base_noext}_${model_base_noext}.png"
        command="import test; test.save_test(image_path='$test_image', model_load_path='$model_path', save_path='$save_path', blend_strength=${BLEND_STRENGTH})"
        echo $command
        python -c "$command"
        echo
    done

    # test multiple videos across a single model

    test_videos_dir=($HOME/Documents/Boredealis_Media/less_test_videos/*)
    mkdir -p $output_dir

    for file in "${test_videos_dir[@]}"; do
        file_base="${file##*/}"
        file_base_noext="${file_base%.*}"
        model_base="${model_path##*/}"
        model_base_noext="${model_base%.*}"
        save_path="${output_dir}${file_base_noext}_${model_base_noext}.mp4"
        command="-i=$file -o=$save_path -c=$model_path --blend=${BLEND_STRENGTH}"
        echo "./run.sh $command"
        ./run.sh $command
    done

done
