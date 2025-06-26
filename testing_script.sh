#!/usr/bin/env bash

# Uncomment the part that you want to use

trap "Exiting . . ." SIGINT

# test multiple videos across multiple models

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
# test_models_dir=(/home/pressjson/Documents/Boredealis/vgg_96_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_96_various_stages/"

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

# do the same thing, but with 128 filter models
# @TODO: remove this

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
# test_models_dir=(/home/pressjson/Documents/Boredealis/vgg_128_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_128_various_stages/"

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

# test_models_dir=(/home/pressjson/Documents/Boredealis/vgg_96_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_96_various_stages/"
# test_images_dir=(/home/pressjson/Documents/Boredealis_Media/test_images/*)
# # test_image="/home/pressjson/Documents/Boredealis/readme_images/output_0258.png"

# source venv/bin/activate
# for test_image in ${test_images_dir[@]}; do
#     test_image_base="${test_image##*/}"
#     test_image_base_noext="${test_image_base%.*}"
#     output_subdir="${output_dir}${test_image_base_noext}"
#     echo $output_subdir
#     mkdir -p "$output_subdir"

#     for file in ${test_models_dir[@]}; do
#         file_base="${file##*/}"
#         file_base_noext="${file_base%.*}"
#         save_path="${output_subdir}/${file_base_noext}.png"
#         command="import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
#         echo $command
#         python -c "$command"
#         echo
#     done
# done
# deactivate

# do the same thing with 128 filter models
# why refactor for loops when i can copy-paste for a one time solution?
# @TODO: remove this

test_models_dir=(/home/pressjson/Documents/Boredealis/vgg_128_models/*)
output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_128_various_stages/"
test_images_dir=(/home/pressjson/Documents/Boredealis_Media/test_images/*)
# test_image="/home/pressjson/Documents/Boredealis/readme_images/output_0258.png"

source venv/bin/activate
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
deactivate


# test multiple videos across a single model

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/randthree_128_various_stages/"
# model_path="/home/pressjson/Documents/Boredealis/models/vgg_96_checkpoint_7.pth"

# for file in ${test_videos_dir[@]}; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     model_base="${model_path##*/}"
#     model_base_noext="${model_base%.*}"
#     save_path="${output_dir}${file_base_noext}_${model_base_noext}.mov"
#     echo "./run.sh -i=$file -o=$save_path -c=$model_path"
#     ./run.sh -i=$file -o=$save_path -c=$model_path
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

# test_models_dir=(/home/pressjson/Documents/Boredealis/randthree_96_filter_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/randthree_96_various_stages/"
# test_video="/home/pressjson/Documents/Boredealis_Media/test/clouds_sample.avi"

# for file in ${test_models_dir[@]}; do
#     source venv/bin/activate
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}.mov"
#     # python -c "import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
#     ./run.sh -i=$test_video -o=$save_path -c=$file
#     deactivate
# done
