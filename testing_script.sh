#!/usr/bin/env bash

# Uncomment the part that you want to use

trap "Exiting . . ." SIGINT

# test multiple videos across multiple models

test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
test_models_dir=(/home/pressjson/Documents/Boredealis/models/*)
output_dir="/home/pressjson/Documents/Boredealis_Media/vgg_multiple_128_models/"

for model in ${test_models_dir[@]}; do
    model_base="${model##*/}"
    model_base_noext="${model_base%.*}"
    for file in ${test_videos_dir[@]}; do
        file_base="${file##*/}"
        file_base_noext="${file_base%.*}"
        mkdir -p ${output_dir}${model_base_noext}
        save_path="${output_dir}${model_base_noext}/${file_base_noext}_${model_base_noext}.mp4"
        echo "./run.sh -i=$file -o=$save_path -c=$model"
        ./run.sh -i=$file -o=$save_path -c=$model
    done
done

# test multiple videos across a single model

# test_videos_dir=(/home/pressjson/Documents/Boredealis_Media/test_videos/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/randthree_128_various_stages/"
# model_path="/home/pressjson/Documents/Boredealis/models/vgg_128_epoch_30.pth"

# for file in ${test_videos_dir[@]}; do
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     model_base="${model_path##*/}"
#     model_base_noext="${model_base%.*}"
#     save_path="${output_dir}${file_base_noext}_${model_base_noext}.mp4"
#     echo "./run.sh -i=$file -o=$save_path -c=$model_path"
#     ./run.sh -i=$file -o=$save_path -c=$model_path
# done


# test a single image across multiple models

# test_models_dir=(/home/pressjson/Documents/Boredealis/randthree_128_filter_models/*)
# output_dir="/home/pressjson/Documents/Boredealis_Media/randthree_128_various_stages/"
# # test_image="/home/pressjson/Documents/Boredealis_Media/test/clouds_output/output_3591.png"
# test_image="/home/pressjson/Documents/Boredealis/readme_images/output_0258.png"

# for file in ${test_models_dir[@]}; do
#     source venv/bin/activate
#     file_base="${file##*/}"
#     file_base_noext="${file_base%.*}"
#     save_path="${output_dir}${file_base_noext}.png"
#     echo "python -c import test; test.save_test(image_path=$test_image, model_load_path=$file, save_path=$save_path)"
#     python -c "import test; test.save_test(image_path='$test_image', model_load_path='$file', save_path='$save_path')"
#     deactivate
# done


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
