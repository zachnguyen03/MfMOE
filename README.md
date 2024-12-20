# [MfMOE: Mask-free Multi-Object Editing](https://vcl.seoultech.ac.kr)

Anh Q. Nguyen (in development stage)

## :electron: Overall Architecture :electron:

pending

## :tada: Introduction :tada:


## :hammer_and_wrench: Installation :hammer_and_wrench:

Requirements

The following installation suppose `python=3.9` `pytorch=2.0.1` and `cuda>=11.7`

- Clone the repository

  ```
  git clone https://github.com/zachnguyen03/MfMOE.git
  cd MfMOE
  ```

- Environment Installation
  ```
  conda create -n mfmoe python=3.9
  conda activate mfmoe
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```

## :nut_and_bolt: Inference :nut_and_bolt:
- Configure the data path and run the following command for inference
  ```
    python main.py \
    --sd_version 2.0 \
    --image_path <insert path to source image> \
    --source_prompt <insert source prompt if given> \
    --token_position <insert token position of editing objects> \
    --mask_paths <insert paths to masks> \
    --fg_negative <insert negative prompts> \
    --bg_negative <insert negative prompt for background> \
    --H 512 \
    --W 512 \
    --bootstrapping <insert number of bootstrapping steps> \
    --result_dir <insert directory to save results> \
    --rec_path <insert save path of reconstructed image> \
    --edit_path <insert save path of edited image> \
    --merged_path <insert save path of merged reconstructed and edited images> \
    --fg_prompts <insert prompts for each mask> \
    --num_fgmasks <insert number of masks to edit> \
    --seed <insert seed> \
  ```

## :fireworks: Demo app :fireworks:
- To test the demo app, run the following command
  ```
    python app.py
  ```

- Steps to run the editing process: upload image :arrow_right: Generate Prompt :arrow_right: Specify token position, editing and negative prompts :arrow_right: Run

- User interface preview ![demo](./assets/demo.png)
