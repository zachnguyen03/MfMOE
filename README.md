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
    --mask_paths <insert paths to masks> \
    --fg_negative "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image" \
    --H 512 \
    --W 512 \
    --bootstrapping <insert number of bootstrapping steps> \
    --rec_path <insert save path of reconstructed image> \
    --edit_path <insert save path of edited image> \
    --fg_prompts <insert prompts for each mask> \
    --num_fgmasks <insert number of masks to edit> \
    --seed <insert seed> \
    --save_path <insert save path of merged reconstructed and edited images>
  ```