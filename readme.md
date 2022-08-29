# Deep learning-based rig-agnostic encoding
This is my master thesis project about studying the possibility of transfer learning for data-driven motion generation frameworks.
All necessary code for producing the results described in the thesis are provided here as it is. 

**Objective-driven motion generation model architecture**
![ Objective-driven motion generation model architecture ]( imgs/OMG.png )

**Rig-agnostic encoding approaches**
![ Rig-agnostic encoding approaches illustration ]( imgs/RAE.png )

## Implementation and tuning
The created motion data are exported from Unity as JSON files, which are parsed and
extracted to Numpy arrays and stored as bzip­2 compressed binary files.

The models are implemented in Python using Pytorch and Pytorch­Lightning. 

The implementation of the models are based on [MANN][1], [NSM][2], [LMP­MoE][3], [MVAE][4] and [TRLSTM][5]
The implemented models are tested with a small subset of the dataset, to verify
the implementation. Ensuring that the reconstruction errors are optimised during
the training, and the models are capable of generating correct animations. The
hyperparameters such as the number of layers, the layer sizes and the learning rates are
tuned using Ray Tune 4 with ASHA scheduler and a grid search algorithm.

## Demo
### Transferred FE-OMG 
The white character is playing the target animation. The blue character is the generated animation from the vanilla OMG model with limited training and data. The red character is from the warm-started OMG model with parameters from a pre-trained model that was previously trained on another rig. In this case, only the autoencoders are being optimized, meaning only the input and the output models are being trained. The green character is the same as the red one but also the core generation model is being trained.
![ Transferred FE-OMG ]( imgs/transferred_FE.gif )

### Transferred FS-OMG 
In this case, the pose inputs to the OMG models only contain data for 6 key-joints (hands, feet, head and pelvis). OMG model is responsible to not only predict the next pose but also upscale it to the full resolution pose.
![ Transferred FS-OMG ]( imgs/transferred_FS.gif )

## Contents
1. [ Jupyter Notebooks ]( src/notebooks ) - contains the notebooks for computing and plotting the results (assuming the models are trained and available).
2. [ MLP with adversarial net ]( src/autoencoder/MLP_Adversarial.py ) - is the default Autoencoder (3-layer MLP) + an adversarial Conv-LSGAN model for providing the adversarial error of the generated poses.
3. [ Clustering models ]( src/clustering_modes ) - contains four variants of AE with an extra layer between the encoder and decoder for performing the clustering on the embeddings
4. [ Experiments ]( src/experiments ) - contains code for training, validating and testing the various models
5. [ func ]( src/func ) - contains miscellanenous functions for extracting, preparing data
6. [ motion_generation_models ]( src/motion_generation_models ) - contains the various OMG models and MoGenNet

## References
[1]: Zhang, He, Starke, Sebastian, Komura, Taku, and Saito, Jun. “Mode­adaptive
neural networks for quadruped motion control”. In: ACM Transactions on
Graphics (TOG) 37.4 (2018), pp. 1–11. ISSN: 0730­0301. DOI: 10.1145/3197517.
3201366.

[2]: Starke, Sebastian, Zhang, He, Komura, Taku, and Saito, Jun. “Neural state machine
for character­scene interactions”. In: ACM Transactions on Graphics (TOG) 38.6
(2019), pp. 1–14. ISSN: 0730­0301. DOI: 10.1145/3355089.3356505.

[3]: Starke, Sebastian, Zhao, Yiwei, Komura, Taku, and Zaman, Kazi. “Local motion
phases for learning multi­contact character movements”. In: ACM Transactions
on Graphics (TOG) 39.4 (2020), 54:1–54:13. ISSN: 0730­0301. DOI: 10 . 1145 /
3386569.3392450.

[4]: Ling, Hung Yu, Zinno, Fabio, Cheng, George, and Panne, Michiel Van De.
“Character controllers using motion VAEs”. In: ACM Transactions on Graphics
(TOG) 39.4 (2020), 40:1–40:12. ISSN: 0730­0301. DOI: 10 . 1145 / 3386569 .
3392422.

[5]: Harvey, Félix G., Yurick, Mike, Nowrouzezahrai, Derek, and Pal, Christopher.
“Robust motion in­betweening”. In: ACM Transactions on Graphics (TOG) 39.4
(2020), 60:1–60:12. ISSN: 0730­0301. DOI: 10.1145/3386569.3392480.

