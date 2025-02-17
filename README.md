# Code-of-KGNet
The code of IEEE IoTJ 2022 "Deep Learning-based Physical-Layer Secret Key Generation for FDD Systems"

# Dataset
Download data from [DeepMIMO](https://www.deepmimo.net/scenarios/i1-scenario/). We only consider one antenna in this scenario.

# Environment
Tensorflow 2.1 + python3.6

# Key Generation
* Step1: **python NN_predict.py** to perform channel mapping.
* Step2: **python Quantify.py** to perform quantification.

# Citation
Please cite our work as follows:

```
@ARTICLE{9526766,
  author={Zhang, Xinwei and Li, Guyue and Zhang, Junqing and Hu, Aiqun and Hou, Zongyue and Xiao, Bin},
  journal={IEEE Internet of Things Journal}, 
  title={Deep-Learning-Based Physical-Layer Secret Key Generation for FDD Systems}, 
  year={2022},
  volume={9},
  number={8},
  pages={6081-6094},
  doi={10.1109/JIOT.2021.3109272}}
```
and
```
@ARTICLE{10440494,
  author={Zhang, Xinwei and Li, Guyue and Zhang, Junqing and Peng, Linning and Hu, Aiqun and Wang, Xianbin},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={Enabling Deep Learning-Based Physical-Layer Secret Key Generation for FDD-OFDM Systems in Multi-Environments}, 
  year={2024},
  volume={73},
  number={7},
  pages={10135-10149},
  keywords={Deep learning;Downlink;Feature extraction;Training;OFDM;Metalearning;Physical layer security;Transfer learning;Metalearning;Physical-layer security;secret key generation;frequency division duplexing;deep transfer learning;meta-learning},
  doi={10.1109/TVT.2024.3367362}}

```
