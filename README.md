# Analysis-on-GAN

![Authour](https://img.shields.io/badge/Author-Wapiti08-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.8-brightgreen.svg) 
![GAN](https://img.shields.io/badge/GAN-GAN-redgreen.svg)
![Ransomware](https://img.shields.io/badge/Ransomware-Locky%20Crypto-redgreen.svg)
![License](https://img.shields.io/badge/license-MIT3.0-green.svg)
[![DOI](https://zenodo.org/badge/264062663.svg)](https://doi.org/10.5281/zenodo.13881242)

---

This is my project about Ransome Detection using GAN, the experiment was implemented by cuckoo. 

### Features:

- read report files from specific location (set in cuckoo) and extract features from them
- features engineering on choosing features with highest weights
- generate matrix based on the features(occurences of features)
- experiments with black attack for the GAN model
- experiments based on different parameters, number of hidden layers, number of nodes
- experiments based on ensemble classifiers
- experiments based on ensemble neural networks
- experiments on both ransomware dataset and generous malware

### Pipelines：

    - collect dataset:
        - please put the generated reports seperately under dataset folder
        - run the col_reports inside the processing (specify your own folder)
    
    - generate raw features:
        - run the the feature_gen to generated seperate features

    - get the total unique feature list:
        - run the feature_dict to get features list

    - get the database based on feature occurrences
        - run the feature_fin to get dataset.csv

    - classifier analysis (in classifiers):
       - analyse and filter the generated datasets and classifier algorithms

    - model experiment:
        - test the models with filter features

For the ransomware and cuckoo configuration, please check the link [cuckoo-download-instructions](https://github.com/Wapiti08/cuckoo-download-instructions)
