[![Site](https://img.shields.io/badge/WikiCheck-API-2ea44f?style=for-the-badge)](https://nli.wmflabs.org/)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=ResearchGate&logoColor=white)](https://www.researchgate.net/publication/356246861_WikiCheck_An_End-to-end_Open_Source_Automatic_Fact-Checking_API_based_on_Wikipedia)
[![Wikipedia](https://img.shields.io/badge/Wikipedia-%23000000.svg?style=for-the-badge&logo=wikipedia&logoColor=white)](https://meta.wikimedia.org/wiki/Research:Implementing_a_prototype_for_Automatic_Fact_Checking_in_Wikipedia)
# WikiCheck API

Repository with the implementation of WikiCheck API, end-to-end open source Automatic Fact-Checking based on Wikipedia.

The research was published in **CIKM2021** applied track:
- *Trokhymovych, Mykola, and Diego Saez-Trumper.* 
**WikiCheck: An End-to-End Open Source Automatic Fact-Checking API Based on Wikipedia.**
Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 
Association for Computing Machinery, 2021, pp. 4155–4164, CIKM ’21.
[![DOI:10.1145/3459637.3481961](https://zenodo.org/badge/DOI/10.1145/3459637.3481961.svg)](https://dl.acm.org/doi/10.1145/3459637.3481961)

- The preprint **WikiCheck: An End-to-End Open Source Automatic Fact-Checking API Based on Wikipedia**: [![DOI:10.48550/arXiv.2109.00835](https://zenodo.org/badge/DOI/10.48550/arXiv.2109.00835.svg)](
https://doi.org/10.48550/arXiv.2109.00835)

We encourage you to test the WikiCheck API by yourself: [![Website](https://img.shields.io/website?style=flat-square&down_color=red&down_message=offline&label=WikiCheck&logo=WikiCheck&style=plastic&up_color=green&up_message=online&url=https://nli.wmcloud.org/docs)](https://nli.wmflabs.org/)

## Installation and Usage: 
The project consists of **modules** directory with the implementation of modules 
used for inference along with the script for NLI models training. 

The **configs** directory includes configuration files for training and inference. 

The **notebooks** directory includes .ipynb notebooks with experiments done during the research.

If you want to get access to our fine-tuned models, you can load them from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11660051.svg)](https://doi.org/10.5281/zenodo.11660051)


Also, you can train your model by running the ```modules/model_trainer.py``` script. 


### API setup and run

- Clone the official WikiCheck repo and cd into it 

```git clone https://github.com/trokhymovych/WikiCheck.git```

```cd WikiCheck```

- Create and activate virtualenv: 

```virtualenv -p python venv```

```source venv/bin/activate```

- Install requirements from  requirements.txt:

```pip install -r requirements.txt```

- Load pretrained models:
    
    - Loading models from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11660051.svg)](https://doi.org/10.5281/zenodo.11660051)

- Run the API:

```python run.py --config configs/inference/sentence_bert_config.json```


## Citation
If you find this work is useful, please cite our paper:

**WikiCheck: An End-to-End Open Source Automatic Fact-Checking API Based on Wikipedia.**
```
@inproceedings{10.1145/3459637.3481961,
author = {Trokhymovych, Mykola and Saez-Trumper, Diego},
title = {WikiCheck: An End-to-End Open Source Automatic Fact-Checking API Based on Wikipedia},
year = {2021},
isbn = {9781450384469},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3459637.3481961},
doi = {10.1145/3459637.3481961},
booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
pages = {4155–4164},
numpages = {10},
keywords = {applied research, nlp, nli, wikipedia, fact-checking},
location = {Virtual Event, Queensland, Australia},
series = {CIKM '21}
}
```
