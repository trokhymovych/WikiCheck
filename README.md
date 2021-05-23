# WikiCheck API

Repository with implementation of WikiCheck API. 
The project is done in cooperation with Wikimedia Foundation and Ukrainian Catholic University. 

The link to API: https://nli.wmcloud.org

#### The structure of the project: 
The project consists of **modules** directory with implementation of modules 
used for inference along with script for NLI models training. 

The **configs** directory includes configuration files for training and inference. 

The **notebooks** directory (not added yet) includes .ipynb notebooks with experiments done during the research.

We use DVC with Google drive remote for efficient models version control. 
In case you want to get access to our fine tuned models, you can load them from from [here](https://drive.google.com/drive/folders/1ABnPliL2ouDX7vK9RpaUZLLawxPRRgyb?usp=sharing). 
Also you can train your own model by running ```modules/model_trainer.py``` script. 


#### API setup and run

- Clone the official WikiCheck repo and cd into it 

```git clone https://github.com/trokhymovych/WikiCheck.git```

```cd WikiCheck```

- Create and activate virtualenv: 

```virtualenv -p python venv```

```source venv/bin/activate```

- Install requirements from  requirements.txt:

```pip install -r requirements.txt```

- Load pretrained models. There are two options: 
    - Loading models with DVC (preferred):

    ```dvc pull``` 
    
    - Loading models from [here](https://drive.google.com/drive/folders/1ABnPliL2ouDX7vK9RpaUZLLawxPRRgyb?usp=sharing)

- Run the API:

```python start.py --config configs/inference/sentence_bert_config.json```


