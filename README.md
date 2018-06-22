# twitter_sentiment_classification

Files inside data folder are generated from: [odymao](https://github.com/odymao/Representations-for-linguistic-sentiment-content-using-computational-intelligence)



## Installation guide

1.  Download and setup [git](https://git-scm.com/downloads).
    

2.  Download and setup [anaconda](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe) (for installing Python 3.6).
    

3.  Install Tensorflow (with Anaconda):
```
conda create -n tensorflow pip python=3.5
activate tensorflow
```

* ...without GPU support:

    
    
```
pip install --ignore-installed --upgrade tensorflow
```

* ...with GPU support:

```
pip install --ignore-installed --upgrade tensorflow-gpu 
```

4.  Download repo, run:

```
git clone https://github.com/sotiristsak/twitter_sentiment_classification.git
```

5.  Install dependencies, run:
```
cd twitter_sentiment_classification
pip install -r requirements.txt
```

6.  Start training, run:
```
python main.py
```

## Run demo notebook on Google Colab
[demo.ipynb](https://github.com/sotiristsak/twitter_sentiment_classification/blob/master/demo.ipynb)
