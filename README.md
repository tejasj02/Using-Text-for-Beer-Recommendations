# Text for beer recommender

## Motivation
Often it is hard to connect non-quantitative reviews together, so I wanted to create a sort of recommender or general model that could take in descriptions of what you'd want in a product and return a product similar to that.
## Data
The original dataset can be found here: https://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect.
The data was cleaned to just the 20 most popular beers, which equated to around 50,000 reviews. That cleaned dataset can be found under the data folder.
## Naive
The naive approach consisted of returning the most common beer in the dataset. This was quite unsuccessful due to the even distribution of beers, resulting in little accuracy.
## Non-DL
The Non-DL approach was TF-IDF. The reveiws were grouped so each beer funcioned like a document which composed all of its reviews. The maximum features per vector was then tested upon to be optimized, and then retrained and vectorized again. The vectors were then related via cosine similarity for testing.
## Deep Learning
The Deep Learning approach used a fine-tuned version of DistillBERT. The lightweightness of the model allowed more experimentation in the training process, hence its selection. The last two layers were unfrozen for training, and parameters were varied till a decision was made. Another output layer was trained for the classification esque task.
## Evaluation
The evaluation metric used was macro average. This was selected to see if there were balanced results across all 20 beers. The results were as follows:
<br/> naive = 0
<br/> nondl = .39
<br/> dl = .68
## Demo
You can view the demo at this link http://18.221.22.32:8501/ and enter your own beer description to see what beer is best for you!
## How to Run
Pull model, tokenizer, etc. from here https://drive.google.com/drive/folders/1xbt7Xfi9VKPr-DMACggLUl5WAjKZpto4?usp=sharing save in folder called saved_model.
Scripts can be run from main.py. Data is already included in the data folder, to access the full dataset see the link above. main.py runs all model training scripts and returns classification reports where you can view the macro average. If you would like to run specific functions, you may adjust main.py accordingly to call what is needed.
