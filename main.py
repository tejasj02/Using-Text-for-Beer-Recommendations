from scripts.naive import naive_results
from scripts.inference import predict_beer
from scripts.deep import deep_main
from scripts.non_dl import non_dl_main

if __name__ == "__main__":
    
    # train all models and retrieve classification reports
    naive_result = naive_results() 
    non_dl_result = non_dl_main()
    deep_results = deep_main()

    # run inference on deep learning model
    predict_beer_result = predict_beer("Crisp and hoppy with a slight citrus aftertaste. Pretty refreshing!")