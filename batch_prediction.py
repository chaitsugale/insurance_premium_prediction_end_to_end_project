from india_insurance.pipeline.train_pipeline import start_training_pipeline
from india_insurance.pipeline.batch_prediction import start_batch_prediction

file_path="notebook/insurance-premium-prediction/insurance.csv"

print(__name__)

if __name__=="__main__":
    try:
        output_file = start_batch_prediction(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        print(e)
