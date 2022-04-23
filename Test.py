from transformers import   AutoModelForSequenceClassification
from transformers import pipeline

def main():
    model = AutoModelForSequenceClassification.from_pretrained('Artifact/Model')
    print(model)
    model_checkpoint = "Artifact\Model"
    token_classifier = pipeline( "sentiment-analysis", model=model_checkpoint )
    print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))
    
if __name__=="__main__":
    main()