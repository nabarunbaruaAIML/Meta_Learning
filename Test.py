from transformers import   AutoModelForSequenceClassification


def main():
    model = AutoModelForSequenceClassification.from_pretrained('Artifact/Model')
    print(model)
    
if __name__=="__main__":
    main()