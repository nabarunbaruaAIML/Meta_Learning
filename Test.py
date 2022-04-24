from transformers import   AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from transformers import pipeline
import torch

def main():
    model = AutoModelForSequenceClassification.from_pretrained('Artifact/Model')
    tokenizer = AutoTokenizer.from_pretrained( 'Artifact/Model')
    # data_collator = DataCollatorWithPadding(tokenizer,padding = 'max_length', max_length=512  )
    sentence = "This chair is very comfortable, but I don't think the leather is real. The seat ripped before one month was up from purchase. It was easy to assemble, but not worth the money."
    token = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        output = model(**token)
    predictions = output.logits.argmax(dim=-1)
    soft = torch.nn.Softmax(dim=1)(output.logits).tolist()
    print('Label: ',model.config.id2label[predictions[0].tolist()])
    print('Probability: ',round(soft[0][predictions[0].tolist()],3))
    
    # model_checkpoint = "Artifact\Model"
    # token_classifier = pipeline( "sentiment-analysis", model=model_checkpoint )
    # print(token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn."))
    
if __name__=="__main__":
    main()