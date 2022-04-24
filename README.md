# Meta Learning the Huggingface Way

## Authors
**Nabarun Barua**     

[Git](https://github.com/nabarunbaruaAIML)/ [LinkedIn](https://www.linkedin.com/in/nabarun-barua-aiml-engineer/)/ [Towardsdatascience](https://medium.com/@nabarun.barua)

**Arjun Kumbakkara** 

[Git](https://github.com/arjunKumbakkara)/ [LinkedIn](https://www.linkedin.com/in/arjunkumbakkara/)/ [Towardsdatascience](https://medium.com/@arjunkumbakkara)

Meta - Learning is eciting trend in Research and before we jump into Project implementation, I think we first should understand Meta - Learning basics. 

In Traditional ML/DL approach, what we follow is we get huge dataset and we start training on that dataset and eventually we get good accuracy score.

Now Meta - Learning approach is like the way human learns, first Learns on one domain after getting knowledge then it tries to learn on next domain with handful of examples based on the knowledge aquired in the past. In this way, by repeating this step multiple times, we can improve the accuracy of model with limited training data.

There are multiple types of Meta - Learning:
- metric-based meta-learning
    - Siamese Network
    - Matching Network
    - Prototypical Network
    - Relation Network
- optimization-based meta-learning
    - MAML (model-agnostic meta learning) 
    - FOMAML—First-Order MAML
    - Reptile 

Details for the above meta-learning techniques will be available in this [Paper](https://arxiv.org/pdf/2007.09604.pdf)

In our example we used Reptile a variaent MAML (model-agnostic meta learning) from optimization-based meta-learning. 
