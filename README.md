# GPT  
Toy implementation of [gpt2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Because language models are cool.

Due to compute restraints I can not train the full size GPT-2 model. The largest one I could train is a 352M variant (can be run with the `train.sh` script) and it converged to a loss of 3.1, which is alright.

![image](/images/gpt2_loss.png)

The total footprint of the code is quite small so it is fairly easy to modifie. `train.py` exposes a cli to set all the hyperparameters of the model to make it easy to train and iterate on model.
