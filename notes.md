# Obeservations and Next Stepss
## Emperical Analysis
1. Step limit of 10 making current pre-trained model not able to get the success signal - So incresing the step limit might help


## ancient-planet vs vocal-sun (GPRO runs comparison)
1. kl divergence increases as the training continues
2. Train entropy is also not decreasing
3. Loss is also not converging
4. Mean reward is in range of -0/28 to -0.38 and the postive reward fractions is also very less in between 0.18 adn 0.06.

### Fixes for the above problems
1. Increasing the max step size helped the model to reach answer
2. Fixing the kl divergence will make model more stable and avoid catastrophic forgetting, so i tried increasing grpo kl divergence constraint. This has fixed the entropy and loss.
3. Changing the dataloader by slowly increasing the complexity(number of tool calls required) of the queries worked for the converging of the loss