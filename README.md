#  Experiment

# Dataset
This dataset is a 10.47 hours speech dataset consisting of 16,830 samples collected from 287 contributors, specifically focusing on 418 acupuncture points.

## Split Ratio
Taking into account the size of our dataset, we partition it into training, validation, and test sets using a **balanced** 0.7:0.15:0.15 ratio. To ensure an even distribution, we randomly select samples from each class for every subset, maintaining a balanced representation of all classes.

## Loss Function
We use **Binary Cross Entropy** as the loss function instead of Cross Entropy, which means that we use sigmoid instead of softmax for the activation function in the last layer. We choose BCE as the loss in order to facilitate our training of the audio classifier using the *mixup*[^1] technique.

## Hyperparameters
| $lr$ | $epoch$ | $mask_{frequence}$ | $mask_{time}$ | $mixup$ |
| -------- | -------- | -------- | -------- |  -------- | 
| 0.0003  | 50  | 40  | 40 | 0.6 |

We utilize the Adam Optimizer for our model training.

Warmup: During the initial phase, we increase the learning rate every 50 steps, gradually reaching a learning rate of 0.0003 by the 2000th step.

Scheduler: After the warmup phase, starting from the 5th epoch, we implement a learning rate schedule by adjusting the learning rate for each subsequent epoch, multiplying it by 0.9 times.

## Evaluation Metrics 

We choose the model based on the **recall** on the validation set. Specifically, we select the model that achieves the highest recall score on the validation set across all epochs. All metrics presented below are evaluated using this chosen model.

All metrics are calculated using the **macro** approach, which involves computing the metrics for each class separately and then averaging them to obtain a single metric.

###  Metrics on the validation set

| $recall$ | $precision$ | $f1$ | $accuracy$ | $AP$ |
| -------- | -------- | -------- | -------- |  -------- | 
| 95.43%  | 95.72%  | 95.22%  | 95.34% | 97.17% |


### Metrics on the test set
| $recall$ | $precision$ | $f1$ | $accuracy$ | $AP$ |
| -------- | -------- | -------- | -------- |  -------- | 
| 95.35%  | 96.01%  | 95.34%  | 95.35% | 97.15% |

### PR curve on the test set
| $threshold$ | $precision$ | $recall$ |
| -------- | -------- | -------- | 
| 0.3  | 93.4%  | 93.29%  |
| 0.3  | 96.14%  | 90.23%  |
| 0.4  | 98.04%  | 86.29%  |
| 0.5  | 98.86%  | 80.1%  |
| 0.6  | 99.62%  | 72.98%  |
| 0.7  | 99.94%  | 63.32%  |

### PR curve on the validation set
| $threshold$ | $precision$ | $recall$ |
| -------- | -------- | -------- | 
| 0.3  | 93.4%  | 93.29%  |
| 0.3  | 96.14%  | 90.23%  |
| 0.4  | 98.04%  | 86.29%  |
| 0.5  | 98.43%  | 80.92%  |
| 0.6  | 99.62%  | 72.98%  |
| 0.7  | 99.94%  | 63.32%  |

[^1]: Tokozume, Y., Ushiku, Y., & Harada, T. (2018). Learning from between-class examples for deep sound recognition. In _ICLR_.

