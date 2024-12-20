## Training and validation logs from Colab Notebook

```
model running on:  cuda
EPOCH: 1
  0%|          | 0/391 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/utils/data/dataloader.py:617: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
Loss=1.260455846786499 Batch_id=390 Accuracy=41.37: 100%|██████████| 391/391 [00:16<00:00, 23.52it/s]

Test set: Average loss: 1.6352, Accuracy: 4560/10000 (45.60%)

EPOCH: 2
Loss=1.2167580127716064 Batch_id=390 Accuracy=55.96: 100%|██████████| 391/391 [00:14<00:00, 26.09it/s]

Test set: Average loss: 1.0665, Accuracy: 6169/10000 (61.69%)

EPOCH: 3
Loss=1.2578181028366089 Batch_id=390 Accuracy=61.50: 100%|██████████| 391/391 [00:16<00:00, 24.11it/s]

Test set: Average loss: 1.0071, Accuracy: 6464/10000 (64.64%)

EPOCH: 4
Loss=0.8524954915046692 Batch_id=390 Accuracy=64.97: 100%|██████████| 391/391 [00:15<00:00, 25.73it/s]

Test set: Average loss: 1.0250, Accuracy: 6453/10000 (64.53%)

EPOCH: 5
Loss=0.8324249982833862 Batch_id=390 Accuracy=67.39: 100%|██████████| 391/391 [00:14<00:00, 26.16it/s]

Test set: Average loss: 0.9250, Accuracy: 6725/10000 (67.25%)

EPOCH: 6
Loss=0.9456140398979187 Batch_id=390 Accuracy=69.79: 100%|██████████| 391/391 [00:15<00:00, 25.55it/s]

Test set: Average loss: 0.7415, Accuracy: 7384/10000 (73.84%)

EPOCH: 7
Loss=0.9349117279052734 Batch_id=390 Accuracy=71.49: 100%|██████████| 391/391 [00:14<00:00, 26.09it/s]

Test set: Average loss: 0.8566, Accuracy: 7000/10000 (70.00%)

EPOCH: 8
Loss=0.9847301244735718 Batch_id=390 Accuracy=72.49: 100%|██████████| 391/391 [00:15<00:00, 25.93it/s]

Test set: Average loss: 0.7214, Accuracy: 7488/10000 (74.88%)

EPOCH: 9
Loss=0.6765730977058411 Batch_id=390 Accuracy=74.05: 100%|██████████| 391/391 [00:15<00:00, 25.83it/s]

Test set: Average loss: 0.6638, Accuracy: 7685/10000 (76.85%)

EPOCH: 10
Loss=0.6368850469589233 Batch_id=390 Accuracy=74.81: 100%|██████████| 391/391 [00:14<00:00, 26.26it/s]

Test set: Average loss: 0.6398, Accuracy: 7771/10000 (77.71%)

EPOCH: 11
Loss=0.545637309551239 Batch_id=390 Accuracy=75.50: 100%|██████████| 391/391 [00:16<00:00, 23.33it/s]

Test set: Average loss: 0.6251, Accuracy: 7819/10000 (78.19%)

EPOCH: 12
Loss=0.7309141755104065 Batch_id=390 Accuracy=76.53: 100%|██████████| 391/391 [00:14<00:00, 26.19it/s]

Test set: Average loss: 0.6467, Accuracy: 7789/10000 (77.89%)

EPOCH: 13
Loss=0.5841480493545532 Batch_id=390 Accuracy=77.06: 100%|██████████| 391/391 [00:15<00:00, 25.95it/s]

Test set: Average loss: 0.6176, Accuracy: 7876/10000 (78.76%)

EPOCH: 14
Loss=0.610268235206604 Batch_id=390 Accuracy=77.70: 100%|██████████| 391/391 [00:14<00:00, 26.15it/s]

Test set: Average loss: 0.6018, Accuracy: 7909/10000 (79.09%)

EPOCH: 15
Loss=0.6814438104629517 Batch_id=390 Accuracy=78.17: 100%|██████████| 391/391 [00:14<00:00, 26.10it/s]

Test set: Average loss: 0.5509, Accuracy: 8113/10000 (81.13%)

EPOCH: 16
Loss=0.6868442296981812 Batch_id=390 Accuracy=78.39: 100%|██████████| 391/391 [00:14<00:00, 26.22it/s]

Test set: Average loss: 0.5833, Accuracy: 8035/10000 (80.35%)

EPOCH: 17
Loss=0.7898532152175903 Batch_id=390 Accuracy=79.08: 100%|██████████| 391/391 [00:14<00:00, 26.21it/s]

Test set: Average loss: 0.5843, Accuracy: 8040/10000 (80.40%)

EPOCH: 18
Loss=0.5668607354164124 Batch_id=390 Accuracy=79.05: 100%|██████████| 391/391 [00:14<00:00, 26.41it/s]

Test set: Average loss: 0.5589, Accuracy: 8131/10000 (81.31%)

EPOCH: 19
Loss=0.48928794264793396 Batch_id=390 Accuracy=79.66: 100%|██████████| 391/391 [00:15<00:00, 26.03it/s]

Test set: Average loss: 0.5395, Accuracy: 8188/10000 (81.88%)

EPOCH: 20
Loss=0.5538708567619324 Batch_id=390 Accuracy=80.09: 100%|██████████| 391/391 [00:14<00:00, 26.19it/s]

Test set: Average loss: 0.5978, Accuracy: 7986/10000 (79.86%)

EPOCH: 21
Loss=0.6138526201248169 Batch_id=390 Accuracy=80.39: 100%|██████████| 391/391 [00:14<00:00, 26.19it/s]

Test set: Average loss: 0.5429, Accuracy: 8092/10000 (80.92%)

EPOCH: 22
Loss=0.6080974340438843 Batch_id=390 Accuracy=80.37: 100%|██████████| 391/391 [00:15<00:00, 25.99it/s]

Test set: Average loss: 0.5224, Accuracy: 8239/10000 (82.39%)

EPOCH: 23
Loss=0.5727637410163879 Batch_id=390 Accuracy=80.63: 100%|██████████| 391/391 [00:14<00:00, 26.38it/s]

Test set: Average loss: 0.5109, Accuracy: 8276/10000 (82.76%)

EPOCH: 24
Loss=0.47472482919692993 Batch_id=390 Accuracy=81.04: 100%|██████████| 391/391 [00:15<00:00, 25.88it/s]

Test set: Average loss: 0.5217, Accuracy: 8226/10000 (82.26%)

EPOCH: 25
Loss=0.5662104487419128 Batch_id=390 Accuracy=81.33: 100%|██████████| 391/391 [00:14<00:00, 26.31it/s]

Test set: Average loss: 0.4765, Accuracy: 8363/10000 (83.63%)

EPOCH: 26
Loss=0.42380914092063904 Batch_id=390 Accuracy=81.58: 100%|██████████| 391/391 [00:16<00:00, 24.31it/s]

Test set: Average loss: 0.5138, Accuracy: 8274/10000 (82.74%)

EPOCH: 27
Loss=0.45632320642471313 Batch_id=390 Accuracy=81.69: 100%|██████████| 391/391 [00:14<00:00, 26.16it/s]

Test set: Average loss: 0.4855, Accuracy: 8376/10000 (83.76%)

EPOCH: 28
Loss=0.45893198251724243 Batch_id=390 Accuracy=81.95: 100%|██████████| 391/391 [00:15<00:00, 26.00it/s]

Test set: Average loss: 0.4971, Accuracy: 8341/10000 (83.41%)

EPOCH: 29
Loss=0.577580988407135 Batch_id=390 Accuracy=82.07: 100%|██████████| 391/391 [00:14<00:00, 26.08it/s]

Test set: Average loss: 0.4789, Accuracy: 8349/10000 (83.49%)

EPOCH: 30
Loss=0.5034649968147278 Batch_id=390 Accuracy=82.23: 100%|██████████| 391/391 [00:15<00:00, 26.01it/s]

Test set: Average loss: 0.4842, Accuracy: 8366/10000 (83.66%)

EPOCH: 31
Loss=0.6198814511299133 Batch_id=390 Accuracy=82.18: 100%|██████████| 391/391 [00:14<00:00, 26.47it/s]

Test set: Average loss: 0.5106, Accuracy: 8268/10000 (82.68%)

EPOCH: 32
Loss=0.48808568716049194 Batch_id=390 Accuracy=82.54: 100%|██████████| 391/391 [00:14<00:00, 26.19it/s]

Test set: Average loss: 0.4603, Accuracy: 8422/10000 (84.22%)

EPOCH: 33
Loss=0.3705518841743469 Batch_id=390 Accuracy=82.85: 100%|██████████| 391/391 [00:14<00:00, 26.16it/s]

Test set: Average loss: 0.4497, Accuracy: 8466/10000 (84.66%)

EPOCH: 34
Loss=0.40688222646713257 Batch_id=390 Accuracy=82.79: 100%|██████████| 391/391 [00:14<00:00, 26.28it/s]

Test set: Average loss: 0.4865, Accuracy: 8368/10000 (83.68%)

EPOCH: 35
Loss=0.42879390716552734 Batch_id=390 Accuracy=82.99: 100%|██████████| 391/391 [00:15<00:00, 26.05it/s]

Test set: Average loss: 0.4702, Accuracy: 8416/10000 (84.16%)

EPOCH: 36
Loss=0.2693183720111847 Batch_id=390 Accuracy=83.30: 100%|██████████| 391/391 [00:14<00:00, 26.36it/s]

Test set: Average loss: 0.4868, Accuracy: 8377/10000 (83.77%)

EPOCH: 37
Loss=0.452004998922348 Batch_id=390 Accuracy=83.32: 100%|██████████| 391/391 [00:14<00:00, 26.21it/s]

Test set: Average loss: 0.4521, Accuracy: 8489/10000 (84.89%)

EPOCH: 38
Loss=0.603645384311676 Batch_id=390 Accuracy=83.37: 100%|██████████| 391/391 [00:14<00:00, 26.11it/s]

Test set: Average loss: 0.4501, Accuracy: 8479/10000 (84.79%)

EPOCH: 39
Loss=0.552642285823822 Batch_id=390 Accuracy=83.46: 100%|██████████| 391/391 [00:15<00:00, 25.97it/s]

Test set: Average loss: 0.4470, Accuracy: 8474/10000 (84.74%)

EPOCH: 40
Loss=0.39833930134773254 Batch_id=390 Accuracy=83.77: 100%|██████████| 391/391 [00:16<00:00, 23.65it/s]

Test set: Average loss: 0.4671, Accuracy: 8453/10000 (84.53%)

EPOCH: 41
Loss=0.4198873043060303 Batch_id=390 Accuracy=83.72: 100%|██████████| 391/391 [00:14<00:00, 26.13it/s]

Test set: Average loss: 0.4467, Accuracy: 8477/10000 (84.77%)

EPOCH: 42
Loss=0.4761854112148285 Batch_id=390 Accuracy=83.71: 100%|██████████| 391/391 [00:14<00:00, 26.28it/s]

Test set: Average loss: 0.4487, Accuracy: 8491/10000 (84.91%)

EPOCH: 43
Loss=0.28720229864120483 Batch_id=390 Accuracy=84.20: 100%|██████████| 391/391 [00:14<00:00, 26.28it/s]

Test set: Average loss: 0.4406, Accuracy: 8504/10000 (85.04%)

EPOCH: 44
Loss=0.466238409280777 Batch_id=390 Accuracy=84.15: 100%|██████████| 391/391 [00:14<00:00, 26.32it/s]

Test set: Average loss: 0.4404, Accuracy: 8535/10000 (85.35%)

EPOCH: 45
Loss=0.37305712699890137 Batch_id=390 Accuracy=84.16: 100%|██████████| 391/391 [00:14<00:00, 26.60it/s]

Test set: Average loss: 0.4427, Accuracy: 8495/10000 (84.95%)

EPOCH: 46
Loss=0.5065498948097229 Batch_id=390 Accuracy=84.06: 100%|██████████| 391/391 [00:15<00:00, 26.03it/s]

Test set: Average loss: 0.4446, Accuracy: 8513/10000 (85.13%)

EPOCH: 47
Loss=0.4016569256782532 Batch_id=390 Accuracy=84.44: 100%|██████████| 391/391 [00:15<00:00, 25.87it/s]

Test set: Average loss: 0.4517, Accuracy: 8481/10000 (84.81%)

EPOCH: 48
Loss=0.2598184645175934 Batch_id=390 Accuracy=84.40: 100%|██████████| 391/391 [00:14<00:00, 26.47it/s]

Test set: Average loss: 0.4251, Accuracy: 8580/10000 (85.80%)

EPOCH: 49
Loss=0.36957162618637085 Batch_id=390 Accuracy=84.43: 100%|██████████| 391/391 [00:14<00:00, 26.35it/s]

Test set: Average loss: 0.4303, Accuracy: 8567/10000 (85.67%)

EPOCH: 50
Loss=0.46526843309402466 Batch_id=390 Accuracy=84.66: 100%|██████████| 391/391 [00:14<00:00, 26.24it/s]

Test set: Average loss: 0.4422, Accuracy: 8543/10000 (85.43%)

+-------+-------------------+---------------+-------+---------------+-----------+
| Epoch | Training Accuracy | Test Accuracy |  Diff | Training Loss | Test Loss |
+-------+-------------------+---------------+-------+---------------+-----------+
|   1   |       41.37%      |     45.60%    | -4.23 |     1.2605    |   1.6352  |
|   2   |       55.96%      |     61.69%    | -5.73 |     1.2168    |   1.0665  |
|   3   |       61.50%      |     64.64%    | -3.14 |     1.2578    |   1.0071  |
|   4   |       64.97%      |     64.53%    |  0.44 |     0.8525    |   1.0250  |
|   5   |       67.39%      |     67.25%    |  0.14 |     0.8324    |   0.9250  |
|   6   |       69.79%      |     73.84%    | -4.05 |     0.9456    |   0.7415  |
|   7   |       71.49%      |     70.00%    |  1.49 |     0.9349    |   0.8566  |
|   8   |       72.49%      |     74.88%    | -2.39 |     0.9847    |   0.7214  |
|   9   |       74.05%      |     76.85%    | -2.80 |     0.6766    |   0.6638  |
|   10  |       74.81%      |     77.71%    | -2.90 |     0.6369    |   0.6398  |
|   11  |       75.50%      |     78.19%    | -2.69 |     0.5456    |   0.6251  |
|   12  |       76.53%      |     77.89%    | -1.36 |     0.7309    |   0.6467  |
|   13  |       77.06%      |     78.76%    | -1.70 |     0.5841    |   0.6176  |
|   14  |       77.70%      |     79.09%    | -1.39 |     0.6103    |   0.6018  |
|   15  |       78.17%      |     81.13%    | -2.96 |     0.6814    |   0.5509  |
|   16  |       78.39%      |     80.35%    | -1.96 |     0.6868    |   0.5833  |
|   17  |       79.08%      |     80.40%    | -1.32 |     0.7899    |   0.5843  |
|   18  |       79.05%      |     81.31%    | -2.26 |     0.5669    |   0.5589  |
|   19  |       79.66%      |     81.88%    | -2.22 |     0.4893    |   0.5395  |
|   20  |       80.09%      |     79.86%    |  0.23 |     0.5539    |   0.5978  |
|   21  |       80.39%      |     80.92%    | -0.53 |     0.6139    |   0.5429  |
|   22  |       80.37%      |     82.39%    | -2.02 |     0.6081    |   0.5224  |
|   23  |       80.63%      |     82.76%    | -2.13 |     0.5728    |   0.5109  |
|   24  |       81.04%      |     82.26%    | -1.22 |     0.4747    |   0.5217  |
|   25  |       81.33%      |     83.63%    | -2.30 |     0.5662    |   0.4765  |
|   26  |       81.58%      |     82.74%    | -1.16 |     0.4238    |   0.5138  |
|   27  |       81.69%      |     83.76%    | -2.07 |     0.4563    |   0.4855  |
|   28  |       81.95%      |     83.41%    | -1.46 |     0.4589    |   0.4971  |
|   29  |       82.07%      |     83.49%    | -1.42 |     0.5776    |   0.4789  |
|   30  |       82.23%      |     83.66%    | -1.43 |     0.5035    |   0.4842  |
|   31  |       82.18%      |     82.68%    | -0.50 |     0.6199    |   0.5106  |
|   32  |       82.54%      |     84.22%    | -1.68 |     0.4881    |   0.4603  |
|   33  |       82.85%      |     84.66%    | -1.81 |     0.3706    |   0.4497  |
|   34  |       82.79%      |     83.68%    | -0.89 |     0.4069    |   0.4865  |
|   35  |       82.99%      |     84.16%    | -1.17 |     0.4288    |   0.4702  |
|   36  |       83.30%      |     83.77%    | -0.47 |     0.2693    |   0.4868  |
|   37  |       83.32%      |     84.89%    | -1.57 |     0.4520    |   0.4521  |
|   38  |       83.37%      |     84.79%    | -1.42 |     0.6036    |   0.4501  |
|   39  |       83.46%      |     84.74%    | -1.28 |     0.5526    |   0.4470  |
|   40  |       83.77%      |     84.53%    | -0.76 |     0.3983    |   0.4671  |
|   41  |       83.72%      |     84.77%    | -1.05 |     0.4199    |   0.4467  |
|   42  |       83.71%      |     84.91%    | -1.20 |     0.4762    |   0.4487  |
|   43  |       84.20%      |     85.04%    | -0.84 |     0.2872    |   0.4406  |
|   44  |       84.15%      |     85.35%    | -1.20 |     0.4662    |   0.4404  |
|   45  |       84.16%      |     84.95%    | -0.79 |     0.3731    |   0.4427  |
|   46  |       84.06%      |     85.13%    | -1.07 |     0.5065    |   0.4446  |
|   47  |       84.44%      |     84.81%    | -0.37 |     0.4017    |   0.4517  |
|   48  |       84.40%      |     85.80%    | -1.40 |     0.2598    |   0.4251  |
|   49  |       84.43%      |     85.67%    | -1.24 |     0.3696    |   0.4303  |
|   50  |       84.66%      |     85.43%    | -0.77 |     0.4653    |   0.4422  |
+-------+-------------------+---------------+-------+---------------+-----------+

```