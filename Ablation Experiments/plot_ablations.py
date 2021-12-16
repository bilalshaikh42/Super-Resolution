import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


ablation_results = pd.read_csv('/home/ctebright/EDSR-PyTorch/experiment/test/results-DIV2K/PSNR_ablated_res.csv', header= None)


ablation_blocks = ablation_results[ablation_results[0].str.contains('body')].values

# block_numbers = ablation_blocks[0].str.split('.').str[1]
for i, x in enumerate(ablation_blocks[:,0]):
    ablation_blocks[i][0] = int(x[5:])



ablation_blocks = ablation_blocks[ablation_blocks[:,0].argsort()]
# ablation_blocks[:,0] = np.array(x[5:] for x in ablation_blocks[:,0])
# print(ablation_blocks[:,0])
# ablation_blocks[:][0] = ablation_blocks[:][0]
# print(ablation_blocks)

plt.plot(ablation_blocks[:,1])
plt.xlabel('Ablated Block')
plt.ylabel('PSNR')
plt.title('Ablation Testing by Block')
plt.savefig('/home/ctebright/EDSR-PyTorch/experiment/test/results-DIV2K/AblationTestingByBlock.png')