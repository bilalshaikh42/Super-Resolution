import glob
import os
from random import shuffle
import random
from data import srdata

class BrainTumor(srdata.SRData):
    def __init__(self, args, name='BrainTumor', train=True, benchmark=False):

        data_range = [[1,5000],[5000, 5023]]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        

        super(BrainTumor, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )


    def _scan(self):
        names_hr = glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        # Since the data is sorted by tumor type, we want to shuffle so that validation set is not all the same type
        random.shuffle(names_hr)
        
        
        
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
        
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        return names_hr, names_lr

    
    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/BrainTumor'
        super(BrainTumor, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR_train')

        self.dir_lr = os.path.join(self.apath, 'LR_train')
        self.ext = ('.png', '.png')
        

