import torch
from torch import nn

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from torchsummary import summary

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            print(loader.loader_train.batch_size)

            _model = model.Model(args, checkpoint)
            #summary(_model.model, (3,128,128))
            if(args.fine_tune):
                l = [module for module in _model.model.modules() if not isinstance(module, nn.Sequential)]
                
                for layer in l[0:-5]:
                    #print(layer)
                    for param in layer.parameters():
                        param.requires_grad = False
             #   summary(_model.model, (3,128,128))

                for layer in l[-5:]:
                    print("Trainable layers")
                    print(layer)
                    for param in layer.parameters():
                        param.requires_grad = True
                
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
