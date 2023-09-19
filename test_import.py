from cifar_model.quant_resnet4cifar_my import resnet20_quant
from easydict import EasyDict as edict
from trainer import Trainer

def main():
    args = edict()
    args['QWeightFlag'] = True
    args['QActFlag'] = True
    args['bkwd_scaling_factorW'] = 1.0
    args['bkwd_scaling_factorA'] = 1.0
    print(args.QWeightFlag)
    print(resnet20_quant(args))


if __name__ == '__main__':
    main()
