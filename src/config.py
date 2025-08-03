
class MedAlignConfig():
    MODEL = "MedAlign"
    TASK = 'MIII'
    RATIO = 2/3

    SEED = 2023
    USE_CUDA = True
    GPU = '0'
    EPOCH = 1000
    DIM = 64
    LR = 5e-4 
    BATCH = 32
    WD = 0
    DDI = 0.06
    KP = 0.08
    HIST = 3 

    ROOT = '../data/'
    LOG = '../log/'

config = vars(MedAlignConfig)
config = {k:v for k,v in config.items() if not k.startswith('__')}
