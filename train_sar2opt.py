"""SAR2Opt 512x512: independent S0/S1/S2 training with explicit SAR/Optical pairs."""
from train_qxs import main

if __name__ == '__main__':
    main(dataset='SAR2Opt', image_size=512)
