import torch
import VAE
import ConvEncoderDecoder as CED
import LinearEncoderDecoder as LED

TrainedLinearVAE = VAE.VAE(LED.LinearEncoder(2, 400, 100), LED.LinearDecoder(2, 400, 100))
TrainedLinearVAE.load_state_dict(torch.load('LinearModel.pth', torch.device('cpu')))

TrainedConvVAE = VAE.VAE(CED.ConvEncoder(2, 400), CED.ConvDecoder(2, 400))
TrainedConvVAE.load_state_dict(torch.load('ny_conv.pth', torch.device('cpu')))