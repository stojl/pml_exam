import torch
import VAE
import ConvEncoderDecoder as CED
import LinearEncoderDecoder as LED

TrainedConvVAE = VAE.VAE(CED.ConvEncoder(2, 16), CED.ConvDecoder(2, 16))
TrainedConvVAE.load_state_dict(torch.load('conv_model.pth', torch.device('cpu')))

TrainedLinearVAE = VAE.VAE(LED.LinearEncoder(2, 400, 100), LED.LinearDecoder(2, 400, 100))
TrainedLinearVAE.load_state_dict(torch.load('LinearModel.pth', torch.device('cpu')))