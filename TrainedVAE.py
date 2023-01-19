import torch
import VAE
import VAE2
import ConvEncoderDecoder as CED
import LinearEncoderDecoder as LED

TrainedLinearVAE = VAE.VAE(LED.LinearEncoder(2, 400, 100), LED.LinearDecoder(2, 400, 100))
TrainedLinearVAE.load_state_dict(torch.load('LinearModel.pth', torch.device('cpu')))

TrainedLinearVAE2 = VAE2.VAE(LED.LinearEncoder(2, 400, 100), LED.LinearDecoder(2, 400, 100), 0.05)
TrainedLinearVAE2.load_state_dict(torch.load('NewLinearModel.pth', torch.device('cpu')))

TrainedConvVAE = VAE.VAE(CED.ConvEncoder(2, 200), CED.ConvDecoder(2, 200))
TrainedConvVAE.load_state_dict(torch.load('betterconv.pth', torch.device('cpu')))

TrainedConvVAE2 = VAE2.VAE(CED.ConvEncoder(2, 200), CED.ConvDecoder(2, 200), 0.05)
TrainedConvVAE2.load_state_dict(torch.load('NewConvModel.pth', torch.device('cpu')))
