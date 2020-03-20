import torch.nn as nn
import torch.nn.functional as F

# =======================
# CVAE (CNN version)
# =======================
class CVAE_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        def c_block(in_filters, out_filters, kernel_size = 3, stride = 1, padding = 1, bn=False):
            if bn:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding), 
                         nn.BatchNorm2d(out_filters),
                         nn.LeakyReLU(0.2, inplace = True), 
                         nn.Dropout2d(0.25)]
            else:
                block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding), 
                         nn.LeakyReLU(0.2, inplace = True), 
                         nn.Dropout2d(0.25)]
            return block
        
        #self.deconv_en = nn.ConvTranspose2d(2,1,10, stride = 1, padding=0)
        #self.decon_de = nn.ConvTranspose2d(2,1,10, stride = 1, padding = 0)
        
        # ====================
        # Encoder part
        # ====================
        self.deconv_en = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
        )
        self.encoder = nn.Sequential(
            *c_block(7,   16, 3, 2, 1),
            *c_block(16,  32, 3, 1, 0),
            *c_block(32,  64, 3, 1, 0),
        )
        
        self.mu_mlp = nn.Linear(64, 2)
        self.logvar_mlp = nn.Linear(64, 2)
        
        # ====================
        # Decoder part
        # ====================
        self.deconv_de = nn.Sequential(
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners = True),
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
            nn.ConvTranspose2d(6, 6, 3, 1, 0),
        )
        self.decoder = nn.Sequential(
            *c_block(7,   16, 3, 2, 1),
            *c_block(16,  32, 3, 1, 0),
            *c_block(32,  64, 3, 1, 0),
        )
        
        self.predict_mlp = nn.Linear(64, 2)
        
        
    def forward(self, x, c):
        # Encoder
        x = x.view(x.shape[0], 2, 1, -1)
        cat = c[:, 0:4]
        cat = cat.view(cat.shape[0], 4, 1, -1)
        x_en = torch.cat((x, cat), 1)
        c_en = c[:, 4:].view(c.shape[0], 1, 10, -1)
        deconv_x = self.deconv_en(x_en)
        input_en = torch.cat((deconv_x, c_en), 1)
        output_en = self.encoder(input_en)
        output_en = output_en.view(output_en.shape[0], 1, 64)
        z_mu = self.mu_mlp(output_en)
        z_logvar = self.logvar_mlp(output_en)

        
        # Latent space
        eps = torch.randn(size = z_mu.shape)
        if torch.cuda.is_available():
            eps = eps.cuda()
        tmp = torch.exp(z_logvar / 2)*eps
        if torch.cuda.is_available():
            tmp = tmp.cuda()
        z = z_mu + tmp
        z = z.view(z.shape[0], 2, 1, -1)
        # Decoder 
        z = torch.cat((z, cat), 1)
        z_deconv = self.deconv_de(z)
        input_decoder = torch.cat((z_deconv,c_en), 1)
        output_decoder = self.decoder(input_decoder)
        output_decoder = output_decoder.view(output_decoder.shape[0], 1, 64)
        predict = self.predict_mlp(output_decoder)
        predict = predict.view(predict.shape[0],-1)
        return z_mu, z_logvar, predict

    
def KL_loss(z_mu, z_logvar):
    return 2 * torch.sum(torch.exp(z_logvar) + 
                        z_mu**2 - 1. - z_logvar, dim=1).mean()
def Recon_loss(labels, predictions):
    loss = nn.MSELoss()
    return loss(labels, predictions)

