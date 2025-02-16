import torch

def train(vae, data_loader, optimizer, epochs=5000, interval=100, device='cuda'):
    """
    Trains the VAE model.
    Args:
        vae (nn.Module): VAE model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        interval (int): Interval for logging progress.
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        tuple: Lists of reconstruction loss, regularization loss, total loss, and latent variables.
    """
    rec_error_record, reg_error_record, total_error_record, z_list = [], [], [], []
    for epoch in range(epochs):
        vae.train()
        loss_rec, loss_reg, loss_total = 0, 0, 0
        batch_count = 0 
        for k, (x,) in enumerate(data_loader):
            x = x.to(device).squeeze(0)
            y, z, mu, logvar = vae(x)
            lrec, lreg = vae.loss(y, x, mu, logvar)
            loss = lrec + lreg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_rec += lrec.item()
            loss_reg += lreg.item()
            loss_total += loss.item()
            batch_count += 1
            if epoch == epochs - 1:
                z_list.append(z.cpu().detach().numpy())
        loss_rec /= batch_count
        loss_reg /= batch_count
        loss_total /= batch_count
        rec_error_record.append(loss_rec)
        reg_error_record.append(loss_reg)
        total_error_record.append(loss_total)
        if epoch % interval == 0:
            print(f"Epoch:{epoch} Loss_Rec:{loss_rec} Loss_Reg:{loss_reg} Loss_Total:{loss_total}")
    return rec_error_record, reg_error_record, total_error_record, np.concatenate(z_list, axis=0)
