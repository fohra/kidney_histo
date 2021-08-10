def adjust_LR(lr, batch_size):
    # Adjust learning rate based on batch_size
    new_lr = (batch_size/128)*lr
    
    return new_lr