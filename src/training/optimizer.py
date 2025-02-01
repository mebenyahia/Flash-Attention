import torch

def create_optimizer_and_scheduler(model, cfg):
    """
    Creates an Adam optimizer with a learning rate scheduler (inverse sqrt with warmup).
    """
    learning_rate = cfg["training"]["learning_rate"]
    warmup_steps = cfg["training"]["warmup_steps"]
    d_model = cfg["model"]["d_model"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
