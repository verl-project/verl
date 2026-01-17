# %%
import torch
from torch.nn import KLDivLoss
from typing import Optional
# %%
class JSD(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.float,
    ):
        super(JSD, self).__init__()
        self.kl = KLDivLoss(reduction="none", log_target=True)
        self.beta = beta
        self.ignore_index = ignore_index
        self.dtype = dtype

    def forward(
        self,
        log_q: torch.Tensor,  # input
        log_p: torch.Tensor,  # target
        label: Optional[torch.Tensor] = None,
    ):
        if self.beta == 0.0:
            loss = self.kl(log_q, log_p).sum(dim=-1)
        elif self.beta == 1.0:
            loss = self.kl(log_p, log_q).sum(dim=-1)
        else:
            log_p, log_q = log_p.to(torch.float), log_q.to(torch.float)
            log_p, log_q = log_p.view(-1, log_p.size(-1)), log_q.view(
                -1, log_q.size(-1)
            )
            m = torch.lerp(torch.exp(log_q), torch.exp(log_p), self.beta)
            loss = self.beta * self.kl(torch.log(m), log_p).sum(dim=-1) + (
                1 - self.beta
            ) * self.kl(torch.log(m), log_q).sum(dim=-1)

        # if label is not None:
        #     loss = torch.where(label != self.ignore_index, loss, 0.0)
        #     n_non_ignore = (label != self.ignore_index).sum().item()
        #     if n_non_ignore == 0:
        #         loss = torch.tensor(0.0).to(loss.device)
        #     else:
        #         loss = (loss / n_non_ignore).sum()
        # else:
        #     loss = (loss / log_q.shape[0]).sum()
        return loss.to(self.dtype)
# %%
class MyJSD():
    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def __call__(self, log_q: torch.Tensor, log_p: torch.Tensor):
        """
        log_q: student log probabilities
        log_p: teacher log probabilities
        beta: weight for forward KL divergence, (1 - beta) is the weight for reverse KL divergence
        forward KL: KL(p || m) where m = beta * p + (1 - beta) * q
        reverse KL: KL(q || m) where m = beta * p + (1 - beta) * q
        """

        log_q = log_q.flatten(0, -2)
        log_p = log_p.flatten(0, -2)
        q = log_q.exp()
        p = log_p.exp()
        m = self.beta * p + (1 - self.beta) * q
        log_m = m.log()
        reverse_kl = (log_q.exp() * (log_q - log_m)).sum(dim=-1)
        forward_kl = (log_p.exp() * (log_p - log_m)).sum(dim=-1)
        loss = self.beta * forward_kl + (1 - self.beta) * reverse_kl
        return loss
    
from verl.trainer.distillation.losses import jensen_shannon_divergence, kullback_leibler_divergence
class verlJSD():
    def __init__(self, beta: float = 0.5):
        self.beta = beta

    def __call__(self, log_q: torch.Tensor, log_p: torch.Tensor):
        if self.beta == 1.0:
            print("Using reverse KL divergence")
            return kullback_leibler_divergence(log_q, log_p, loss_mode="reverse")
        if self.beta == 0.0:
            print("Using forward KL divergence")
            return kullback_leibler_divergence(log_q, log_p, loss_mode="forward")
        return jensen_shannon_divergence(log_q, log_p, self.beta)
    
# %%
B, T, V = (2, 1024, 3200)
betas = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

for beta in betas:
    torch.manual_seed(0)
    torch_jsd = JSD(beta=beta)
    my_jsd = MyJSD(beta=beta)
    verl_jsd = verlJSD(beta=beta)
    input = torch.randn(B * T, V).log_softmax(dim=-1)
    target = torch.randn(B * T, V).log_softmax(dim=-1)

    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)
    x3 = input.detach().clone().requires_grad_(True)
    loss1 = torch_jsd(x1, target)
    my_loss1 = my_jsd(x2, target)
    verl_loss1 = verl_jsd(x3, target)

    # print((loss1 - my_loss1).abs().max(), loss1.min(), loss1.max(), my_loss1.min(), my_loss1.max())
    print((loss1 - verl_loss1).abs().max(), loss1.min(), loss1.max(), verl_loss1.min(), verl_loss1.max())


# %%
verl_loss1.shape
# %%
res = torch.nn.functional.kl_div(x1, target, reduction="none", log_target=True)
res.shape
# %%
(target.exp() * (target - x1)).equal(res)
# %%
