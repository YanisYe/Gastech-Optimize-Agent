from typing import Optional

import torch
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import  reduce


class CategoricalMasked(Categorical):
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype
            )
            logits = torch.where(self.mask.reshape(-1), logits.reshape(-1), self.mask_value.to(logits.device))
            logits = logits.reshape(self.batch, self.nb_action)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
