"""
Noam Learning Rate Scheduler

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

class NoamScheduler(LRScheduler):

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    # Compute LR scaling factor for current step
    def _get_lr_scale(self) -> float:

        step = self.last_epoch + 1

        # Noam schedule formula
        return (
            self.d_model ** (-0.5)
            * min(
                step ** (-0.5),                      # inverse sqrt decay
                step * self.warmup_steps ** (-1.5), # linear warmup
            )
        )

    # Return updated LR for each optimizer param group
    def get_lr(self) -> list[float]:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list[float]:

    # Dummy model only needed to create optimizer
    dummy_model = torch.nn.Linear(1, 1)

    # Base LR usually kept at 1.0 for Noam scheduling
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler = NoamScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
    )
    history = []

    # Simulate training loop
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    #Run an example to visualize the learning rate
    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20_000

    # Generate LR schedule
    lrs = get_lr_history(
        D_MODEL,
        WARMUP_STEPS,
        TOTAL_STEPS,
    )

    # Create figure
    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(
        WARMUP_STEPS,
        color="red",
        linestyle="--",
        label=f"warmup={WARMUP_STEPS}",
    )

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()