import math

class LossResponsiveLRSchedule:
    def __init__(self, initial_lr: float, final_lr: float, decay_fraction: float,
                 spike_threshold: float = 1.1,  # Multiplicative increase in loss to trigger
                 spike_lr_multiplier: float = 2.0,  # How much to multiply LR on spike
                 cooldown_steps: int = 10,  # Steps to wait after adjustment
                 min_lr: float = 1e-6,
                 max_lr: float = 1.0,
                 decay_type: str = "exponential",  # "exponential" or "linear"
                 k: float = 5.0  # For exponential decay steepness
                 ):
        """
        Loss-responsive learning rate scheduler.

        Args:
            initial_lr: Initial learning rate.
            final_lr: Final learning rate for decay.
            decay_fraction: Fraction of total steps for decay.
            spike_threshold:  Loss increase factor to trigger a spike (e.g., 1.1 = 10% increase).
            spike_lr_multiplier: Factor to multiply LR by on a spike.
            cooldown_steps: Steps to wait before considering another adjustment.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
            decay_type: Type of decay ("exponential" or "linear").
            k: Steepness for exponential decay (higher k = steeper).
        """
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_fraction = decay_fraction
        self.spike_threshold = spike_threshold
        self.spike_lr_multiplier = spike_lr_multiplier
        self.cooldown_steps = cooldown_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.decay_type = decay_type
        self.k = k
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.steps_since_adjustment = 0
        self.total_steps_taken = 0  # Track total steps across the entire training
        self.in_cooldown = False
        self.last_adjustment_step = -cooldown_steps  # Ensure we don't start in cooldown


    def _exponential_decay(self, progress):
        return self.final_lr + (self.current_lr - self.final_lr) * math.exp(-self.k * progress)
    
    def _linear_decay(self, progress):
        return self.current_lr - progress * (self.current_lr - self.final_lr)

    def get_lr(self, current_loss: float, current_step: int, total_training_steps :int) -> float:
        """
        Gets the learning rate, adjusting based on loss spikes.

        Args:
            current_loss: The current training loss.
            current_step: The current step *within the current decay phase*.
            total_training_steps: Total number of training steps

        Returns:
            The calculated learning rate.
        """
        self.total_steps_taken +=1
        progress_remaining = 1.0 - (current_step / (total_training_steps * self.decay_fraction))
        progress = 1.0 - max(0, min(progress_remaining, 1.0)) # force between 0-1, where 0 is start and 1 is end.

        if self.in_cooldown:
            self.steps_since_adjustment += 1
            if self.steps_since_adjustment >= self.cooldown_steps:
                self.in_cooldown = False
                self.steps_since_adjustment = 0
                self.best_loss = float('inf')  # Reset best loss after cooldown


        if not self.in_cooldown and current_loss > self.best_loss * self.spike_threshold:
            # Loss spike detected!
            self.current_lr = min(self.current_lr * self.spike_lr_multiplier, self.max_lr)
            self.in_cooldown = True
            self.last_adjustment_step = self.total_steps_taken #Global step count
            print(f"Spike detected at step {self.total_steps_taken}. Increasing LR to {self.current_lr}")
            # Resetting best loss helps avoid repeated triggers by small increases
            self.best_loss = current_loss
            return self.current_lr  # Return the increased LR immediately

        # Apply decay (only if not in cooldown or after a spike)
        if self.total_steps_taken >= self.last_adjustment_step + self.cooldown_steps:
          #Normal decay schedule
          if self.decay_type == "exponential":
            decayed_lr = self._exponential_decay(progress)
          elif self.decay_type == "linear":
            decayed_lr = self._linear_decay(progress)
          else:
              raise ValueError("Invalid decay_type. Choose 'exponential' or 'linear'.")

          self.current_lr = max(decayed_lr, self.min_lr)


        # Update best loss (only if not during cooldown from a spike increase)
        if not self.in_cooldown:
          self.best_loss = min(self.best_loss, current_loss)

        return self.current_lr

# --- Example Usage ---
total_training_steps = 1000
scheduler = LossResponsiveLRSchedule(
    initial_lr=0.01,
    final_lr=0.0001,
    decay_fraction=0.8,
    spike_threshold=1.2,  # Trigger on 20% loss increase
    spike_lr_multiplier=2.0,
    cooldown_steps=20,
    min_lr=1e-7,
    max_lr=0.1,
    decay_type="exponential"
)

# Simulate a training loop with fluctuating loss
losses = []
learning_rates = []
current_loss = 1.0  # Initial loss

for step in range(total_training_steps):
    # Simulate loss fluctuations (replace with your actual loss)
    if step < 200:
        current_loss *= 0.95 + 0.1 * (0.5 - random.random()) # Initial rapid decrease + noise
    elif step < 400:
        current_loss *= 0.98 + 0.05 * (0.5 - random.random()) # Slower decrease + less noise
    elif step < 500:
          current_loss *= 1.3 + 0.1 * (0.5-random.random())   # Simulate getting stuck + spike
    elif step < 700:
          current_loss *= 0.97 + 0.03 *(0.5-random.random())  # Moderate Fluctuations
    else:
        current_loss *= 0.99 + 0.01 * (0.5- random.random())  # Final convergence + small noise


    lr = scheduler.get_lr(current_loss, step, total_training_steps)
    losses.append(current_loss)
    learning_rates.append(lr)

    #print(f"Step: {step}, Loss: {current_loss:.4f}, LR: {lr:.6f}")

# Plot the results (optional, requires matplotlib)
import matplotlib.pyplot as plt
import random

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(learning_rates)
plt.title("Learning Rate")
plt.xlabel("Step")
plt.ylabel("LR")
plt.yscale("log")  # Use log scale for LR
plt.show()