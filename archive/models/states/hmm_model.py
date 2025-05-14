"""
Gaussian Hidden Markov Model (HMM) for Cryptocurrency Trading

This script implements a GPU-accelerated Hidden Markov Model using PyTorch,
designed for detecting market regimes (bullish/bearish) in cryptocurrency price data.

Features:
- PyTorch implementation with GPU acceleration
- Baum-Welch algorithm for parameter estimation
- Viterbi algorithm for state prediction
- N-state model to identify different market regimes

Usage:
1. Set your data path
2. Customize model parameters
3. Run to train the model and predict states
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time
import os


class GaussianHMM(nn.Module):
    """
    PyTorch implementation of a Gaussian Hidden Markov Model.

    This model is designed to identify different market regimes (states)
    in financial time series data, particularly for cryptocurrency markets.
    """

    def __init__(self, n_states: int, n_features: int, device: str = "cuda"):
        """
        Initialize the Gaussian HMM.

        Args:
            n_states (int): Number of hidden states (e.g., 2 for bullish/bearish).
            n_features (int): Number of features in the observation (e.g., 1 for returns).
            device (str): Computing device ('cuda' or 'cpu').
        """
        super(GaussianHMM, self).__init__()
        self.n_states = n_states
        self.n_features = n_features
        self.device = device

        # Initial state probabilities (to be softmaxed later)
        self.start_prob = nn.Parameter(torch.randn(n_states))

        # Transition matrix (to be softmaxed over rows)
        self.trans_prob = nn.Parameter(torch.randn(n_states, n_states))

        # Emission parameters: means and variances (ensure variances are positive)
        self.means = nn.Parameter(torch.randn(n_states, n_features))
        # Initialize with exp to ensure positivity
        self.vars = nn.Parameter(torch.exp(torch.randn(n_states, n_features)))

        # Move to appropriate device
        self.to(device)

        # For tracking convergence
        self.log_likelihood_history = []

    def _log_gaussian(
        self, x: torch.Tensor, state: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute log probability of observations under Gaussian emissions.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features) or (n_features,).
            state (int, optional): Specific state index. If None, compute for all states.

        Returns:
            torch.Tensor: Log probabilities of shape (n_states,) or scalar.
        """
        if state is None:
            # For all states: (n_states, n_features)
            means = self.means
            vars = torch.clamp(self.vars, min=1e-6)  # Prevent numerical issues

            # Broadcasting for batched computation
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (1, n_features)

            # Compute for all states at once: (T, n_states, n_features)
            return -0.5 * (
                torch.log(2 * np.pi * vars) + ((x.unsqueeze(1) - means) ** 2 / vars)
            ).sum(dim=-1)
        else:
            # For specific state: (n_features,)
            means = self.means[state]
            vars = torch.clamp(self.vars[state], min=1e-6)

            return -0.5 * (torch.log(2 * np.pi * vars) + ((x - means) ** 2 / vars)).sum(
                dim=-1
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the forward algorithm to get log-likelihood.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (log_alpha, log_likelihood)
        """
        T = x.shape[0]
        log_alpha = torch.zeros(T, self.n_states).to(self.device)

        # Normalize parameters
        start_prob = torch.softmax(self.start_prob, dim=0)
        trans_prob = torch.softmax(self.trans_prob, dim=1)

        # Initialize first step with log probabilities
        log_alpha[0] = torch.log(start_prob) + self._log_gaussian(x[0])

        # Forward recursion
        for t in range(1, T):
            for j in range(self.n_states):
                # For each state j, combine previous alpha with transition to j
                log_alpha[t, j] = torch.logsumexp(
                    log_alpha[t - 1] + torch.log(trans_prob[:, j]), dim=0
                ) + self._log_gaussian(x[t], j)

        # Total log-likelihood is the log sum of the final alpha values
        log_likelihood = torch.logsumexp(log_alpha[-1], dim=0)
        return log_alpha, log_likelihood

    def backward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the backward algorithm.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features).

        Returns:
            torch.Tensor: Log beta values of shape (T, n_states).
        """
        T = x.shape[0]
        log_beta = torch.zeros(T, self.n_states).to(self.device)
        trans_prob = torch.softmax(self.trans_prob, dim=1)

        # Backward recursion (initialized with zeros for the last time step)
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = torch.logsumexp(
                    torch.log(trans_prob[i, :])
                    + self._log_gaussian(x[t + 1])
                    + log_beta[t + 1],
                    dim=0,
                )

        return log_beta

    def fit(
        self,
        x: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = True,
    ):
        """
        Train the HMM using the Baum-Welch algorithm.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features).
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
            verbose (bool): Whether to print progress information.

        Returns:
            self: The trained model.
        """
        start_time = time.time()
        T = x.shape[0]
        prev_log_likelihood = None

        for iter in range(max_iter):
            # E-step: Forward-backward algorithm
            log_alpha, log_likelihood = self.forward(x)
            log_beta = self.backward(x)

            self.log_likelihood_history.append(log_likelihood.item())

            # Normalize parameters
            trans_prob = torch.softmax(self.trans_prob, dim=1)

            # Compute posterior probabilities (gamma and xi)
            log_gamma = log_alpha + log_beta - log_likelihood

            # Initialize xi: p(q_t=i, q_{t+1}=j | O)
            log_xi = torch.zeros(T - 1, self.n_states, self.n_states).to(self.device)

            # Compute xi for all time steps and state pairs
            for t in range(T - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        log_xi[t, i, j] = (
                            log_alpha[t, i]
                            + torch.log(trans_prob[i, j])
                            + self._log_gaussian(x[t + 1], j)
                            + log_beta[t + 1, j]
                            - log_likelihood
                        )

            # M-step: Update parameters

            # Update initial probabilities (start_prob)
            self.start_prob.data = log_gamma[0]  # Will be softmaxed in forward pass

            # Update transition probabilities
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.trans_prob.data[i, j] = torch.logsumexp(
                        log_xi[:, i, j], dim=0
                    ) - torch.logsumexp(log_gamma[:-1, i], dim=0)

            # Update emission parameters (means and variances)
            for j in range(self.n_states):
                # Sum of gamma for state j
                gamma_sum = torch.exp(torch.logsumexp(log_gamma[:, j], dim=0))
                # Expand gamma to match observation dimensions
                gamma_exp = torch.exp(log_gamma[:, j]).unsqueeze(1)

                # Update state means
                self.means.data[j] = torch.sum(gamma_exp * x, dim=0) / gamma_sum

                # Update state variances
                self.vars.data[j] = (
                    torch.sum(gamma_exp * (x - self.means[j]) ** 2, dim=0) / gamma_sum
                )

                # Ensure numerical stability of variances
                self.vars.data[j] = torch.clamp(self.vars.data[j], min=1e-6)

            # Check convergence
            if prev_log_likelihood is not None:
                improvement = abs(log_likelihood - prev_log_likelihood)
                if verbose and iter % 10 == 0:
                    print(
                        f"Iteration {iter}/{max_iter}: log-likelihood = {log_likelihood.item():.4f}, "
                        + f"improvement = {improvement.item():.6f}"
                    )
                if improvement < tol:
                    if verbose:
                        print(
                            f"Converged at iteration {iter} with log-likelihood {log_likelihood.item():.4f}"
                        )
                    break

            prev_log_likelihood = log_likelihood

        elapsed_time = time.time() - start_time
        if verbose:
            print(
                f"Training completed in {elapsed_time:.2f} seconds, {iter + 1} iterations"
            )
            print(f"Final log-likelihood: {log_likelihood.item():.4f}")

        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely sequence of hidden states using the Viterbi algorithm.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features).

        Returns:
            torch.Tensor: Predicted states of shape (T,).
        """
        T = x.shape[0]
        delta = torch.zeros(T, self.n_states).to(self.device)
        psi = torch.zeros(T, self.n_states, dtype=torch.long).to(self.device)

        # Normalize parameters
        start_prob = torch.softmax(self.start_prob, dim=0)
        trans_prob = torch.softmax(self.trans_prob, dim=1)

        # Initialize first step
        delta[0] = torch.log(start_prob) + self._log_gaussian(x[0])

        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                # Find max probability path to state j at time t
                max_val, argmax_val = torch.max(
                    delta[t - 1] + torch.log(trans_prob[:, j]), dim=0
                )
                delta[t, j] = max_val + self._log_gaussian(x[t], j)
                psi[t, j] = argmax_val

        # Backtrack to find the most likely sequence
        states = torch.zeros(T, dtype=torch.long).to(self.device)
        states[-1] = torch.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def get_state_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the probability of being in each state for each time step.

        Args:
            x (torch.Tensor): Observations of shape (T, n_features).

        Returns:
            torch.Tensor: State probabilities of shape (T, n_states).
        """
        log_alpha, log_likelihood = self.forward(x)
        log_beta = self.backward(x)

        # Compute gamma (normalized state probabilities)
        log_gamma = log_alpha + log_beta - log_likelihood
        state_probs = torch.exp(log_gamma)

        return state_probs

    def plot_states(
        self, x: torch.Tensor, prices: torch.Tensor, save_path: Optional[str] = None
    ):
        """
        Plot the predicted states alongside price data.

        Args:
            x (torch.Tensor): Observations (returns) used for prediction.
            prices (torch.Tensor): Original price data.
            save_path (str, optional): Path to save the figure.
        """
        states = self.predict(x)
        state_probs = self.get_state_probabilities(x)

        # Move tensors to CPU for plotting
        states_np = states.cpu().numpy()
        prices_np = prices.cpu().numpy()
        state_probs_np = state_probs.cpu().numpy()

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot prices and states
        ax1.plot(prices_np, label="Price", color="blue", alpha=0.7)

        # Color regions by state
        for state in range(self.n_states):
            mask = states_np == state

            # Mark regions of each state with different colors
            if np.any(mask):
                # Generate ranges of consecutive True values
                ranges = np.where(np.diff(np.hstack(([False], mask, [False]))))[
                    0
                ].reshape(-1, 2)

                # Find state mean to determine color
                state_mean = self.means[state].item()
                color = "green" if state_mean > 0 else "red"
                label = f"State {state}: {'Bullish' if state_mean > 0 else 'Bearish'}"

                # Highlight each range
                for start, end in ranges:
                    ax1.axvspan(start, end - 1, alpha=0.2, color=color)

                # Add a single legend entry for this state
                ax1.fill_between([0], [0], [0], color=color, alpha=0.2, label=label)

        ax1.set_ylabel("Price")
        ax1.set_title("Cryptocurrency Price with HMM States")
        ax1.legend(loc="upper left")

        # Plot state probabilities
        for state in range(self.n_states):
            state_mean = self.means[state].item()
            color = "green" if state_mean > 0 else "red"
            label = f"P(State {state}): {'Bullish' if state_mean > 0 else 'Bearish'}"
            ax2.plot(state_probs_np[:, state], color=color, label=label, alpha=0.8)

        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State Probability")
        ax2.set_title("HMM State Probabilities Over Time")
        ax2.legend(loc="upper left")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        plt.show()

    def plot_log_likelihood(self, save_path: Optional[str] = None):
        """
        Plot the log-likelihood history during training.

        Args:
            save_path (str, optional): Path to save the figure.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.log_likelihood_history)
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("HMM Training Convergence")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Figure saved to {save_path}")

        plt.show()

    def save_model(self, path: str):
        """
        Save the model parameters to a file.

        Args:
            path (str): Path to save the model.
        """
        torch.save(
            {
                "n_states": self.n_states,
                "n_features": self.n_features,
                "start_prob": self.start_prob,
                "trans_prob": self.trans_prob,
                "means": self.means,
                "vars": self.vars,
                "log_likelihood_history": self.log_likelihood_history,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, device: str = "cuda"):
        """
        Load a saved model.

        Args:
            path (str): Path to the saved model.
            device (str): Computing device ('cuda' or 'cpu').

        Returns:
            GaussianHMM: Loaded model.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            n_states=checkpoint["n_states"],
            n_features=checkpoint["n_features"],
            device=device,
        )

        model.start_prob.data = checkpoint["start_prob"]
        model.trans_prob.data = checkpoint["trans_prob"]
        model.means.data = checkpoint["means"]
        model.vars.data = checkpoint["vars"]
        model.log_likelihood_history = checkpoint["log_likelihood_history"]

        print(f"Model loaded from {path}")
        return model


def load_and_prepare_data(data_path, feature_columns=["close"], target_column="close"):
    """
    Load and prepare data for the HMM model.

    Args:
        data_path (str): Path to the data file (.csv, .feather).
        feature_columns (list): Columns to use for feature calculation.
        target_column (str): Column to use for target calculation.

    Returns:
        tuple: (returns_tensor, prices_tensor) - prepared data tensors.
    """
    # Choose loading method based on file extension
    file_ext = os.path.splitext(data_path)[1].lower()

    if file_ext == ".csv":
        df = pd.read_csv(data_path)
    elif file_ext == ".feather":
        df = pd.read_feather(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .feather")

    # Convert data to numpy arrays
    prices = df[target_column].values

    # Calculate returns (simple percentage returns)
    returns = np.diff(prices) / prices[:-1]

    # Reshape returns for model input
    # For multiple features, shape would be (n_samples, n_features)
    returns = returns.reshape(-1, 1)

    # Convert to PyTorch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    returns_tensor = torch.from_numpy(returns).float().to(device)
    prices_tensor = (
        torch.from_numpy(prices[1:]).float().to(device)
    )  # Match returns length

    print(
        f"Data loaded and prepared: {returns.shape[0]} samples, {returns.shape[1]} features"
    )
    print(f"Using device: {device}")

    return returns_tensor, prices_tensor


def main():
    """
    Main function to run the HMM model.
    """
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # EDIT THIS SECTION WITH YOUR DATA PATHS AND PARAMETERS
    # ---------------------------------------------------------

    # Data loading parameters
    data_path = "path/to/your/crypto_data.csv"  # <-- REPLACE WITH YOUR DATA PATH

    # Model parameters
    n_states = 2  # Number of market regimes to identify
    n_features = 1  # Dimensionality of observations
    max_iter = 100  # Maximum EM iterations
    tol = 1e-4  # Convergence tolerance

    # ---------------------------------------------------------
    # NO NEED TO MODIFY BELOW THIS LINE
    # ---------------------------------------------------------

    # Load and prepare data
    returns_tensor, prices_tensor = load_and_prepare_data(data_path)

    # Create and train the model
    hmm = GaussianHMM(n_states, n_features, device=device)
    hmm.fit(returns_tensor, max_iter=max_iter, tol=tol, verbose=True)

    # Predict states
    states = hmm.predict(returns_tensor)

    # Print model parameters
    state_means = hmm.means.cpu().detach().numpy()
    state_vars = hmm.vars.cpu().detach().numpy()

    print("\nLearned Model Parameters:")
    for i in range(n_states):
        print(f"State {i}:")
        print(f"  Mean: {state_means[i][0]:.6f}")
        print(f"  Variance: {state_vars[i][0]:.6f}")
        print(
            f"  Interpretation: {'Bullish' if state_means[i][0] > 0 else 'Bearish'} Market Regime"
        )

    # Plot results
    hmm.plot_states(returns_tensor, prices_tensor)
    hmm.plot_log_likelihood()

    # Save the model if needed
    # hmm.save_model('crypto_hmm_model.pt')


if __name__ == "__main__":
    main()
