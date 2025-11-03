import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import collections
from typing import List, Set, Dict, Tuple, DefaultDict, Optional
import random
from collections import deque
from tqdm import tqdm, trange
import pickle
import os

# Import improved HMM Oracle
import sys
sys.path.append('.')
from hmmAni_v4_improved import HMMOracle

# ------------------------------------------------------------------
# 1. IMPROVED HANGMAN ENVIRONMENT
# ------------------------------------------------------------------

class HangmanEnv:
    """Improved Hangman environment with better reward shaping."""
    
    def __init__(self, word: str, max_lives: int = 6):
        self.word = word.upper()
        self.max_lives = max_lives
        self.lives = max_lives
        self.guessed_letters: Set[str] = set()
        self.correct_guesses: Set[str] = set()
        self.wrong_guesses: Set[str] = set()
        self.repeated_guesses: int = 0
        self.masked_word = ['_'] * len(word)
        self.step_count = 0
        self.total_unique_letters = len(set(word.upper()))
        
    def get_masked_word(self) -> str:
        return "".join(self.masked_word)
    
    def is_won(self) -> bool:
        return '_' not in self.masked_word
    
    def is_lost(self) -> bool:
        return self.lives <= 0
    
    def is_done(self) -> bool:
        return self.is_won() or self.is_lost()
    
    def get_progress(self) -> float:
        """Percentage of word revealed."""
        revealed = len(self.correct_guesses)
        return revealed / max(1, self.total_unique_letters)
    
    def guess_letter(self, letter: str) -> Tuple[bool, float, bool]:
        """
        Improved reward structure:
        - Large reward for winning
        - Progress reward (revealing letters)
        - Penalty for mistakes
        - Penalty for repetition
        """
        letter = letter.upper()
        self.step_count += 1
        
        # Penalize repeated guesses
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return False, -10.0, self.is_done()
        
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            self.correct_guesses.add(letter)
            for i, char in enumerate(self.word):
                if char == letter:
                    self.masked_word[i] = letter
            reward = 5.0  # Reward for progress
        else:
            self.wrong_guesses.add(letter)
            self.lives -= 1
            reward = -20.0  # Larger penalty for wrong guess
        
        done = self.is_done()
        if done:
            if self.is_won():
                # Large reward for winning, scaled by remaining lives
                reward += 200.0 + (self.lives * 10.0)
            else:
                # Larger penalty for losing
                reward -= 100.0
        
        return letter in self.word, reward, done
    
    def reset(self, word: str):
        self.word = word.upper()
        self.lives = self.max_lives
        self.guessed_letters = set()
        self.correct_guesses = set()
        self.wrong_guesses = set()
        self.repeated_guesses = 0
        self.masked_word = ['_'] * len(word)
        self.step_count = 0
        self.total_unique_letters = len(set(word.upper()))
    
    def get_stats(self) -> Dict:
        return {
            'won': self.is_won(),
            'wrong_guesses': len(self.wrong_guesses),
            'repeated_guesses': self.repeated_guesses,
            'total_guesses': len(self.guessed_letters),
        }


# ------------------------------------------------------------------
# 2. IMPROVED STATE ENCODER
# ------------------------------------------------------------------

class StateEncoder:
    """Enhanced state representation."""
    
    def __init__(self, alphabet_size: int = 26, device: str = 'cuda'):
        self.alphabet_size = alphabet_size
        self.device = device
        # guessed_mask(26) + prob_vector(26) + wrong_mask(26) + lives(1) + word_len(1) + progress(1) = 81
        self.state_size = alphabet_size * 3 + 3
    
    def encode(self, env: HangmanEnv, hmm_probs: Dict[str, float]) -> torch.Tensor:
        """
        Encode state:
        [guessed_mask, hmm_probs, wrong_mask, lives_norm, word_len_norm, progress]
        """
        guessed_mask = torch.zeros(self.alphabet_size, dtype=torch.float32, device=self.device)
        prob_vector = torch.zeros(self.alphabet_size, dtype=torch.float32, device=self.device)
        wrong_mask = torch.zeros(self.alphabet_size, dtype=torch.float32, device=self.device)
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for i, letter in enumerate(alphabet):
            if letter in env.guessed_letters:
                guessed_mask[i] = 1.0
            if letter in env.wrong_guesses:
                wrong_mask[i] = 1.0
            prob_vector[i] = hmm_probs.get(letter, 0.0)
        
        lives_norm = torch.tensor([env.lives / env.max_lives], dtype=torch.float32, device=self.device)
        word_len_norm = torch.tensor([len(env.word) / 20.0], dtype=torch.float32, device=self.device)
        progress = torch.tensor([env.get_progress()], dtype=torch.float32, device=self.device)
        
        state = torch.cat([guessed_mask, prob_vector, wrong_mask, lives_norm, word_len_norm, progress])
        return state


# ------------------------------------------------------------------
# 3. IMPROVED DUELING DQN
# ------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """Larger and deeper network for better learning."""
    
    def __init__(self, state_size: int, action_size: int = 26, hidden_dim: int = 512):
        super(DuelingDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size),
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state):
        feat = self.feature(state)
        advantage = self.advantage(feat)
        value = self.value(feat)
        
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


# ------------------------------------------------------------------
# 4. IMPROVED DQN + HMM AGENT
# ------------------------------------------------------------------

class DQNHMMAgent:
    """Enhanced DQN+HMM agent with better hyperparameters."""
    
    def __init__(self, 
                 hmm_oracle: HMMOracle,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 5e-4,
                 hidden_dim: int = 512,
                 hmm_weight: float = 0.5):
        
        self.device = torch.device(device)
        self.hmm_oracle = hmm_oracle
        self.hmm_weight = hmm_weight
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.action_size = 26
        
        # Networks
        self.state_encoder = StateEncoder(alphabet_size=26, device=str(self.device))
        self.q_network = DuelingDQN(
            state_size=self.state_encoder.state_size,
            action_size=self.action_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.target_network = DuelingDQN(
            state_size=self.state_encoder.state_size,
            action_size=self.action_size,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # DQN hyperparameters
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.05
        self.target_update_freq = 50
        self.update_counter = 0
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=50000)  # Larger buffer
        self.min_buffer_size = 1000
        
        # Statistics
        self.episode_rewards = []
        self.episode_losses = []
    
    def get_valid_actions(self, env: HangmanEnv) -> torch.Tensor:
        mask = torch.ones(self.action_size, dtype=torch.bool, device=self.device)
        for letter in env.guessed_letters:
            idx = ord(letter) - ord('A')
            mask[idx] = False
        return mask
    
    def select_action(self, 
                     state: torch.Tensor, 
                     valid_mask: torch.Tensor,
                     hmm_probs: Dict[str, float],
                     training: bool = True) -> int:
        """Select action with improved exploration."""
        
        if training and np.random.random() < self.epsilon:
            valid_indices = torch.where(valid_mask)[0].cpu().numpy()
            return np.random.choice(valid_indices)
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
        
        q_values = q_values.squeeze(0).clone()
        
        # Create HMM prior
        hmm_prior = torch.zeros(self.action_size, device=self.device)
        for i, letter in enumerate(self.alphabet):
            hmm_prior[i] = hmm_probs.get(letter, 0.0)
        
        # Mask invalid actions
        q_values[~valid_mask] = -1e9
        hmm_prior[~valid_mask] = -1e9
        
        # Softmax for probabilities
        q_probs = torch.softmax(q_values, dim=0)
        hmm_probs_norm = torch.softmax(hmm_prior, dim=0)
        
        # Blend with higher HMM weight
        final_scores = (1 - self.hmm_weight) * q_probs + self.hmm_weight * hmm_probs_norm
        final_scores[~valid_mask] = -1e9
        
        action = torch.argmax(final_scores).item()
        return action
    
    def store_transition(self, state: torch.Tensor, action: int, reward: float, 
                        next_state: torch.Tensor, done: bool):
        self.replay_buffer.append({
            'state': state.cpu().detach(),
            'action': action,
            'reward': reward,
            'next_state': next_state.cpu().detach(),
            'done': done,
        })
    
    def play_episode(self, env: HangmanEnv, training: bool = True) -> Tuple[bool, Dict, float]:
        env.reset(env.word)
        episode_reward = 0.0
        
        while not env.is_done():
            hmm_probs = self.hmm_oracle.get_letter_probabilities(
                env.get_masked_word(),
                env.guessed_letters
            )
            
            state = self.state_encoder.encode(env, hmm_probs)
            valid_mask = self.get_valid_actions(env)
            
            action = self.select_action(state, valid_mask, hmm_probs, training=training)
            letter = self.alphabet[action]
            is_correct, reward, done = env.guess_letter(letter)
            episode_reward += reward
            
            if done:
                next_state = torch.zeros_like(state)
            else:
                next_hmm = self.hmm_oracle.get_letter_probabilities(
                    env.get_masked_word(),
                    env.guessed_letters
                )
                next_state = self.state_encoder.encode(env, next_hmm)
            
            if training:
                self.store_transition(state, action, reward, next_state, done)
        
        return env.is_won(), env.get_stats(), episode_reward
    
    def training_step(self, batch_size: int = 64) -> float:
        if len(self.replay_buffer) < self.min_buffer_size:
            return 0.0
        
        actual_batch_size = min(batch_size, len(self.replay_buffer))
        batch = random.sample(list(self.replay_buffer), actual_batch_size)
        
        states = torch.stack([t['state'] for t in batch]).to(self.device)
        actions = torch.tensor([t['action'] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t['reward'] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t['next_state'] for t in batch]).to(self.device)
        dones = torch.tensor([t['done'] for t in batch], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_max, _ = next_q_values.max(dim=1)
            q_targets = rewards + (1 - dones) * self.gamma * next_q_max
        
        q_values = self.q_network(states)
        q_preds = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        loss = nn.SmoothL1Loss()(q_preds, q_targets)  # Huber loss is more robust
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, 
              words: List[str], 
              num_episodes: int = 10000,
              training_steps_per_episode: int = 8,
              batch_size: int = 64):
        
        print(f"\n{'='*70}")
        print("IMPROVED DQN+HMM AGENT - TRAINING")
        print(f"{'='*70}")
        print(f"Total Episodes:              {num_episodes}")
        print(f"Training Steps/Episode:      {training_steps_per_episode}")
        print(f"Batch Size:                  {batch_size}")
        print(f"HMM Weight:                  {self.hmm_weight}")
        print(f"Device:                      {self.device}")
        print(f"{'='*70}\n")
        
        episode_pbar = trange(num_episodes, desc="Episodes", position=0)
        
        for episode in episode_pbar:
            word = random.choice(words)
            env = HangmanEnv(word)
            
            won, stats, ep_reward = self.play_episode(env, training=True)
            self.episode_rewards.append(ep_reward)
            
            step_losses = []
            for step in range(training_steps_per_episode):
                loss = self.training_step(batch_size)
                if loss > 0:
                    step_losses.append(loss)
            
            avg_loss = np.mean(step_losses) if step_losses else 0.0
            self.episode_losses.append(avg_loss)
            
            recent_wins = sum(1 for r in self.episode_rewards[-100:] if r > 100)
            recent_episodes = min(100, len(self.episode_rewards))
            win_rate = (recent_wins / recent_episodes * 100) if recent_episodes > 0 else 0
            
            episode_pbar.set_postfix({
                'ε': f'{self.epsilon:.3f}',
                'win%': f'{win_rate:.1f}',
                'loss': f'{avg_loss:.4f}',
                'reward': f'{ep_reward:.1f}'
            })
    
    def evaluate(self, words: List[str], num_eval_episodes: int = 500) -> Dict:
        total_wins = 0
        total_wrong = 0
        total_repeated = 0
        total_reward = 0
        
        eval_pbar = tqdm(range(num_eval_episodes), desc="Evaluation", position=0)
        
        for _ in eval_pbar:
            word = random.choice(words)
            env = HangmanEnv(word)
            won, stats, ep_reward = self.play_episode(env, training=False)
            
            if won:
                total_wins += 1
            total_wrong += stats['wrong_guesses']
            total_repeated += stats['repeated_guesses']
            total_reward += ep_reward
        
        competition_score = (total_wins * 2000) - (total_wrong * 5) - (total_repeated * 2)
        
        metrics = {
            'win_rate': (total_wins / num_eval_episodes) * 100,
            'avg_wrong': total_wrong / num_eval_episodes,
            'avg_repeated': total_repeated / num_eval_episodes,
            'total_wins': total_wins,
            'competition_score': competition_score,
            'per_game_score': competition_score / num_eval_episodes,
            'avg_episode_reward': total_reward / num_eval_episodes,
        }
        
        return metrics
    
    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)
        print(f"\nAgent saved to {path}")
    
    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())
        print(f"Agent loaded from {path}")


# ------------------------------------------------------------------
# 5. MAIN TRAINING SCRIPT
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("IMPROVED DQN + HMM HANGMAN AGENT")
    print("="*70)
    
    CORPUS_PATH = "./Data/corpus.txt"
    TEST_PATH = "./Data/test.txt"
    
    try:
        print("\n[1] Loading HMM Oracle v4...")
        hmm_oracle = HMMOracle(CORPUS_PATH)
        
        print("\n[2] Loading test words...")
        with open(TEST_PATH, 'r') as f:
            test_words = [line.strip().upper() for line in f if line.strip().isalpha()]
        
        print(f"   Test words (unseen): {len(test_words)}")
        
        random.shuffle(test_words)
        train_words = test_words[:int(0.8 * len(test_words))]
        val_words = test_words[int(0.8 * len(test_words)):]
        
        print(f"   Training: {len(train_words)} words, Validation: {len(val_words)} words")
        
        print("\n[3] Initializing improved agent...")
        agent = DQNHMMAgent(
            hmm_oracle=hmm_oracle,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            learning_rate=5e-4,
            hidden_dim=512,
            hmm_weight=0.5
        )
        
        print("\n[4] Training...")
        agent.train(
            words=train_words,
            num_episodes=10000,
            training_steps_per_episode=8,
            batch_size=64
        )
        
        print("\n[5] Evaluating...")
        metrics = agent.evaluate(val_words, num_eval_episodes=500)
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Win Rate:              {metrics['win_rate']:.2f}%")
        print(f"Total Wins:            {metrics['total_wins']}/500")
        print(f"Avg Wrong Guesses:     {metrics['avg_wrong']:.2f}")
        print(f"Avg Repeated Guesses:  {metrics['avg_repeated']:.2f}")
        print(f"Avg Episode Reward:    {metrics['avg_episode_reward']:.2f}")
        print(f"Competition Score:     {metrics['competition_score']:.0f}")
        print(f"Per-Game Score:        {metrics['per_game_score']:.2f}")
        print("="*70)
        
        agent.save("dqn_hmm_agent_v2.pth")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")