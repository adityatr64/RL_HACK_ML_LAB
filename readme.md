# Hangman AI Agent

A **Hangman-playing agent** developed for an engineering project.
The goal is to solve Hangman puzzles with the **minimum number of incorrect guesses**.

This implementation uses a **hybrid strategy**, combining a probabilistic **HMM-based “Oracle”** with a **Dueling DQN (Deep Q-Network)** for strategic decision-making.

> ⚠️ **Note:** This is an experimental model.
> Current benchmarks on our test set show a **~11.8% success rate**, indicating that the model’s strategy and/or hyperparameters are still under development and require further tuning.

---

## 🏗️ Architecture

The agent’s decision-making process blends two complementary components:

### 🧩 1. HMM Oracle (`hmmAni.py`)

**Purpose:**
Provides a data-driven, probabilistic guess.

**How it works:**

* Pre-processes a large `corpus.txt` file to build a **statistical model** of the English language.
  (Includes word lists and letter frequencies by word length.)
* During gameplay, given a pattern (e.g., `_ _ T _`) and a set of guessed letters:

  * Filters its dictionary for all possible matches.
  * Returns a **probability distribution** for the remaining unguessed letters.

---

### ⚙️ 2. Dueling DQN Agent (`dqn_hmm_agent.py`)

**Purpose:**
Learns a long-term strategy (a “policy”) that maximizes the final game score.

**How it works:**
This is a **Reinforcement Learning** agent.

* **State:**
  A vector representing the game’s status, including:

  * Probability distribution from the HMM.
  * Mask of already-guessed letters.
  * Number of lives left.
* **Action:**
  The DQN outputs **Q-values** (expected future rewards) for guessing each of the 26 letters.

---

### 🎯 Action Selection

The agent’s final choice is a **weighted blend** of the HMM’s probabilities and the DQN’s Q-values:

[
\text{final_score} = (1 - \text{HMM_WEIGHT}) \times \text{DQN_Q_Values} + (\text{HMM_WEIGHT}) \times \text{HMM_Probabilities}
]

This allows the agent to balance:

* the HMM’s **immediate probabilistic “best guess”**, and
* the DQN’s **learned long-term strategy** for efficient puzzle solving.

---

## 🚀 How to Run

### 1. Install Dependencies

Ensure you have **PyTorch**, **NumPy**, and **tqdm** installed:

```bash
pip install torch numpy tqdm
```

---

### 2. Data Setup

The agent requires two word lists inside a `Data/` directory.

#### Create the directory:

```bash
mkdir Data
```

#### Add your word lists:

* `Data/corpus.txt` → *(For HMM Training)*
  A very large list of English words (one per line).
  Used to build the HMM’s statistical model.

* `Data/test.txt` → *(For DQN Training/Evaluation)*
  A separate list of **unseen words** not included in `corpus.txt`.
  Used for training and validating the DQN.

---

### 3. Train & Evaluate

Run the main training script:

```bash
python dqn_hmm_agent.py
```

This will:

1. Train the **HMM Oracle** on `corpus.txt`.
2. Train the **DQN Agent** using words from `test.txt`.
3. Evaluate performance and print results.
4. Save the trained model as `dqn_hmm_agent.pth`.

---

## 📂 File Structure

```
├── dqn_hmm_agent.py      # Dueling DQN, HangmanEnv, StateEncoder, and training loop
├── hmmAni.py             # HMMOracle for probabilistic letter guessing
├── Data/
│   ├── corpus.txt        # (User-provided) Word list for HMM training
│   └── test.txt          # (User-provided) Word list for DQN evaluation
└── dqn_hmm_agent.pth     # (Output) Saved model weights after training
```

---

## 🧩 Future Improvements

* Hyperparameter tuning for improved learning stability.
* Dynamic blending between DQN and HMM based on confidence levels.
* Expansion of training corpora for better generalization.
* Visualization of agent decision-making patterns.

---

## 🧠 Authors & Acknowledgments

Developed as part of an **engineering project** exploring the fusion of **probabilistic models** and **deep reinforcement learning** for natural-language games.

---

Would you like me to add a **"Results" section** (e.g., plots or metrics formatting) or a **"Citation / References" section** for academic submission style?
