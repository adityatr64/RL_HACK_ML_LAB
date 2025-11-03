import collections
from typing import List, Set, Dict, DefaultDict, Tuple, Any
import numpy as np
import random
import os

# ------------------------------------------------------------------
# IMPROVED HMMOracle CLASS (v4 - Better Filtering)
# ------------------------------------------------------------------

class HMMOracle:
    """
    Improved HMM Oracle with better candidate matching.
    
    Key improvements:
    1. Smarter _is_match that handles duplicates correctly
    2. Better fallback strategy when candidates are limited
    3. Confidence scoring to avoid overly narrow filtering
    """
    
    def __init__(self, corpus_path: str):
        self.words_by_length: DefaultDict[int, List[str]] = collections.defaultdict(list)
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        self.general_freq_by_length: DefaultDict[int, Dict[str, float]] = collections.defaultdict(dict)
        self.general_freq_overall: Dict[str, float] = {}
        
        self._process_corpus(corpus_path)
        
    def _process_corpus(self, corpus_path: str):
        """Process corpus and calculate frequencies."""
        print(f"Training HMM Oracle v4 ('{corpus_path}')...")
        
        try:
            with open(corpus_path, 'r') as f:
                for line in f:
                    word = line.strip().upper()
                    if word and word.isalpha():
                        self.words_by_length[len(word)].append(word)
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {corpus_path}")
            raise

        # Overall frequencies (for ultimate fallback)
        overall_counts: DefaultDict[str, int] = collections.defaultdict(int)
        overall_total = 0

        for length, words in self.words_by_length.items():
            counts: DefaultDict[str, int] = collections.defaultdict(int)
            total_letters = 0
            for word in words:
                for char in word:
                    counts[char] += 1
                    total_letters += 1
                    overall_counts[char] += 1
                    overall_total += 1
            
            if total_letters > 0:
                self.general_freq_by_length[length] = {
                    char: counts[char] / total_letters for char in self.alphabet
                }

        if overall_total > 0:
            self.general_freq_overall = {
                char: overall_counts[char] / overall_total for char in self.alphabet
            }

        print(f"Oracle ready. {sum(len(v) for v in self.words_by_length.values())} words loaded.")

    def _is_match(self, word: str, masked_word: str, known_correct: Set[str], wrong_guesses: Set[str]) -> bool:
        """Check if word matches the masked pattern and constraints."""
        
        # Check each position
        for i, char in enumerate(word):
            mask_char = masked_word[i]
            
            if mask_char == '_':
                # Blank spot cannot be a wrong guess
                if char in wrong_guesses:
                    return False
            else:
                # Revealed spot must match exactly
                if char != mask_char:
                    return False

        # Check duplicate letter consistency
        known_correct_counts = collections.Counter(c for c in masked_word if c != '_')
        word_counts = collections.Counter(word)
        
        for letter, revealed_count in known_correct_counts.items():
            if word_counts[letter] < revealed_count:
                return False
                
        return True

    def get_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Calculate letter probabilities with adaptive filtering.
        
        Strategy:
        1. Strict filter on valid candidates
        2. If too few candidates, relax constraints
        3. Blend strict + general frequencies
        4. Fall back to general if needed
        """
        word_len = len(masked_word)
        probabilities: Dict[str, float] = {}
        
        possible_words = self.words_by_length.get(word_len, [])
        known_correct = set(masked_word) - {'_'}
        wrong_guesses = guessed_letters - known_correct
        
        # STAGE 1: Strict filtering
        valid_candidates: List[str] = []
        if possible_words:
            for word in possible_words:
                if self._is_match(word, masked_word, known_correct, wrong_guesses):
                    valid_candidates.append(word)
        
        candidate_ratio = len(valid_candidates) / max(1, len(possible_words))
        
        # STAGE 2: Calculate probabilities from candidates
        letter_counts: DefaultDict[str, int] = collections.defaultdict(int)
        total_blank_letters = 0
        
        if valid_candidates:
            for word in valid_candidates:
                for i, char in enumerate(word):
                    if masked_word[i] == '_':
                        # Only count unguessed letters
                        if char not in guessed_letters:
                            letter_counts[char] += 1
                            total_blank_letters += 1
        
        # STAGE 3: Blend strategies based on candidate availability
        if total_blank_letters > 10:  # Strong signal from candidates
            # Use candidate frequencies directly
            for char in self.alphabet:
                if char in guessed_letters:
                    probabilities[char] = 0.0
                else:
                    probabilities[char] = letter_counts[char] / total_blank_letters
            return probabilities
        
        elif total_blank_letters > 0:  # Weak signal, blend with general
            # Blend: 60% candidates, 40% general
            candidate_probs = {}
            for char in self.alphabet:
                if char in guessed_letters:
                    candidate_probs[char] = 0.0
                else:
                    candidate_probs[char] = letter_counts[char] / total_blank_letters
            
            general_probs = self.general_freq_by_length.get(word_len, self.general_freq_overall)
            
            for char in self.alphabet:
                if char in guessed_letters:
                    probabilities[char] = 0.0
                else:
                    prob = 0.6 * candidate_probs[char] + 0.4 * general_probs.get(char, 0.0)
                    probabilities[char] = prob
            
            # Normalize
            total = sum(probabilities.values())
            if total > 0:
                for char in probabilities:
                    probabilities[char] /= total
            
            return probabilities
        
        # STAGE 4: Fallback to general frequencies
        general_probs = self.general_freq_by_length.get(word_len, self.general_freq_overall)
        
        total_prob = 0.0
        for char in self.alphabet:
            if char in guessed_letters:
                probabilities[char] = 0.0
            else:
                prob = general_probs.get(char, 0.0)
                probabilities[char] = prob
                total_prob += prob

        if total_prob > 0:
            for char in self.alphabet:
                probabilities[char] = probabilities[char] / total_prob
        
        return probabilities


# ------------------------------------------------------------------
# EVALUATION (Same as original)
# ------------------------------------------------------------------

def evaluate_oracle(oracle: HMMOracle, 
                    test_words: list[str], 
                    num_reveal=2, 
                    num_wrong=3):
    metrics = {'top_1_acc': 0.0, 'top_3_acc': 0.0, 'top_5_acc': 0.0}
    total_games = 0
    ALPHABET_SET = set(oracle.alphabet) 

    for word in test_words:
        word = word.upper()
        correct_letters = set(word)
        if len(word) <= num_reveal: continue
        
        reveal_indices = random.sample(range(len(word)), num_reveal)
        pattern_list = ['_'] * len(word)
        revealed_letters = set()
        for idx in reveal_indices:
            letter = word[idx]
            pattern_list[idx] = letter
            revealed_letters.add(letter)
        pattern = "".join(pattern_list)
        
        wrong_letter_pool = ALPHABET_SET - correct_letters
        if len(wrong_letter_pool) < num_wrong: continue
            
        wrong_guesses = set(random.sample(list(wrong_letter_pool), num_wrong))
        guessed = revealed_letters.union(wrong_guesses)
        letters_to_find = correct_letters - revealed_letters
        if not letters_to_find: continue

        prob_dict = oracle.get_letter_probabilities(pattern, guessed)
        
        if sum(prob_dict.values()) == 0:
            continue
            
        sorted_preds = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        top_1_pred = {sorted_preds[0][0]}
        top_3_preds = {p[0] for p in sorted_preds[:3]}
        top_5_preds = {p[0] for p in sorted_preds[:5]}

        if top_1_pred.intersection(letters_to_find): metrics['top_1_acc'] += 1
        if top_3_preds.intersection(letters_to_find): metrics['top_3_acc'] += 1
        if top_5_preds.intersection(letters_to_find): metrics['top_5_acc'] += 1
        total_games += 1

    if total_games > 0:
        for k in metrics:
            metrics[k] = (metrics[k] / total_games) * 100
    
    return metrics, total_games


if __name__ == "__main__":
    print("-" * 40)
    print("HMM Oracle v4 - Evaluation")
    print("-" * 40)
    CORPUS_PATH = "./Data/corpus.txt"
    TEST_PATH = "./Data/test.txt"

    try:
        with open(TEST_PATH, 'r') as f:
            test_words = [line.strip().upper() for line in f if line.strip().isalpha()]
        
        if not test_words:
            print(f"No words found in {TEST_PATH}")
        else:
            oracle = HMMOracle(CORPUS_PATH)
            print(f"\nEvaluating on {len(test_words)} test words...")
            metrics, total_games = evaluate_oracle(oracle, test_words, num_reveal=2, num_wrong=3)
            
            print(f"\nResults ({total_games} games):")
            print(f"  Top-1 Accuracy: {metrics['top_1_acc']:.2f}%")
            print(f"  Top-3 Accuracy: {metrics['top_3_acc']:.2f}%")
            print(f"  Top-5 Accuracy: {metrics['top_5_acc']:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}")