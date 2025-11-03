import collections
from typing import List, Set, Dict, DefaultDict, Tuple, Any
import numpy as np
import random
import os


class HMMOracle:
    def __init__(self, corpus_path: str):
        self.words_by_length: DefaultDict[int, List[str]] = collections.defaultdict(list)
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        self.general_freq_by_length: DefaultDict[int, Dict[str, float]] = collections.defaultdict(dict)
        
        self._process_corpus(corpus_path)
        
    def _process_corpus(self, corpus_path: str):

        print(f"Training HMM Oracle (preprocessing '{corpus_path}')...")
        
        try:
            with open(corpus_path, 'r') as f:
                for line in f:
                    word = line.strip().upper()
                    if word and word.isalpha():
                        self.words_by_length[len(word)].append(word)
        except FileNotFoundError:
            print(f"Error: Corpus file not found at {corpus_path}")
            raise

        for length, words in self.words_by_length.items():
            counts: DefaultDict[str, int] = collections.defaultdict(int)
            total_letters = 0
            for word in words:
                for char in word:
                    counts[char] += 1
                    total_letters += 1
            
            if total_letters > 0:
                self.general_freq_by_length[length] = {
                    char: counts[char] / total_letters for char in self.alphabet
                }

        print(f"Oracle training complete. Loaded {sum(len(v) for v in self.words_by_length.values())} words.")

    def _is_match(self, word: str, masked_word: str, known_correct: Set[str], wrong_guesses: Set[str]) -> bool:

        for i, char in enumerate(word):
            mask_char = masked_word[i]
            
            if mask_char == '_':
                if char in wrong_guesses:
                    return False
            else:
                if char != mask_char:
                    return False
        
        known_correct_counts = collections.Counter(c for c in masked_word if c != '_')
        word_counts = collections.Counter(word)
        
        for letter, revealed_count in known_correct_counts.items():
            if word_counts[letter] < revealed_count:
                return False
                
        return True

    def get_letter_probabilities(self, masked_word: str, guessed_letters: Set[str]) -> Dict[str, float]:

        word_len = len(masked_word)
        probabilities: Dict[str, float] = {}
        
        possible_words = self.words_by_length.get(word_len, [])
        known_correct = set(masked_word) - {'_'}
        wrong_guesses = guessed_letters - known_correct
        
        valid_candidates: List[str] = []
        if possible_words:
            for word in possible_words:
                if self._is_match(word, masked_word, known_correct, wrong_guesses):
                    valid_candidates.append(word)
                
        if valid_candidates:
            letter_counts: DefaultDict[str, int] = collections.defaultdict(int)
            total_blank_letters = 0
            
            for word in valid_candidates:
                for i, char in enumerate(word):
                    if masked_word[i] == '_':
                        if char not in known_correct:
                            letter_counts[char] += 1
                            total_blank_letters += 1

                        elif collections.Counter(word)[char] > collections.Counter(masked_word)[char]:
                             letter_counts[char] += 1
                             total_blank_letters += 1

            
            if total_blank_letters > 0:
                for char in self.alphabet:
                    if char in guessed_letters:
                        probabilities[char] = 0.0
                    else:
                        probabilities[char] = letter_counts[char] / total_blank_letters
                return probabilities

        general_probs = self.general_freq_by_length.get(word_len)

        if not general_probs:
            return {char: 0.0 for char in self.alphabet}
        
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
    print("-" * 30)
    print("Running Formal Evaluation on corpus.txt / test.txt...")
    CORPUS_PATH = "./Data/corpus.txt"
    TEST_PATH = "./Data/test.txt"

    try:
        with open(TEST_PATH, 'r') as f:
            test_words = [line.strip().upper() for line in f if line.strip().isalpha()]
        
        if not test_words:
            print(f"No words found in {TEST_PATH}. Skipping formal evaluation.")
        else:
            oracle = HMMOracle(CORPUS_PATH)
            print(f"\nEvaluating oracle on {len(test_words)} test words...")
            metrics, total_games = evaluate_oracle(oracle, test_words, num_reveal=2, num_wrong=3)
            
            print("\n" + ("-" * 20))
            print(f"Evaluation Results ({total_games} simulated games):")
            print(f"  Top-1 Accuracy: {metrics['top_1_acc']:.2f}%")
            print(f"  Top-3 Accuracy: {metrics['top_3_acc']:.2f}%")
            print(f"  Top-5 Accuracy: {metrics['top_5_acc']:.2f}%")
            print("-" * 20)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Skipping formal evaluation. Please make sure 'corpus.txt' and 'test.txt' are present.")