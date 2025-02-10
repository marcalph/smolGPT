from collections import defaultdict, Counter
import math

# Example corpus: a list of sentences, each sentence is a list of words.
corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "rug"],
    ["the", "cat", "chased", "the", "mouse"],
]

# Step 1: Count bigrams and unigrams.
bigram_counts = defaultdict(Counter)
unigram_counts = Counter()

for sentence in corpus:
    # Add a start-of-sentence token if needed:
    previous_word = "<s>"
    unigram_counts[previous_word] += 1
    for word in sentence:
        unigram_counts[word] += 1
        bigram_counts[previous_word][word] += 1
        previous_word = word
    # Optionally, mark the end of the sentence:
    bigram_counts[previous_word]["</s>"] += 1
    unigram_counts["</s>"] += 1


# Step 2: Calculate bigram probabilities.
def bigram_prob(prev, word):
    if unigram_counts[prev] == 0:
        return 0.0
    return bigram_counts[prev][word] / unigram_counts[prev]


# Example: calculate probability of "sat" given "cat"
print("P(sat | cat) =", bigram_prob("cat", "sat"))


# Step 3: Calculate the probability of a sentence using the bigram model.
def sentence_probability(sentence):
    # Optionally, add start and end tokens
    words = ["<s>"] + sentence + ["</s>"]
    prob = 1.0
    for i in range(1, len(words)):
        prob *= bigram_prob(words[i - 1], words[i])
    return prob


sentence = ["the", "cat", "sat", "on", "the", "mat"]
print("Sentence probability =", sentence_probability(sentence))


# Optionally, you can calculate log probability to avoid underflow:
def sentence_log_probability(sentence):
    words = ["<s>"] + sentence + ["</s>"]
    log_prob = 0.0
    for i in range(1, len(words)):
        p = bigram_prob(words[i - 1], words[i])
        if p > 0:
            log_prob += math.log(p)
        else:
            log_prob += float("-inf")
    return log_prob


print("Sentence log probability =", sentence_log_probability(sentence))
