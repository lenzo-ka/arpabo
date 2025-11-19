"""Model interpolation utilities for mixing n-gram orders."""

import math

from arpabo.arpa_io import get_ngram_prob
from arpabo.lm import OOV_PROBABILITY, ArpaBoLM
from arpabo.normalize import normalize_line


class InterpolatedModel:
    """
    Interpolated language model that mixes probabilities from multiple n-gram orders.

    Instead of backoff (try order N, then N-1, then N-2...),
    interpolation computes: P(w|h) = λ1*P1(w|h) + λ2*P2(w|h) + λ3*P3(w|h)

    This can provide smoother probability distributions and better handling
    of sparse contexts.

    Example:
        # Train models
        lm = ArpaBoLM(max_order=3)
        lm.read_corpus(open("corpus.txt"))
        models = lm.compute_multiple_orders([1, 2, 3])

        # Interpolate with custom weights
        interpolated = InterpolatedModel(
            models=models,
            weights={1: 0.1, 2: 0.3, 3: 0.6}
        )

        # Evaluate
        results = interpolated.perplexity(open("test.txt"))
    """

    def __init__(self, models: dict[int, ArpaBoLM], weights: dict[int, float]):
        """
        Initialize interpolated model.

        Args:
            models: Dictionary mapping order → trained ArpaBoLM model
            weights: Dictionary mapping order → interpolation weight
                     Must sum to 1.0

        Raises:
            ValueError: If weights don't sum to 1.0 or orders mismatch

        Example:
            models = {1: unigram, 2: bigram, 3: trigram}
            weights = {1: 0.1, 2: 0.3, 3: 0.6}
            interpolated = InterpolatedModel(models, weights)
        """
        self.models = models
        self.weights = weights

        # Validate
        if not models:
            raise ValueError("Must provide at least one model")

        if set(models.keys()) != set(weights.keys()):
            raise ValueError(f"Model orders {set(models.keys())} must match weight orders {set(weights.keys())}")

        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")

        # Get max order and vocab from models
        self.max_order = max(models.keys())
        self.vocab = set(list(models.values())[0].grams[0].keys())

        # Get normalization settings from first model
        first_model = list(models.values())[0]
        self.add_start = first_model.add_start
        self.unicode_norm = first_model.unicode_norm
        self.token_norm = first_model.token_norm
        self.case = first_model.case

    def get_interpolated_probability(self, word: str, context: tuple[str, ...]) -> float:
        """
        Get interpolated probability for word given context.

        P(word|context) = Σ λ_i * P_i(word|context_i)

        Where context_i is the appropriate context length for order i.

        Args:
            word: Word to get probability for
            context: Context words (tuple)

        Returns:
            Interpolated probability (raw probability 0-1, not log)
        """
        if word not in self.vocab:
            # OOV word - use very small probability
            return OOV_PROBABILITY

        interpolated_prob = 0.0

        for order, weight in self.weights.items():
            model = self.models[order]

            # Get appropriate context for this order
            context_len = order - 1
            if context_len == 0:
                # Unigram
                prob = model.probs[0].get(word, OOV_PROBABILITY)
            else:
                # Higher-order n-gram
                order_context = context[-context_len:] if len(context) >= context_len else context
                ngram_words = list(order_context) + [word]

                # Query probability
                prob_result = get_ngram_prob(model.probs[order - 1], ngram_words)

                if isinstance(prob_result, dict) or prob_result is None or prob_result <= 0:
                    # Not found, back off to lower order within this model
                    # For simplicity, use unigram
                    prob = model.probs[0].get(word, OOV_PROBABILITY)
                else:
                    prob = prob_result

            interpolated_prob += weight * prob

        return interpolated_prob

    def perplexity(self, test_corpus, oov_handling: str = "unk") -> dict:
        """
        Calculate perplexity of interpolated model on test corpus.

        Args:
            test_corpus: File handle to test data
            oov_handling: How to handle OOV words ("unk", "skip", "error")

        Returns:
            Dictionary with evaluation metrics (same format as ArpaBoLM.perplexity)
        """
        total_log_prob = 0.0
        total_words = 0
        oov_count = 0
        sentence_count = 0

        for line in test_corpus:
            if not line.strip():
                continue

            sentence_count += 1

            # Normalize the line
            words = normalize_line(
                line,
                unicode_norm=self.unicode_norm,
                case=self.case,
                token_norm=self.token_norm,
                add_markers=self.add_start,
            )

            for i, word in enumerate(words):
                # Check if word is in vocabulary
                if word not in self.vocab:
                    oov_count += 1

                    if oov_handling == "error":
                        raise ValueError(f"OOV word: '{word}'")
                    elif oov_handling == "skip":
                        continue

                # Get context
                context = tuple(words[max(0, i - self.max_order + 1) : i])

                # Get interpolated probability
                prob = self.get_interpolated_probability(word, context)

                # Convert to log10
                log_prob = math.log10(prob) if prob > 0 else math.log10(OOV_PROBABILITY)

                total_log_prob += log_prob
                total_words += 1

        if total_words == 0:
            raise ValueError("No words found in test corpus")

        # Calculate metrics
        avg_log_prob = total_log_prob / total_words
        perplexity = math.pow(10, -avg_log_prob)
        cross_entropy = -avg_log_prob / math.log10(2)

        return {
            "perplexity": perplexity,
            "cross_entropy": cross_entropy,
            "num_sentences": sentence_count,
            "num_words": total_words,
            "num_oov": oov_count,
            "oov_rate": oov_count / total_words if total_words > 0 else 0.0,
        }


def tune_interpolation_weights(models: dict[int, ArpaBoLM], dev_corpus, max_iterations: int = 10) -> dict[int, float]:
    """
    Automatically tune interpolation weights using EM algorithm on dev corpus.

    Uses expectation-maximization to find optimal weights that minimize
    perplexity on development data.

    Args:
        models: Dictionary mapping order → trained model
        dev_corpus: File handle to development corpus for tuning
        max_iterations: Maximum EM iterations (default: 10)

    Returns:
        Dictionary mapping order → optimal weight

    Example:
        models = lm.compute_multiple_orders([1, 2, 3])
        weights = tune_interpolation_weights(models, open("dev.txt"))
        interpolated = InterpolatedModel(models, weights)
    """
    orders = sorted(models.keys())
    n_orders = len(orders)

    # Initialize with uniform weights
    weights = dict.fromkeys(orders, 1.0 / n_orders)

    # Get normalization settings from first model
    first_model = list(models.values())[0]

    for _iteration in range(max_iterations):
        # E-step: Compute expected counts
        expected_counts = dict.fromkeys(orders, 0.0)
        total_prob = 0.0

        dev_corpus.seek(0)  # Reset file pointer

        for line in dev_corpus:
            if not line.strip():
                continue

            # Normalize
            words = normalize_line(
                line,
                unicode_norm=first_model.unicode_norm,
                case=first_model.case,
                token_norm=first_model.token_norm,
                add_markers=first_model.add_start,
            )

            for i, word in enumerate(words):
                # Skip OOV words
                if word not in first_model.probs[0]:
                    continue

                # Get probability from each model
                order_probs = {}
                for order in orders:
                    model = models[order]
                    context_len = order - 1
                    context = words[max(0, i - context_len) : i]

                    if context_len == 0:
                        # Unigram
                        prob = model.probs[0].get(word, OOV_PROBABILITY)
                    else:
                        # Higher order
                        ngram_words = context + [word]
                        prob_result = get_ngram_prob(model.probs[order - 1], ngram_words)

                        if isinstance(prob_result, dict) or prob_result is None or prob_result <= 0:
                            prob = model.probs[0].get(word, OOV_PROBABILITY)
                        else:
                            prob = prob_result

                    order_probs[order] = prob

                # Compute weighted probability
                weighted_prob = sum(weights[o] * order_probs[o] for o in orders)

                if weighted_prob > 0:
                    # Update expected counts
                    for order in orders:
                        expected_counts[order] += weights[order] * order_probs[order] / weighted_prob

                    total_prob += 1.0

        # M-step: Update weights
        if total_prob > 0:
            weights = {order: expected_counts[order] / total_prob for order in orders}

    return weights
