"""Debug and interactive tools for language models"""

from math import exp, log

from arpalm.arpa_io import get_ngram_prob


def debug_sentence(lm, sentence: str) -> None:
    """Interactive debug mode for stepping through a sentence with the language model.

    This function allows you to see how the model processes each word, including:
    - Word probabilities at different n-gram orders
    - Backoff weights (alphas) used
    - Which n-gram order was used for each prediction
    - Total probability of the sentence

    Args:
        lm: ArpaBoLM instance
        sentence: Sentence to debug
    """
    words = sentence.strip().split()
    if not words:
        print("Empty sentence!")
        return

    print(f"Debugging sentence: '{sentence}'")
    print(f"Words: {words}")
    print("=" * 60)

    total_log_prob = 0.0

    for i, word in enumerate(words):
        print(f"\nStep {i + 1}: Processing word '{word}'")
        print("-" * 40)

        # Try to find probability at each order, starting from highest
        found_prob = False
        used_order = -1

        for order in range(min(i, lm.max_order - 1), -1, -1):
            if order == 0:
                # Unigram probability
                if word in lm.probs[0]:
                    prob = lm.probs[0][word]
                    print(f"  Found unigram probability: P({word}) = {prob:.6f}")
                    used_order = 0
                    found_prob = True
                    break
                else:
                    print(f"  Word '{word}' not in unigram vocabulary")
            else:
                # Higher-order n-gram
                context_words = words[max(0, i - order) : i]
                ngram_words = context_words + [word]

                # Check if this n-gram exists
                prob = get_ngram_prob(lm.probs[order], ngram_words)
                if isinstance(prob, dict):
                    prob = None

                if prob is not None and prob > 0:
                    print(f"  Found {order + 1}-gram probability: P({word}|{' '.join(context_words)}) = {prob:.6f}")

                    # Show backoff weights for lower orders (only if discount_mass is set)
                    if order > 0 and lm.discount_mass is not None:
                        alpha = get_ngram_prob(lm.alphas[order - 1], context_words)
                        if isinstance(alpha, dict):
                            alpha = 1.0
                        print(f"  Backoff weight α({order}) = {alpha:.6f}")

                    used_order = order
                    found_prob = True
                    break
                else:
                    print(f"  {order + 1}-gram '{' '.join(ngram_words)}' not found")

        if found_prob:
            log_prob = log(prob) if prob > 0 else float("-inf")
            total_log_prob += log_prob
            print(f"  → Using order {used_order + 1}, log probability = {log_prob:.6f}")
        else:
            print(f"  → Word '{word}' not found in any n-gram order!")
            print("  → Using OOV (out-of-vocabulary) probability")
            # Use a small probability for unseen words
            oov_prob = 1.0 / (lm.sum_1 + 1)  # Laplace smoothing
            log_prob = log(oov_prob)
            total_log_prob += log_prob
            print(f"  → OOV log probability = {log_prob:.6f}")

    print("\n" + "=" * 60)
    print(f"Total sentence log probability: {total_log_prob:.6f}")
    print(f"Total sentence probability: {exp(total_log_prob):.2e}")
    print(f"Perplexity: {exp(-total_log_prob / len(words)):.2f}")

    entropy, perplexity = compute_perplexity(lm)
    print("\nModel statistics:")
    print(f"  Vocabulary size: {len(lm.probs[0])}")
    print(f"  Total word count: {lm.sum_1}")
    print(f"  Max order: {lm.max_order}")
    print(f"  Smoothing method: {lm.smoothing_method}")
    if lm.discount_mass is not None:
        print(f"  Discount mass: {lm.discount_mass:.3f}")
    print(f"  Unigram entropy: {entropy:.4f}")
    print(f"  Unigram perplexity: {perplexity:.2f}")


def interactive_debug(lm) -> None:
    """Start an interactive debug session where you can input sentences
    and step through them with the language model.

    Args:
        lm: ArpaBoLM instance
    """
    print("Language Model Interactive Debug Mode")
    print("=" * 50)
    print("Commands:")
    print("  <sentence>  - Debug a sentence")
    print("  /stats     - Show model statistics")
    print("  /vocab     - Show vocabulary")
    print("  /quit      - Exit debug mode")
    print()

    while True:
        try:
            command = input("debug> ").strip()

            if command.lower() in ["/quit", "/exit", "/q", "quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif command.lower() in ["/stats", "stats"]:
                entropy, perplexity = compute_perplexity(lm)
                print("\nModel Statistics:")
                print(f"  Vocabulary size: {len(lm.probs[0])}")
                print(f"  Total word count: {lm.sum_1}")
                print(f"  Max order: {lm.max_order}")
                print(f"  Smoothing method: {lm.smoothing_method}")
                if lm.discount_mass is not None:
                    print(f"  Discount mass: {lm.discount_mass:.3f}")
                print(f"  N-gram counts: {lm.counts}")
                print(f"  Unigram entropy: {entropy:.4f}")
                print(f"  Unigram perplexity: {perplexity:.2f}")
                print()
            elif command.lower() in ["/vocab", "vocab"]:
                print(f"\nVocabulary ({len(lm.probs[0])} words):")
                vocab_words = sorted(lm.probs[0].keys())
                for i, word in enumerate(vocab_words):
                    if i % 5 == 0:
                        print()
                    print(f"{word:12}", end="")
                print("\n")
            elif command:
                debug_sentence(lm, command)
            else:
                print("Please enter a sentence to debug or a command")

        except EOFError:
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def compute_perplexity(lm) -> tuple[float, float]:
    """Compute unigram entropy and perplexity on training data.

    Args:
        lm: ArpaBoLM instance

    Returns:
        (entropy, perplexity) tuple based on unigram distribution
    """
    if lm.sum_1 == 0:
        return 0.0, 1.0

    log_prob_sum = 0.0
    for word, count in lm.grams[0].items():
        if word in lm.probs[0] and lm.probs[0][word] > 0:
            log_prob_sum += count * log(lm.probs[0][word])

    entropy = -log_prob_sum / lm.sum_1
    perplexity = exp(entropy)
    return entropy, perplexity


def print_stats(lm) -> None:
    """Print model statistics to stdout.

    Args:
        lm: ArpaBoLM instance
    """
    print("Model Statistics:")
    print("=" * 50)
    print(f"Vocabulary size: {len(lm.probs[0])}")
    print(f"Total word count: {lm.sum_1}")
    print(f"Max order: {lm.max_order}")
    print(f"Smoothing method: {lm.smoothing_method}")
    if lm.discount_mass is not None:
        print(f"Discount mass: {lm.discount_mass:.3f}")
    print(f"N-gram counts: {lm.counts}")

    entropy, perplexity = compute_perplexity(lm)
    print(f"Unigram entropy: {entropy:.4f}")
    print(f"Unigram perplexity: {perplexity:.2f}")

    print(f"\nVocabulary ({len(lm.probs[0])} words):")
    vocab_words = sorted(lm.probs[0].keys())
    for i, word in enumerate(vocab_words):
        if i % 5 == 0:
            print()
        print(f"{word:12}", end="")
    print("\n")
