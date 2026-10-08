"""
Project-1: Theory of Probability and Mathematical Statistics
Topic: Lottery "k out of n" - combinatorics, conditional probability,
       independent events and frequency distributions.

Run locally:  streamlit run app.py
"""

import math
import random
import statistics
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# PART 1. MATH FUNCTIONS (the "brain" of the project)
# ----------------------------------------------------------------------

def total_combinations(n, k):
    """Number of ways to choose k numbers out of n (order does not matter).
    Formula: C(n, k) = n! / (k! * (n - k)!)   (Topic 1: combinatorics)
    """
    return math.comb(n, k)


def prob_exact_matches(n, k, m):
    """Probability to guess EXACTLY m numbers out of k.

    Favourable outcomes: choose m of the k winning numbers
    AND choose the other (k - m) numbers from the (n - k) losing ones.
        P(M = m) = C(k, m) * C(n - k, k - m) / C(n, k)
    """
    favourable = math.comb(k, m) * math.comb(n - k, k - m)
    return favourable / math.comb(n, k)


def prob_at_least(n, k, t):
    """P(M >= t): probability to guess t or more numbers (sum of events)."""
    return sum(prob_exact_matches(n, k, m) for m in range(t, k + 1))


def conditional_probability(n, k, t):
    """P(M >= t | M >= 1): chance of winning (t+ matches) IF we already
    know we guessed at least one number.   (Topic 3: conditional probability)
        P(A | B) = P(A and B) / P(B)
    Since {M >= t} is inside {M >= 1} (for t >= 1), P(A and B) = P(A).
    """
    return prob_at_least(n, k, t) / prob_at_least(n, k, 1)


def simulate(n, k, tickets, seed=None):
    """Play the lottery `tickets` times with one fixed ticket.
    Returns a list with the number of matches in each draw."""
    rng = random.Random(seed)
    my_ticket = set(rng.sample(range(1, n + 1), k))
    results = []
    for _ in range(tickets):
        draw = set(rng.sample(range(1, n + 1), k))
        results.append(len(my_ticket & draw))
    return results


def one_in(p):
    """Turns a probability into the form '1 in X'."""
    return "-" if p == 0 else f"1 in {round(1 / p):,}"


# ----------------------------------------------------------------------
# PART 2. WEBSITE (Streamlit)
# ----------------------------------------------------------------------

st.set_page_config(page_title="Lottery Probability Lab", page_icon="🎲", layout="wide")

st.title("🎲 Lottery Probability Lab")
st.caption(
    "Project-1 · Theory of Probability and Mathematical Statistics · "
    "Topics: combinatorics, conditional probability, independent events, "
    "frequency distributions"
)

# --- Sidebar: user parameters -----------------------------------------
st.sidebar.header("Lottery settings")
n = st.sidebar.number_input("n - total numbers in the drum", 5, 60, 49)
k = st.sidebar.number_input("k - numbers you must choose", 1, int(n), min(6, int(n)))
n, k = int(n), int(k)
st.sidebar.write(f"Lottery type: **{k} out of {n}**")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. Combinatorics", "2. Simulation", "3. Events & conditional probability", "4. Formulas"]
)

# --- Tab 1: combinatorics ---------------------------------------------
with tab1:
    st.subheader("How many possible tickets are there?")
    total = total_combinations(n, k)
    jackpot = 1 / total

    c1, c2 = st.columns(2)
    c1.metric("Total combinations C(n, k)", f"{total:,}")
    c2.metric("Jackpot probability", f"{jackpot:.10f}")
    st.write(f"Your chance to hit the jackpot is **{one_in(jackpot)}**.")

    rows = []
    for m in range(k + 1):
        ways = math.comb(k, m) * math.comb(n - k, k - m)
        p = prob_exact_matches(n, k, m)
        rows.append(
            {
                "Matches (m)": m,
                "Favourable combinations": f"{ways:,}",
                "Probability P(M = m)": round(p, 8),
                "Odds": one_in(p),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(range(k + 1), [prob_exact_matches(n, k, m) for m in range(k + 1)], color="#4C78A8")
    ax.set_xlabel("Number of matches m")
    ax.set_ylabel("Probability")
    ax.set_title("Theoretical distribution of matches")
    ax.set_xticks(range(k + 1))
    st.pyplot(fig)
    st.info("Check: all probabilities in the table add up to 1 "
            f"(sum = {sum(prob_exact_matches(n, k, m) for m in range(k + 1)):.6f}).")

# --- Tab 2: simulation ------------------------------------------------
with tab2:
    st.subheader("Simulation: does practice match theory?")
    tickets = st.slider("Number of simulated draws", 100, 100000, 10000, step=100)
    use_seed = st.checkbox("Fix random seed (same result every time)", value=True)

    if st.button("▶ Run simulation"):
        results = simulate(n, k, tickets, seed=42 if use_seed else None)
        counts = Counter(results)

        table = []
        for m in range(k + 1):
            table.append(
                {
                    "Matches (m)": m,
                    "Frequency": counts.get(m, 0),
                    "Relative frequency": round(counts.get(m, 0) / tickets, 6),
                    "Theoretical probability": round(prob_exact_matches(n, k, m), 6),
                }
            )
        st.dataframe(pd.DataFrame(table), hide_index=True)

        # Histogram: simulation vs theory
        fig, ax = plt.subplots(figsize=(8, 4))
        xs = list(range(k + 1))
        width = 0.4
        ax.bar([x - width / 2 for x in xs], [counts.get(m, 0) / tickets for m in xs],
               width, label="Simulation (relative frequency)", color="#F58518")
        ax.bar([x + width / 2 for x in xs], [prob_exact_matches(n, k, m) for m in xs],
               width, label="Theory (probability)", color="#4C78A8")
        ax.set_xlabel("Number of matches m")
        ax.set_ylabel("Share of draws")
        ax.set_xticks(xs)
        ax.legend()
        ax.set_title(f"{tickets:,} simulated draws")
        st.pyplot(fig)

        # Descriptive statistics (Topic 5: data description)
        st.markdown("**Descriptive statistics of the simulated data**")
        theoretical_mean = k * k / n
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Mean", f"{statistics.mean(results):.4f}")
        d2.metric("Theoretical mean k²/n", f"{theoretical_mean:.4f}")
        d3.metric("Median", f"{statistics.median(results)}")
        d4.metric("Standard deviation", f"{statistics.pstdev(results):.4f}")
        st.caption("The more draws you simulate, the closer the simulation gets to the theory "
                   "(this idea is called the law of large numbers).")
    else:
        st.write("Press the button to start the simulation.")

# --- Tab 3: events and conditional probability -------------------------
with tab3:
    st.subheader("Independent events and conditional probability")
    t = int(st.number_input("Minimum matches needed to win a prize (t)", 1, k, min(3, k)))
    p_win = prob_at_least(n, k, t)

    st.markdown("### Conditional probability")
    p_any = prob_at_least(n, k, 1)
    p_cond = conditional_probability(n, k, t)
    st.write(f"P(win) = P(M ≥ {t}) = **{p_win:.6f}**")
    st.write(f"P(at least 1 match) = **{p_any:.6f}**")
    st.write(f"If you already guessed at least one number, the chance to win rises to "
             f"P(M ≥ {t} | M ≥ 1) = **{p_cond:.6f}**")

    st.markdown("### Independent events")
    st.write("Every draw is independent: the drum has no memory.")
    draws = int(st.slider("Number of draws you play", 1, 1000, 52))
    p_two = p_win ** 2
    p_atleast_one = 1 - (1 - p_win) ** draws
    st.write(f"Probability to win in 2 draws in a row: P(A)·P(A) = **{p_two:.8f}**")
    st.write(f"Probability to win at least once in {draws} draws: "
             f"1 − (1 − p)^{draws} = **{p_atleast_one:.6f}**")

# --- Tab 4: formulas --------------------------------------------------
with tab4:
    st.subheader("Formulas used in this project")
    st.markdown("**1. Combinations** (order does not matter)")
    st.latex(r"C(n,k)=\frac{n!}{k!\,(n-k)!}")
    st.markdown("**2. Probability of exactly m matches** (favourable / total)")
    st.latex(r"P(M=m)=\frac{C(k,m)\cdot C(n-k,\,k-m)}{C(n,k)}")
    st.markdown("**3. Conditional probability**")
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}")
    st.markdown("**4. Independent events**")
    st.latex(r"P(A\cap B)=P(A)\cdot P(B)")
    st.markdown("**5. At least one success in N independent tries**")
    st.latex(r"P=1-(1-p)^N")
    st.markdown("**6. Relative frequency** (Topic 5)")
    st.latex(r"\text{relative frequency}=\frac{\text{times the value occurred}}{\text{total number of draws}}")
