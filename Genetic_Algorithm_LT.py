import numpy as np
import pandas as pd
import streamlit as st

POPULATION = 300
CHROMOSOME_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 40
MAX_FITNESS = 80  


def fitness_fn(bitstring: np.ndarray) -> float:
    ones = int(np.sum(bitstring))
    return float(MAX_FITNESS - abs(ones - TARGET_ONES))


def init_population(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(POPULATION, CHROMOSOME_LENGTH), dtype=np.int8)


def evaluate(pop: np.ndarray) -> np.ndarray:
    return np.array([fitness_fn(ind) for ind in pop], dtype=float)


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])


def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2


def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    mask = rng.random(x.shape) < mut_rate
    y[mask] = 1 - y[mask]
    return y


def run_ga(
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    tournament_k: int = 3,
    elitism: int = 2,
    seed: int = 42,
    live_chart: bool = True,
):
    rng = np.random.default_rng(seed)

    pop = init_population(rng)
    fit = evaluate(pop)

    history_best, history_avg, history_worst = [], [], []

    chart_area = st.empty()
    info_area = st.empty()

    for gen in range(GENERATIONS):
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))

        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        if live_chart:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            info_area.markdown(f"Generation **{gen+1}/{GENERATIONS}** — Best fitness: **{best_fit:.2f}**")

       
        E = max(0, min(elitism, POPULATION))
        if E > 0:
            elite_idx = np.argpartition(fit, -E)[-E:]
            elites = pop[elite_idx].copy()
        else:
            elites = np.empty((0, CHROMOSOME_LENGTH), dtype=np.int8)

        
        next_pop = []
        while len(next_pop) < POPULATION - E:
            p1 = pop[tournament_selection(fit, tournament_k, rng)]
            p2 = pop[tournament_selection(fit, tournament_k, rng)]

          
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

           
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < POPULATION - E:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop, dtype=np.int8), elites]) if E > 0 else np.array(next_pop, dtype=np.int8)
        fit = evaluate(pop)


    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    history_df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
    return best, best_fit, history_df, pop, fit



st.set_page_config(page_title="Q1(b) GA Bit Pattern", layout="wide")
st.title("Q1(b) Genetic Algorithm: Bit Pattern Generator ")
st.caption(
    "Fixed parameters: Population=300, Chromosome Length=80, Generations=50, "
    "Fitness peaks at ones=40, Max fitness=80."
)

with st.sidebar:
    st.header("GA Controls (optional tuning)")
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    mutation_rate = st.slider("Mutation rate (per bit)", 0.0, 0.2, 0.01, 0.005)
    tournament_k = st.slider("Tournament size (k)", 2, 10, 3)
    elitism = st.slider("Elitism (best kept each gen)", 0, 20, 2)
    seed = st.number_input("Random seed", min_value=0, max_value=2**32 - 1, value=42)
    live_chart = st.checkbox("Live chart while running", value=True)

st.markdown("### Required Fixed Parameters")
st.write(
    {
        "Population": POPULATION,
        "Chromosome Length": CHROMOSOME_LENGTH,
        "Generations": GENERATIONS,
        "Fitness peak (ones)": TARGET_ONES,
        "Max fitness": MAX_FITNESS,
    }
)

if st.button("Run GA", type="primary"):
    best, best_fit, history_df, final_pop, final_fit = run_ga(
        crossover_rate=float(crossover_rate),
        mutation_rate=float(mutation_rate),
        tournament_k=int(tournament_k),
        elitism=int(elitism),
        seed=int(seed),
        live_chart=bool(live_chart),
    )

    st.subheader("Fitness Over Generations")
    st.line_chart(history_df)

    st.subheader("Best Bit Pattern (Final)")
    ones = int(np.sum(best))
    bitstring = "".join(map(str, best.astype(int).tolist()))
    st.code(bitstring, language="text")

    st.write(f"**Best fitness:** {best_fit:.2f} / {MAX_FITNESS}")
    st.write(f"**Number of ones:** {ones} / {CHROMOSOME_LENGTH}")
    st.write(f"**Target ones (peak fitness):** {TARGET_ONES}")

    st.markdown("### Interpretation (auto-generated)")
    if ones == TARGET_ONES:
        st.success(
            "The GA found an optimal chromosome with exactly 40 ones, achieving the peak fitness of 80."
        )
    else:
        st.info(
            f"The GA converged near the target. The best chromosome has {ones} ones, "
            f"so fitness is reduced by |{ones} - {TARGET_ONES}| = {abs(ones - TARGET_ONES)}."
        )

    st.subheader("Final Population Snapshot (first 20)")
    df_pop = pd.DataFrame(final_pop[:20])
    df_pop["fitness"] = final_fit[:20]
    st.dataframe(df_pop, use_container_width=True)
