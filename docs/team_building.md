# Code-ranked team building

Before this change the LLM chose draft picks and Team Preview lineups from
card text and damage tables alone. That is a combinatorial job: 15 ways to
bring 4 of 6, times 6 lead pairs, or up to 17 legal draft cards. Draft picks
were the weakest decisions in our traces (generic picks, no denial).

`agent/pokemon/teambuild.py` now scores the options in code and shows the LLM
a short, explained shortlist. **The LLM still decides.** Code only uses the
shortlist itself when the model fails (the fallback is the top-ranked option
instead of the first legal one).

## How the score is computed

- **Duel(a, b):** can `a` beat `b` one-on-one? It compares the hits each needs
  to KO the other with its best attack at full HP, from the verified damage
  calculator. A tie goes to the one that moves first: priority, then Speed;
  an exact Speed tie scores 0. The value is +1 for a clear win and −1 for a
  clear loss.
- **Team value vs. a set of foes:**
  - For each foe, take our best duel against it.
  - Weight foes by how many of our Pokémon they beat (1–2×).
  - Then add role terms:
    - speed control +0.15, a second source +0.05;
    - Fake Out +0.10;
    - Intimidate, redirection, a spread move and a physical/special mix +0.05 each;
    - Tailwind plus Trick Room −0.10;
    - each stacked weakness −0.08 (one attacking type hits 3 of 4 brought, or 4 of 6 drafted).
- **Team Preview:** score every bring-4 / lead-2 choice against their six.
  - The lead pair's score counts half.
  - If memory knows the opponent's past leads, those foes weigh more in the lead score.
  - Show the top 3, with which of ours answers each of theirs and what stays unanswered.
- **Draft:** for each legal card, combine two parts and show the top 5:
  - **Gain:** how much it improves our team value. The foes are their picks (2×) plus the undrafted pool, as a proxy for their future picks.
  - **Denial:** how much it would improve *their* team value against ours. It is weighted 0.6 when they pick next in the snake order, otherwise 0.3.

All weights are plain constants at the top of `teambuild.py`. They are
hand-set heuristics, not fitted. The ranking runs in ≤0.02 s per decision.

## What the model does with it (replay of a recorded game, gpt-5.5)

- **Draft:** two replays of the same 6 draft states. The model took the
  top-ranked card 9 of 12 times. The other 3 times it took #2 and said why:
  "Dragonite is tempting denial, but Cresselia better stabilizes into their
  fast Thundurus/Rillaboom core and keeps item diversity."
- **Team Preview:** it took the #3 lineup over #1: "Top code line is solid,
  but Milotic is risky into likely Rillaboom and doesn't threaten Gyarados.
  I prefer Garchomp over Milotic: Sash gives insurance…"

This is the intended split. Code does the exhaustive matchup arithmetic,
and the model adds what the heuristic can't see: field effects (Tailwind,
Trick Room, weather, Intimidate, Fake Out), focus fire, and the opponent's
likely plan. The v2 draft and preview prompts say this explicitly. The
traces log `code_ranking` next to the model's choice, so agreement can be
measured on live games.

## Known limits

- **Duels are one-on-one at full HP.** They ignore field effects and two-on-one
  focus fire, abilities like Intimidate, and Speed changes from Tailwind or
  Trick Room. A Trick Room team's slow attackers therefore look worse than
  they are.
- **The future-picks proxy is crude.** The undrafted pool stands in for the
  opponent's future picks, so every remaining card counts, not only those they
  are likely to take.

## A/B test: does it win more?

Replays show the decisions change, not that they're better. To measure it:

1. Start the runtime with the switch on:
   `POKEMON_AB_TEAMBUILD=1 RECORDER_INNER=agent.agent python -m agent --match --agent tools.recorder`
2. Create self-play Testing matches (both seats "Mine (self-hosted)").
3. Each game, one seat is **ranked** (sees the CODE-RANKED shortlists) and
   the other is **plain** (the agent before this change, including its old
   first-legal fallback). Which seat gets which comes from a hash of the
   session id, so it changes from game to game, and draft order is
   balanced on average.
4. Both seats battle with identical code, so a difference in wins comes from
   the draft and Team Preview alone.
5. Each seat appends its arm and result to `logs/ab/teambuild.jsonl`.
   `python -m tools.ab_report` pairs the seats of each game. It prints the
   ranked arm's win rate with a Wilson 95% interval and an exact binomial
   test against 50%, also split by which arm drafted first.

Sample size (two-sided α = 0.05, 80% power): ~200 games detect a true
60% win rate, ~85 detect 65%, ~50 detect 70%. A small run only catches a
large effect or a regression.

The switch is off unless `POKEMON_AB_TEAMBUILD=1` is set; tournament games
always use the ranking.
