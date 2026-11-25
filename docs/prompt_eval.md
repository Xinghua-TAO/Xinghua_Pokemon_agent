# Prompt evaluation (Step 6)

The LLM's instructions live in `agent/pokemon/prompts/<version>/` (one file per
phase: `draft.md`, `preview.md`, `battle.md`, `reflect.md`, plus `common.md`).
`v1` is the original prompt set; `v2` is the rewrite (default). Pick one with
`POKEMON_PROMPT_VERSION`.

## Method

1. **Find real mistakes.** Every recorded battle decision (92 at the time) was
   scanned for clear error patterns: two attacks into a target that one attack
   already knocks out (overkill), Protect with a low success chance, attacks
   that hit our own partner, skipped guaranteed KOs. The scan found 2
   overkills and 4 skipped "guaranteed KOs". Several of the skipped KOs were
   reasonable, because the opponent moved first and would have knocked our
   attacker out. That led to new code-computed facts (who moves first,
   DANGER, double-target KOs) rather than only prompt changes.
2. **Build a test set from real game states.** Seven recorded situations,
   each with a rule for what counts as a mistake
   (`tools/eval_prompts.py`, states in `tools/eval_states/`). Four are
   overkill / focus-fire cases and three are low-odds Protect cases.
3. **Measure.** The agent decides each situation 3 times per prompt version
   with the real model. The score is the share of decisions without the
   mistake.

One label was wrong. For the "Thundurus at 8%" state we first required a KO
on Thundurus. The model instead double-targeted Ting-Lu for a sure KO and
noted that Thundurus's Life Orb recoil (~10%) knocks it out if it attacks.
That line removes both opponents, so it is better than our label, and the
rule was corrected. Human labels need checking too.

## Results (decisions without the mistake, 7 scenarios x 3 runs)

| Model | v1 | v2 |
|---|---|---|
| gpt-5.5 (our battle model) | 21/21 | 21/21 |
| gpt-5.5, reasoning effort none (our fast retry) | 21/21 | 21/21 |
| gpt-5.4-mini | 11/21 (12/21) | **16/21** (15/21) |
| gpt-4o-mini | 8/21 (10/21) | **10/21** |

Numbers in parentheses are a second run; with 3 runs per cell, differences of
1-2 are noise.

- With the code-computed facts, gpt-5.5 makes none of these mistakes under
  either prompt, so this test set is at its ceiling for our model.
- On weaker models, v2 reduces overkill. The first v2 draft made gpt-4o-mini
  worse on Protect (0/3 in two scenarios): its "if in DANGER, Protect" step
  overrode the "only 33% chance" warning. Moving two hard rules to the top
  ("never choose an option marked WARNING", "never overkill") fixed it.
- Draft reasoning (qualitative, same recorded draft): v2 commits to a plan
  ("commit to fast special offense", "Tailwind offense is ideal ... we need
  reliable speed control before the opponent's double pick"). v1 called
  Garchomp "speed control" and stated no overall plan.

## Reasoning effort for battle turns (gpt-5.5)

In a live self-play game with gpt-5.5 at its default reasoning effort, 4 of 25
battle decisions (16%) hit our 30 s request timeout. The fast retry (effort
"none") answered them in ~3.5 s, but each timeout wasted ~30 s of the 420 s
battle bank. On the same 7 scenarios:

| gpt-5.5 battle effort | Score | Time per decision |
|---|---|---|
| default | 21/21 | 3.8-33.4 s (median ~15 s) |
| low | 21/21 | 2.5-13.9 s (median ~3 s) |

We use `POKEMON_REASONING_EFFORT=low` for battles and team preview, and
`none` for draft picks. The eval is saturated at gpt-5.5, so a subtle loss
from lower effort can't be excluded; the timeout cost is measured.

Reproduce: `python -m tools.eval_prompts --versions v1,v2 --runs 3`
(set `POKEMON_MODEL` / `POKEMON_REASONING_EFFORT` to compare models and effort).

## A view bug found in a live game: "refreshing" Tailwind

In a live self-play game (2026-11-25), Tornadus used Tailwind on turn 2
while our Tailwind was already up. Using Tailwind again fails, and Iron
Bundle then knocked Tornadus out before it did anything. The model's
reasoning: "refreshing Tailwind before Tornadus goes down".

The cause was our view, not the model. The server reports active effects as
`{"TAILWIND": 1}`, which means set on turn 1. Our view printed this as
"Your side conditions (turns left): {'TAILWIND': 1}", so the model read
"1 turn left".

Fix (code, not prompt):

- The view now shows each effect with when it was set, its last turn, and
  the turns left. Durations were measured in our recorded battles: Tailwind
  set on turn T lasts through T+3; screens and Trick Room through T+4;
  weather and terrain set by an ability at the start last through turn 5.
- Moves that would fail right now (Tailwind or a screen already up, the same
  weather or terrain already active, Aurora Veil without snow) are marked
  WARNING, which the v2 prompt forbids and the fallback avoids.
- Re-using Trick Room gets a NOTE that it ends Trick Room.

The state is now eval scenario `tailwind_already_up`. Old view: 2/3
decisions avoided the failing Tailwind, plus the live failure. New view: 3/3
(gpt-5.5, v2).

Lesson: check the meaning of every number shown to the model. A
mislabeled fact is worse than a missing one.
