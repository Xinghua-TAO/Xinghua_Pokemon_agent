"""Our team's agent: one entry point that routes each game to its own agent.

The tournament gives a team ONE Official Agent Key and runs ONE process, so a
single `create_agent()` must play every game type. The runtime calls it once
per game (in that game's own process); the router picks the game's agent on
the first decision, from `context.game_type`:

- pokemon_*  -> agent.pokemon.PokemonAgent (LLM-driven, code-guarded)
- anything else (Werewolf, ...) -> a placeholder that plays the first legal
  action and auto-ends discussion rounds, until the teammate's agent is
  plugged in here (e.g. agent.werewolf.WerewolfAgent).
"""

from altruagent import TERMINATE_MESSAGING, DecisionContext, GameState


class _FirstLegalAction:
    """Placeholder for games we have no agent for yet."""

    def choose_action(self, state: GameState, context: DecisionContext):
        return state.legal_actions[0]


class GameRouter:
    def __init__(self) -> None:
        self._agent = None

    def _agent_for(self, context: DecisionContext):
        if self._agent is None:
            game_type = context.game_type or ""
            if game_type.startswith("pokemon"):
                from agent.pokemon import PokemonAgent

                self._agent = PokemonAgent()
            else:
                self._agent = _FirstLegalAction()
        return self._agent

    def choose_action(self, state: GameState, context: DecisionContext):
        return self._agent_for(context).choose_action(state, context)

    def choose_message(self, state: GameState, context: DecisionContext):
        handler = getattr(self._agent_for(context), "choose_message", None)
        return handler(state, context) if callable(handler) else TERMINATE_MESSAGING

    def on_action_result(self, result, context: DecisionContext) -> None:
        handler = getattr(self._agent_for(context), "on_action_result", None)
        if callable(handler):
            handler(result, context)

    def on_game_end(self, final_state: GameState, context: DecisionContext) -> None:
        """Called by the runtime (altruagent/worker.py) once the game is over."""
        if self._agent is None:
            return  # never asked for a move: nothing to remember
        handler = getattr(self._agent, "on_game_end", None)
        if callable(handler):
            handler(final_state, context)


def create_agent():
    return GameRouter()
