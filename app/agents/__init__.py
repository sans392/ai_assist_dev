from app.agents.general import GeneralAgent
from app.agents.financial import FinancialAnalystAgent

AGENT_REGISTRY: dict[str, type] = {
    "general": GeneralAgent,
    "financial": FinancialAnalystAgent,
}


def get_agent_by_mode(mode: str):
    agent_cls = AGENT_REGISTRY.get(mode)
    if agent_cls is None:
        raise ValueError(f"Unknown agent mode: {mode!r}. Available: {list(AGENT_REGISTRY.keys())}")
    return agent_cls()
