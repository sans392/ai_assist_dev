from datetime import date

from app.agents.base import BaseAgent
from app.services.sports import sports_service


class SportsAgent(BaseAgent):
    name = "sports"
    default_system_prompt = ""

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        sports_service._today = date.today()

        summary = sports_service.get_summary()
        recommendations = sports_service.get_recommendations()

        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_msg = m["content"]
                break

        analytics = sports_service.get_analytics(last_user_msg)

        parts = [self.system_prompt, "\n", summary]
        if recommendations:
            parts.extend(["\n\n", recommendations])
        if analytics:
            parts.extend(["\n\n", analytics])
        system_content = "".join(parts)

        return [{"role": "system", "content": system_content}] + [
            m for m in messages if m.get("role") != "system"
        ]
