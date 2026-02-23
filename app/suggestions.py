from typing import List, Dict


def maintenance_suggestions(pr: float, rain_prob: float, efficiency_drop: bool) -> List[Dict[str, str]]:
    suggestions = []
    if rain_prob > 50:
        suggestions.append({"type": "notice", "message": "Delay module cleaning due to high rain probability"})
    if pr < 75:
        suggestions.append({"type": "alert", "message": "Low PR detected. Schedule inspection"})
    if efficiency_drop:
        suggestions.append({"type": "alert", "message": "Inverter efficiency drop. Maintenance required"})
    return suggestions

