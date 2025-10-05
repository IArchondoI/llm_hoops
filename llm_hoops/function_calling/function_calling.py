"""Add function calling capabilities to a model."""

from mistralai import Tool
import pandas as pd
import functools
from dataclasses import dataclass
from typing import Callable, Mapping

@dataclass
class FunctionCallObjects:
    """Hold objects associated with function calling."""
    tools: list[Tool]
    names_to_functions: Mapping[str, Callable[..., object]]

def get_players_ordered_by_fppm(df: pd.DataFrame, player_position: str = "ALL", team: str|None = None) -> list[tuple[str, float]]:
    """Get a list of players ordered by fppm with their FPPM values."""
    filtered_df = df[df["G"] > 30]
    if player_position!="ALL":
        filtered_df = filtered_df[filtered_df["PosH"] == player_position]
    if team:
        if team not in list(filtered_df["Team"].unique()):
            raise ValueError("Team not in database")
        filtered_df = filtered_df[filtered_df["Team"]==team]
    sorted_df = filtered_df.sort_values(["FPPM"], ascending=False)
    return list(sorted_df[["Player", "FPPM"]].itertuples(index=False, name=None))

def setup_function_calling(df:pd.DataFrame)->FunctionCallObjects:
    """Add function calling capabilities to an LLM."""

    tools = [
        {
            "type":"function",
            "function": {
                "name":"get_players_ordered_by_fppm",
                "description": ("Get NBA players ordered by fantasy points per minute (FPPM) from best to worst, "
                "together with their FPPM in a tuple."),
                "parameters":{
                    "type":"object",
                    "properties": {
                        "player_position": {
                            "type": "string",
                            "description": ("Optional player position filter. "
                            "One of 'G', 'F', 'C' or 'ALL'. ALL includes all players.")
                        },
                     "team": {
                            "type": ["string","null"],
                            "description": ("Optional team position filter. Input is the three letter abbreviation for each team in string format. "
                            "If not specified, the function returns players from all teams.")
                        }},
                    "required": ["player_position"]
                    }
            }
        }
    ]

    names_to_functions = {"get_players_ordered_by_fppm":functools.partial(
        get_players_ordered_by_fppm, df=df
    )}


    return FunctionCallObjects(tools=tools,names_to_functions=names_to_functions)


