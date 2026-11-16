"""Smart Tagging's team tags, for the diagnostics (scripts/diagnose.py).

Attacking phases name the team in possession, defending phases the other one. The pipeline does
not read them (tried as decoder evidence, they did not help: EXPERIMENTS.md, 10-31 and 11-15);
scripts/diagnose.py measures how often our predictions agree with them, as an independent check.
team_sides() maps the Team tag values onto home and away by the club names in the metadata.
"""

import re

ATTACKING = ("build up", "progression", "creating chances", "ball in final third",
             "ball in the box", "players in the box", "long ball", "attacking transition")
DEFENDING = ("defending in", "defensive transition")  # prefixes
EDGE_S = 1.0  # left out at each end of a phase, where it is least certain
H, A = 0, 1  # as the label codes


def _words(name):
    return {word for word in re.findall(r"[a-z]+", name.lower()) if len(word) > 2}


def team_sides(values, home, away):
    """{Team tag value: H or A}, by the words it shares with the metadata's club names; None if
    any value matches both or neither."""
    sides = {}
    for value in values:
        shared = [len(_words(value) & _words(home)), len(_words(value) & _words(away))]
        if shared[0] == shared[1]:
            return None
        sides[value] = H if shared[0] > shared[1] else A
    return sides


def tag_values(events):
    """The Team tag values that name a team."""
    return sorted({event["tags"].get("Team") for event in events} - {None, "N/A"})
