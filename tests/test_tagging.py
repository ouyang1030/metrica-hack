"""Smart Tagging's team tags for the diagnostics (mtah/tagging.py), on synthetic data."""

from mtah import tagging


def test_team_names_map_by_shared_words_or_not_at_all():
    assert tagging.team_sides(["FC Barcelona", "Athletic Club"], "FC Barcelona",
                              "Athletic Club") == {"FC Barcelona": tagging.H,
                                                   "Athletic Club": tagging.A}
    assert tagging.team_sides(["Club"], "Club One", "Club Two") is None


def test_tag_values_leave_out_missing_and_na_teams():
    events = [{"tags": {"Team": "Home FC"}}, {"tags": {"Team": "N/A"}}, {"tags": {}},
              {"tags": {"Team": "Away FC"}}, {"tags": {"Team": "Home FC"}}]
    assert tagging.tag_values(events) == ["Away FC", "Home FC"]
