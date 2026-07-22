import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.smoke_test import (check_answerable, check_refused,
                                 check_citation_lookup, check_condensed)


def test_check_answerable():
    assert check_answerable({"citations": [{"n": 1}], "refused": False})[0] is True
    assert check_answerable({"citations": [], "refused": False})[0] is False
    assert check_answerable({"citations": [{"n": 1}], "refused": True})[0] is False


def test_check_refused():
    assert check_refused({"refused": True})[0] is True
    assert check_refused({"refused": False})[0] is False


def test_check_citation_lookup():
    assert check_citation_lookup({"citations": [{"n": 1}]})[0] is True
    assert check_citation_lookup({"citations": []})[0] is False


def test_check_condensed():
    assert check_condensed({"condensed_question": "Tội cướp giật bị phạt sao?"},
                           original="còn tội đó thì sao?")[0] is True
    assert check_condensed({"condensed_question": "còn tội đó thì sao?"},
                           original="còn tội đó thì sao?")[0] is False
