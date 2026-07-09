"""Non-prose (route/structured plaintext) genre flagging."""

from buttcrack.nonprose import (
    GenreModel,
    default_models,
    nonprose_flag,
    route_corpus,
)

# A route/instruction-style plaintext: directions, spelled numbers, units, imperatives.
ROUTE_TEXT = "NORTHSEVENPACESLEFTATTHEOAKEASTTHREEPACESTURNRIGHTCOUNTFOURDOORS"
# A perfectly ordinary English prose sentence.
PROSE_TEXT = "the meeting was held in the old library where they discussed the plans for the year"


def test_route_corpus_is_deterministic_letters_only():
    a = route_corpus(50, seed=0)
    b = route_corpus(50, seed=0)
    assert a == b
    assert a.isupper() and a.isalpha()
    # A different seed produces different text.
    assert route_corpus(50, seed=1) != a
    # More sentences => longer corpus.
    assert len(route_corpus(100, seed=0)) > len(a)


def test_model_scores_are_finite_and_length_independent():
    model = GenreModel.train(route_corpus(100))
    # Smoothing means no input (even all-random or unseen) returns -inf.
    for text in ["", "Q", "QXZ", "ABCDEFG", ROUTE_TEXT]:
        assert model.score(text) > float("-inf")
    # score is a mean per character, so repeating text leaves it ~unchanged.
    one = model.score(ROUTE_TEXT)
    twice = model.score(ROUTE_TEXT + ROUTE_TEXT)
    assert abs(one - twice) < 0.5


def test_route_model_prefers_route_prose_model_prefers_prose():
    route_model, prose_model = default_models()
    # The route model rates the route string higher than a prose sentence.
    assert route_model.score(ROUTE_TEXT) > route_model.score(PROSE_TEXT)
    # The prose model rates the prose sentence higher than the route string.
    assert prose_model.score(PROSE_TEXT) > prose_model.score(ROUTE_TEXT)


def test_nonprose_flag_separates_route_from_prose():
    route_flag = nonprose_flag(ROUTE_TEXT)
    prose_flag = nonprose_flag(PROSE_TEXT)

    assert route_flag["leans_nonprose"] is True
    assert prose_flag["leans_nonprose"] is False

    # The route candidate scores higher under the route model than the prose model,
    # and the prose candidate does the opposite.
    assert route_flag["route_score"] > route_flag["prose_score"]
    assert prose_flag["prose_score"] > prose_flag["route_score"]

    # delta is route - prose and is reported consistently with the fracs.
    assert route_flag["delta"] > 0
    assert prose_flag["delta"] < 0
    assert route_flag["delta"] == route_flag["route_score"] - route_flag["prose_score"]

    # Verdicts are the expected labels.
    assert "non-prose" in route_flag["verdict"]
    assert route_flag["verdict"] != prose_flag["verdict"]


def test_nonprose_flag_accepts_custom_models():
    route_model, prose_model = default_models()
    flag = nonprose_flag(ROUTE_TEXT, route_model=route_model, prose_model=prose_model)
    assert flag == nonprose_flag(ROUTE_TEXT)


def test_frac_axis_genre_typical_near_one_random_near_zero():
    route_model, _ = default_models()
    # Fresh route text (unseen exact string, same genre) sits near the 1.0 anchor.
    fresh_route = route_corpus(20, seed=99)
    assert route_model.frac(fresh_route) > 0.7
    # Uniform-random letters sit near the 0.0 anchor.
    import random

    rng = random.Random(7)
    noise = "".join(chr(65 + rng.randrange(26)) for _ in range(200))
    assert route_model.frac(noise) < 0.3


def test_default_models_are_cached():
    assert default_models() is default_models()
