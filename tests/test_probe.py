"""The evaluation set: one parent per place, and the contamination it admits.

The set exists because the first evaluation written for this work could not
be used: it asked which prefecture each of 1,134 Japanese municipalities is
in, and exactly one of those places appears in this corpus. A probe the
corpus says nothing about measures the model that was there before.

What replaced it asks the same shape of question about places the corpus does
talk about, which makes it a recall test rather than a generalisation test.
The tests below hold that line in both directions: every answer must appear in
cpt, and the set must not silently become something else.
"""
import collections

import build

EXPECTED = {
    "place-in-municipality": 66541,
    "municipality-in-prefecture": 1484,
}

# The municipality level is entirely train. The split is over places only,
# because holding out a municipality takes every sentence about every place
# inside it: 23.9% of the corpus to buy 142 questions.
EXPECTED_SPLIT = {
    "place-in-municipality train": 59885,
    "place-in-municipality eval": 6656,
    "municipality-in-prefecture train": 1484,
}


def test_counts(probe, manifest):
    by_level = collections.Counter(r["level"] for r in probe)
    assert dict(by_level) == EXPECTED
    assert manifest["probe"]["by_level"] == dict(by_level)
    assert manifest["probe"]["rows"] == len(probe)


def test_the_split_is_the_size_it_says(probe, manifest):
    counts = collections.Counter(
        f"{r['level']} {r['split']}" for r in probe)
    assert dict(counts) == EXPECTED_SPLIT
    assert manifest["probe"]["by_level_and_split"] == dict(counts)


def test_one_row_per_child(probe):
    """A place with two answers, both scored, both counted.

    The parent comes from a pair of layers rather than from a relation, and
    nothing in the geometry stops a ward meeting two states. Six of the 23
    wards meet Chiba as well as Tokyo. Those are dropped at build time; this
    is the check that they were.
    """
    seen = collections.Counter(r["child_id"] for r in probe)
    assert [c for c, n in seen.items() if n > 1] == []


def test_every_answer_is_stated_in_the_corpus(probe, cpt):
    """The claim on the card, as a test rather than as a sentence.

    If an answer were absent from cpt, a model scoring on it would be scoring
    on something it was never shown, and the number would mean the opposite of
    what the card says it means.
    """
    stated = {(r["subject_id"], r["object_id"]) for r in cpt}
    missing = [r for r in probe
               if (r["child_id"], r["parent_id"]) not in stated]
    assert missing == []


def test_the_split_holds_in_both_directions(probe, cpt):
    """The whole reason the split exists, checked as an invariant.

    A train question must have its fact in the part of the corpus a run
    trains on, or a score on it is not recall of anything. An eval question
    must have its fact nowhere in that part, or the score is recall wearing
    the word generalisation.

    Both directions matter and only one of them is obvious. The first version
    of the split marked a question eval when its subject was held out and left
    it train when its parent was, and questions whose fact had never been
    trained were counted as recall.
    """
    trained = {(r["subject_id"], r["object_id"])
               for r in cpt if not r["holdout"]}
    for r in probe:
        pair = (r["child_id"], r["parent_id"])
        if r["split"] == "train":
            assert pair in trained, r
        else:
            assert pair not in trained, r


def test_a_held_out_feature_is_absent_from_training_entirely(probe, cpt):
    """Not just absent as a subject: absent.

    A place named in passing is still taught. If 金閣寺 were held out but
    "京都市には金閣寺がある" stayed in the training half, the model would have
    read the answer in the other direction and the eval score would measure
    nothing.

    The held-out set is recomputed here from the rule the card states rather
    than read off the split column, so this also checks that the column is
    what it claims. It cannot be taken from the eval rows: a row is eval when
    either of its features is held out, and the other one is usually an
    ordinary trained municipality.
    """
    held_out = {i for i in
                {r["child_id"] for r in probe} | {r["parent_id"] for r in probe}
                if i.split("-")[0] == "poi" and build.is_eval(i)}
    assert held_out, "the rule held nothing out"
    seen = set()
    for r in cpt:
        if r["holdout"]:
            continue
        seen.add(r["subject_id"])
        seen.add(r["object_id"])
        if r["via_id"]:
            seen.add(r["via_id"])
    leaked = sorted(held_out & seen)[:5]
    assert not leaked, leaked

    # And the column agrees with the rule.
    for r in probe:
        want = "eval" if {r["child_id"], r["parent_id"]} & held_out else "train"
        assert r["split"] == want, r


def test_the_english_half_is_the_smaller_one(probe, manifest):
    """Which way round the language gap runs, as a fact rather than a habit.

    Every feature here is in Japan and carries a Japanese name, so the whole
    set is answerable in Japanese. name:en is optional and most mappers do not
    write it, so fewer than half the questions can be asked in English at all.
    A build where that reversed would mean the labels had started coming from
    somewhere other than the tags.
    """
    ja = [r for r in probe if r["child_ja"] and r["parent_ja"]]
    en = [r for r in probe if r["child_en"] and r["parent_en"]]
    assert manifest["probe"]["answerable_ja"] == len(ja) == len(probe)
    assert len(en) == 29005
    assert len(en) < len(ja)


def test_no_question_contains_its_own_answer(probe):
    """千代田 inside 千代田区, and Aruba inside Aruba.

    A question whose subject has its answer's name scores a model for free.
    Matched on the label rather than on the relation: the country cases are EQ
    and the ward cases are not, so an RCC8 filter catches one kind and leaves
    the other.
    """
    for r in probe:
        assert not (r["child_en"] and r["child_en"] == r["parent_en"]), r
        assert not (r["child_ja"] and r["child_ja"] == r["parent_ja"]), r


def test_the_chance_rate_is_the_generous_one(probe, manifest):
    """A denominator that flattered the result.

    Chance is quoted per level, and the two levels are not choosing among the
    same things. Getting this wrong in the other direction would make 94% at
    the ward level look like nothing.
    """
    c = manifest["probe"]["candidates"]
    assert c["place-in-municipality"] == 1740
    assert c["municipality-in-prefecture"] == 47
    parents = {r["level"]: set() for r in probe}
    for r in probe:
        parents[r["level"]].add(r["parent_id"])
    for level, seen in parents.items():
        assert len(seen) <= c[level], level


def test_every_answer_is_a_containment(probe):
    """The parent is what the child is within, and nothing looser.

    Every layer here is built from the same OpenStreetMap ways, so within is
    exact and there is no reason to accept anything weaker. Nothing in this
    set is PO: a place that half sticks out of its municipality has no single
    answer and was dropped rather than assigned one.

    The Tokyo dataset reads its parent off mere intersection, because Natural
    Earth's two layers disagree about one coastline. Copying that rule here
    would have thrown away every municipality on a prefectural border, which
    is 781 of the 1,740.
    """
    by_rcc8 = collections.Counter(r["rcc8"] for r in probe)
    assert by_rcc8["PO"] == 0
    assert by_rcc8["NTPP"] + by_rcc8["TPP"] + by_rcc8["EQ"] > 0
    # A place mapped as a node is not a region and has no RCC8 relation, so
    # an empty reading here is a point inside its municipality.
    assert by_rcc8[""] == 29391
