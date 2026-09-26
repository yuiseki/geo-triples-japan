#!/usr/bin/env python3
"""Turn the oracle's relations into spatial triples with their provenance.

The oracle is YuisekinGeoSPARQL: a Docker compose that loads pinned Hugging
Face revisions of OpenStreetMap and Natural Earth into Apache Jena Fuseki and
writes relations.tsv, the DE-9IM matrix of every pair of features that is not
disjoint. Nothing here talks to a network or to a model. Given the same
relations.tsv it writes the same Parquet, byte for byte.

Two tables come out.

  triples      one row per pair and predicate, so eight rows per pair, plus
               one row for each triple the composition table entailed.
               Choosing negatives this way leaves no room for a sampling rule
               to be argued with: one matrix decides all eight.

  probe        the evaluation set: which parent each place has, at each
               level of the hierarchy. Derived from the same triples, so it
               is pinned by the same revisions, and contaminated by them:
               every answer is stated somewhere in cpt.

  cpt          one row per true triple per form. Every true triple has an
               N-Triples row; it has an en or ja row only where both features
               carry a label in that language, so the label gap is a missing
               row beside a present one rather than an invisible choice.

    python3 src/build.py --relations ../YuisekinGeoSPARQL/data/relations.tsv \
        --oracle ../YuisekinGeoSPARQL/data/manifest.json --out data

Determinism is the property this file is built around, so nothing that
reaches the output may be iterated in the order a set or a dict happens to
have. Every such loop below is wrapped in sorted(), and the two output tables
are sorted on keys that are total, so no pair of rows is left to be separated
by the order they were appended in.
"""
import argparse
import collections
import csv
import re
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import certify  # noqa: E402
import vocab  # noqa: E402

# What a pair absent from relations.tsv is, for two areas. Disjointness is
# spelled by the kinds, and the oracle's manifest carries the whole table, but
# only this entry is ever reached here: a pair is filled in only when it was
# composed, composition needs RCC8 on both premises, and RCC8 holds only
# between regions. A point never reaches this line.
DISJOINT_MATRIX = "FF2FF1212"

# 2 when the cpt table stopped being one row per sentence and became one row
# per form, with N-Triples as the canonical one. 3 when the certification
# columns arrived and the derivation of an observed row stopped being named
# after the tool that made it. 4 when the named places arrived and with them
# the kind columns, because a pair is no longer always two areas. A reader who
# has an older file can tell from this field alone.
SCHEMA_VERSION = "5"


def read_relations(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# The default pair of kinds, for a caller that has no kinds to give. The
# prover asks for them because sfOverlaps and sfCrosses are defined by cases
# on the operands' dimensions, and since the places arrived not every pair
# here is two areas.
KINDS = ("area", "area")


def certified(verdicts, matrix, predicate, truth, kinds=KINDS):
    """What LeanGeospatial proves about one reading, if it proves anything.

    Three states, kept apart on purpose. The oracle observed the matrix; that
    is not in question here. The question is whether the step from the matrix
    to the predicate is proved, refuted, or left open, and a row that is open
    says uncertified rather than borrowing the oracle's confidence.

    A verdict that contradicts the observation stops the build. The two are
    talking about the same matrix, so they cannot both be right, and shipping
    the disagreement inside a certification column would be the worst of the
    three outcomes.
    """
    verdict = verdicts.get((matrix, kinds[0], kinds[1], predicate))
    if verdict is None:
        return "uncertified", None
    if (verdict == "entailed") != bool(truth):
        raise SystemExit(
            f"LeanGeospatial {verdict} {predicate} for matrix {matrix}, and "
            f"the oracle read it as {truth}. One of the two is wrong about "
            f"this matrix; the build will not ship either reading until it "
            f"is known which.")
    return "certified", f"de9im:{verdict}"


def direct_triples(rows, iris, verdicts):
    """Eight rows per pair: the predicates that hold, and those that do not.

    More than one Simple Features predicate can hold of a pair, because they
    are not mutually exclusive: two equal areas are equals, intersects,
    within and contains all at once. The exclusive reading is the RCC8 column
    beside them, where exactly one of eight holds. What makes the negatives
    defensible is not that there is one positive but that the truth of all
    eight comes from one matrix, so a reader who disagrees with a false has
    the evidence to say why.
    """
    out = []
    for r in rows:
        held = set(r["sf_raw"].split(",")) if r["sf_raw"] else set()
        kinds = (r["subject_kind"], r["object_kind"])
        for predicate in vocab.SF:
            truth = predicate in held
            certification, certificate = certified(
                verdicts, r["de9im_raw"], predicate, truth, kinds)
            out.append({
                "subject_id": r["subject_id"],
                "subject_iri": iris[r["subject_id"]],
                "subject_name": r["subject_name"],
                "subject_source": r["subject_source"],
                "subject_layer": r["subject_layer"],
                "subject_kind": r["subject_kind"],
                "predicate": predicate,
                "object_id": r["object_id"],
                "object_iri": iris[r["object_id"]],
                "object_name": r["object_name"],
                "object_source": r["object_source"],
                "object_layer": r["object_layer"],
                "object_kind": r["object_kind"],
                "truth": truth,
                "certification": certification,
                "certificate": certificate,
                "de9im": r["de9im_raw"],
                "rcc8": r["rcc8_raw"],
                "derivation": "observed",
                "via_id": None,
                "via_iri": None,
                # Empty rather than zero when the oracle did not measure it.
                # Zero is a real reading here, and writing it where nothing
                # was measured would make an unmeasured pair look like a
                # perfectly nested one.
                "outside_ratio": (float(r["outside_ratio"])
                                  if r["outside_ratio"] else None),
            })
    return out


# The eight, by their lower-case spelling, so a table that stores them in one
# case hands back the other without a case rule having to be invented twice.
CANONICAL = {r.lower(): r for r in vocab.RCC8}


def was_compared(oracle):
    """Whether the oracle formed the pair between two layers at all.

    Absent from relations.tsv means disjoint only for a pair the oracle
    looked at. tokyo23-poi is compared against the wards and against nothing
    else, so a place and a country are absent because they were never formed,
    and reading that absence as disjointness says Sensoji is not in Japan.
    """
    limits = oracle["relations"].get("layers_compared") or {}

    def compared(a_layer, b_layer):
        for first, second in ((a_layer, b_layer), (b_layer, a_layer)):
            allowed = limits.get(first)
            if isinstance(allowed, list) and second not in allowed:
                return False
        return True

    return compared


def composed_triples(rows, limit_per_cell=None, compared=None,
                     layer_of=None):
    """Triples the proved composition table settles, with the step they came
    through.

    Only the cells where the table leaves exactly one relation are used. Where
    it leaves several, nothing is entailed and there is nothing to record. The
    table itself is not reimplemented here: it is read from the file
    LeanGeospatial publishes.

    At most one triple is kept per ordered pair. A pair can often be reached
    through many intermediates and they all entail the same relation, so the
    rest would be the same row with a different via_id.
    """
    rel = {(r["subject_id"], r["object_id"]): r for r in rows}
    out_edges = collections.defaultdict(set)
    for (a, b) in rel:
        out_edges[a].add(b)

    table = composition_table()
    seen = set()
    found = []
    per_cell = collections.Counter()
    # Sorted at every level. The intermediate that is kept for a pair is the
    # first one reached, so an unsorted loop would not just reorder the output
    # but change which via_id it records.
    for a in sorted(out_edges):
        for b in sorted(out_edges[a]):
            r = rel[(a, b)]["rcc8_raw"]
            for c in sorted(out_edges.get(b, ())):
                if c == a:
                    continue
                s = rel[(b, c)]["rcc8_raw"]
                allowed = table.get((r.lower(), s.lower()))
                if not allowed or len(allowed) != 1:
                    continue
                if limit_per_cell and per_cell[(r, s)] >= limit_per_cell:
                    continue
                key = (a, c)
                if key in seen:
                    continue
                seen.add(key)
                per_cell[(r, s)] += 1
                # Back to the spelling the rest of the world uses. The table
                # is lower case and upper() would give NTPPI, which is not a
                # relation anybody names. Until the places arrived only DC and
                # EC were ever entailed, and both survive upper() unharmed,
                # so this waited to be found.
                t = CANONICAL[next(iter(allowed))]
                observed = rel.get((a, c))
                if observed is None and compared and layer_of and \
                        not compared(layer_of[a], layer_of[c]):
                    # Entailed, and nothing to check it against. The pair was
                    # never formed, so the row keeps its derivation and leaves
                    # the observation columns empty rather than claiming a
                    # matrix nobody read. These are the rows that reach beyond
                    # what the oracle measured, which is what a composition
                    # table is for.
                    found.append({
                        "a": a, "b": b, "c": c, "r": r, "s": s, "t": t,
                        "observed": None, "observed_de9im": None,
                    })
                    continue
                found.append({
                    "a": a, "b": b, "c": c, "r": r, "s": s, "t": t,
                    "observed": observed["rcc8_raw"] if observed else "DC",
                    "observed_de9im": (observed["de9im_raw"] if observed
                                       else DISJOINT_MATRIX),
                })
    return found, dict(per_cell)


def contradictions(composed):
    """Composed triples whose entailment disagrees with the geometry.

    A row the oracle never formed has nothing to disagree with and is not
    counted here. A row it did form and read differently is the real thing.

    A single-valued cell of a proved table leaves no room for disagreement,
    so a non-empty result here is not a data quirk to be counted and moved
    past: either the oracle mis-read a matrix or the vendored table is not
    the one that was proved. The build stops rather than shipping both
    readings in one row.
    """
    return [c for c in composed
            if c["observed"] is not None and c["t"] != c["observed"]]


def composition_table(path=None):
    """The 64 cells, read from the table LeanGeospatial publishes.

    The file is the RCC8 composition table as Wikipedia states it, and says
    so in its own header. What makes it usable as a premise rather than as a
    citation is LeanGeospatial's theorem table_eq_published, which shows the
    table derived from its own definitions agrees with this one on all 64
    cells. Reading the TSV is therefore reading the proved table, and is far
    less likely to go wrong than transcribing 64 cells by hand a second time.
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "vendor", "rcc8_known_table.tsv")
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or parts[0] == "r":
                continue
            r, s, result = parts
            table[(r, s)] = set(result.split(","))
    return table


def statements(triple, iris):
    """The statements a cpt row writes out, in the order it writes them.

    One for an observation. Three for a deduction: the two premises and then
    the conclusion, because a conclusion on its own is indistinguishable from
    one more observation, and the premises are what make the derivation
    column checkable from the text.
    """
    if triple["derivation"] != "composition":
        return [triple]
    return [
        {"subject_id": triple["subject_id"],
         "subject_iri": triple["subject_iri"],
         "predicate": triple["premise_a_predicate"],
         "object_id": triple["via_id"], "object_iri": triple["via_iri"]},
        {"subject_id": triple["via_id"], "subject_iri": triple["via_iri"],
         "predicate": triple["premise_b_predicate"],
         "object_id": triple["object_id"],
         "object_iri": triple["object_iri"]},
        {"subject_id": triple["subject_id"],
         "subject_iri": triple["subject_iri"],
         "predicate": triple["predicate"],
         "object_id": triple["object_id"],
         "object_iri": triple["object_iri"]},
    ]


# The layers that are administrative units rather than things inside them.
# A row is `admin` when every feature it names is one of these.
#
# The division is here so that a run can take the hierarchy without the
# 80,748 places. A 0.6B model memorising every shrine and school in Japan is
# not the first thing to find out; whether it can hold 1,740 municipalities
# and their prefectures is, and that is a corpus two orders smaller.
ADMIN_LAYERS = ("jp-country", "jp-pref", "jp-muni")


def topic_of(triple, layer_of):
    named = [triple["subject_layer"], triple["object_layer"]]
    if triple["via_id"]:
        named.append(layer_of.get(triple["via_id"], ""))
    return "admin" if all(l in ADMIN_LAYERS for l in named) else "place"


def cpt_rows(triples, iris, labels, held_out=(), layer_of=None):
    """Every true triple, in each form it can be written in.

    The N-Triples form is unconditional, so the row count per form is a
    measurement of label coverage rather than of anything else: a predicate
    that is never spoken, or a feature with no Japanese label, shows up as an
    ntriples row with no ja row beside it.
    """
    out = []
    for t in triples:
        if not t["truth"]:
            continue
        parts = statements(t, iris)
        texts = {"ntriples": vocab.nt_text(parts)}
        for lang in vocab.LANGS:
            text = vocab.sentence_text(parts, labels, lang)
            if text:
                texts[lang] = text
        for form in vocab.FORMS:
            if form not in texts:
                continue
            out.append({
                "text": texts[form], "form": form,
                "subject_id": t["subject_id"], "predicate": t["predicate"],
                "object_id": t["object_id"],
                "derivation": t["derivation"], "via_id": t["via_id"],
                "de9im": t["de9im"], "rcc8": t["rcc8"],
                # Carried through rather than recomputed, so that a sentence
                # a model is trained on can be traced to what certified the
                # triple behind it without a join back to the other table.
                "certification": t["certification"],
                "certificate": t["certificate"],
                # True when any feature this row names is held out. A row that
                # mentions a held-out place in passing still teaches it, so
                # the whole row goes, not just the ones it is the subject of.
                "holdout": bool(
                    held_out & {t["subject_id"], t["object_id"],
                                t["via_id"] or ""}),
                "topic": topic_of(t, layer_of or {}),
                # Which two layers the row is about, so that a run can take
                # one rung of the hierarchy. Without it the only way to train
                # on municipalities and prefectures alone is to filter on the
                # id prefixes by hand, and the country rung is 38% of the
                # containment sentences: every municipality is in Japan, so a
                # model that learns only that has learnt the majority answer.
                "pair": f"{t['subject_layer']}>{t['object_layer']}",
            })
    # A total key. subject, predicate and object do not separate a composed
    # row from the observation it agrees with, and nor does derivation
    # separate two composed rows through different intermediates, so via_id
    # and the text itself are part of the key.
    out.sort(key=lambda r: (r["form"], r["subject_id"], r["predicate"],
                            r["object_id"], r["derivation"],
                            r["via_id"] or "", r["text"]))
    return out


# The IRI of a feature, and its labels, as the oracle wrote them. The trailing
# space matters: without it geo:FeatureCollection matches too, and each
# graph's collection node is picked up as though it were one of its features.
FEATURE = re.compile(r"^src:(\S+) a geo:Feature[ ,]", re.M)
LABEL = re.compile(r'^\s+rdfs:label "((?:[^"\\]|\\.)*)"@(\w+)', re.M)
PREFIX = re.compile(r"^@prefix src:\s+<([^>]+)>", re.M)


def feature_labels(ttl_paths):
    """Every feature's IRI and its label in each language, from the graphs.

    The graphs are where both languages live: relations.tsv carries one
    display name per feature, which is Japanese for a ward and English for a
    country, so a sentence built from it would be half in each. They are also
    where the IRIs live, and the N-Triples form has to use the same ones the
    oracle published or it would name features that nothing else refers to.
    """
    iris, labels = {}, {}
    for path in ttl_paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        m = PREFIX.search(text)
        if not m:
            raise ValueError(f"{path} declares no src: prefix")
        base = m.group(1)
        current = None
        for line in text.split("\n"):
            m = FEATURE.match(line)
            if m:
                current = m.group(1)
                iris.setdefault(current, base + current)
                continue
            if current is None:
                continue
            m = LABEL.match(line)
            if m:
                value = m.group(1).replace('\\"', '"').replace("\\\\", "\\")
                labels.setdefault((current, m.group(2)), value)
            elif line.endswith("."):
                # A Turtle block ends with a full stop on its own last line.
                # Without this the geometry block that follows would still be
                # read as part of the feature.
                current = None
    return iris, labels


# The child layer and the parent layer of each level of the hierarchy.
#
# The parent is the feature the child is within, not merely one it meets.
# That is the opposite of what the Tokyo dataset does, and the reason is the
# sources. There, Natural Earth's admin-0 and admin-1 draw one coastline
# twice and disagree, so 40.7% of states merely overlap their own country and
# within would throw most of them away. Here every layer is built from the
# same OpenStreetMap ways, so within is exact: all 1,740 municipalities are
# within exactly one prefecture.
#
# Meeting is not enough at this size. A municipality on a prefectural border
# shares boundary ways with the prefecture next door, so it touches it, and
# touching is intersecting: 781 of the 1,740 meet two prefectures or more.
# A rule that dropped them would keep the inland ones and quietly throw away
# every border municipality in the country.
LEVELS = {
    ("jp-poi", "jp-muni"): "place-in-municipality",
    ("jp-muni", "jp-pref"): "municipality-in-prefecture",
}

# jp-pref inside jp-country is in the triples and not in the probe. There is
# one country in this graph, so "which country is 東京都 in" has one possible
# answer and measures nothing.
NOT_A_QUESTION = {("jp-pref", "jp-country")}


# One feature in ten is held out, chosen by its own id so that the split is
# the same wherever it is computed and does not depend on the order anything
# was read in.
#
# The split is over features, not over rows. Holding out rows would leave the
# same place named in a dozen other sentences, and a model that had seen
# "金閣寺は京都市に含まれる" in one form would be asked it in another. Held
# out here means the feature is not in the training corpus at all.
#
# Only the places. Holding out a municipality as well takes every sentence
# about every place inside it, which cost 23.9% of the corpus to buy 142
# questions, and the municipality level has 1,484 questions in total.
#
# What the eval half can show is narrower than it sounds, and worth saying
# plainly: a place that never appears in training cannot be recalled from it.
# Two things can still move. A model may learn the naming rule, since 39% of
# these places carry their municipality inside their own name, and that rule
# transfers to places it has never seen. And a model that has merely learnt
# to answer with a plausible municipality will move both halves together,
# which is how this half earns its keep: it is the control that says a rise
# in the other one is the facts going in.
EVAL_SHARE = 10
EVAL_LAYERS = ("jp-poi",)


def is_eval(feature_id):
    digest = hashlib.sha256(feature_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % EVAL_SHARE == 0


def eval_features(triples):
    """The ids held out of training, and the layers they come from."""
    out = set()
    for t in triples:
        for side in ("subject", "object"):
            if t[side + "_layer"] in EVAL_LAYERS and is_eval(t[side + "_id"]):
                out.add(t[side + "_id"])
    return out


def probe_rows(triples, labels, by_id):
    """The question "which parent does this place have", per level.

    An evaluation set derived from the same triples as everything else, so
    that it is pinned by the same revisions and the same digest rather than
    by a note somewhere saying which version was used.

    It measures recall of what the cpt table states, not generalisation. Every
    answer here appears in cpt by construction, and that is the point: the
    experiment is whether continued pretraining puts these facts into a model
    at all. A number from this set is not evidence about places outside it.
    """
    parents = collections.defaultdict(list)
    for t in triples:
        if not (t["truth"] and t["predicate"] == "sfWithin"):
            continue
        level = LEVELS.get((t["subject_layer"], t["object_layer"]))
        if level:
            parents[t["subject_id"]].append((level, t))

    out = []
    for child, found in sorted(parents.items()):
        # Two parents is no single answer. A place that lies in two
        # municipalities at once is a real thing, and scoring it either way
        # measures where the boundary was drawn rather than what the model
        # knows.
        if len(found) != 1:
            continue
        level, t = found[0]
        out.append({
            "child_id": child, "child_iri": t["subject_iri"],
            "child_en": labels.get((child, "en")),
            "child_ja": labels.get((child, "ja")),
            "parent_id": t["object_id"], "parent_iri": t["object_iri"],
            "parent_en": labels.get((t["object_id"], "en")),
            "parent_ja": labels.get((t["object_id"], "ja")),
            "level": level,
            "child_layer": t["subject_layer"],
            "parent_layer": t["object_layer"],
            "rcc8": t["rcc8"],
        })
    out.sort(key=lambda r: (r["level"], r["child_id"]))
    return mark_leaks(drop_self_answering(drop_ambiguous_names(out)))


# What a parent's name is once the kind of thing it is has been taken off.
# 福島市 -> 福島, so that 市立福島第三小学校 is seen to carry it.
JA_SUFFIX = "市区町村県府都道"
EN_SUFFIX = (" Prefecture", " City", " Town", " Village")


def bare_ja(name):
    return name[:-1] if name and name[-1] in JA_SUFFIX else name


def bare_en(name):
    for suffix in EN_SUFFIX:
        if name and name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def mark_split(rows, held_out):
    """Which side of the split each question is on.

    A question is eval when either of its two features is held out, not only
    its subject. If the parent were held out and the child were not, the fact
    would be missing from training while the question claimed to be a recall
    question, and the number would mean neither thing.
    """
    for r in rows:
        r["split"] = ("eval" if {r["child_id"], r["parent_id"]} & held_out
                      else "train")
    return rows


def mark_leaks(rows):
    """Flag the questions whose subject already names the answer.

    39.4% of the place questions are like "which municipality is
    市立福島第三小学校 in", where the answer 福島市 is inside the name. The
    question is still a fair one to ask, and a model that reads the answer off
    it is doing something useful, but it is not recall of anything this corpus
    taught. A 0.6B base model scores 30.8% on that level in Japanese and 15.8%
    in English, which is the leak and not its geography.

    Marked rather than dropped: dropping them would throw away two fifths of
    the level and quietly change what the level is about. With the columns
    here a score can be reported split, which is what the card does.
    """
    for r in rows:
        r["answer_in_child_ja"] = bool(
            r["parent_ja"] and r["child_ja"]
            and bare_ja(r["parent_ja"]) in r["child_ja"])
        r["answer_in_child_en"] = bool(
            r["parent_en"] and r["child_en"]
            and bare_en(r["parent_en"]) in r["child_en"])
    return rows


def drop_ambiguous_names(rows):
    """Questions whose subject names more than one place.

    21 of the places are called 天祖神社 and they are in different wards, so
    "which ward is 天祖神社 in" has 21 answers and any one of them scores a
    model on a coin toss. A place is kept only where its name picks it out
    among the questions of its level, in both languages it is asked in.

    Dropped rather than disambiguated. A name plus a ward would be a question
    containing its own answer.
    """
    seen = collections.defaultdict(collections.Counter)
    for r in rows:
        for lang in ("en", "ja"):
            if r["child_" + lang]:
                seen[(r["level"], lang)][r["child_" + lang]] += 1
    out = []
    for r in rows:
        if all(seen[(r["level"], lang)][r["child_" + lang]] == 1
               for lang in ("en", "ja") if r["child_" + lang]):
            out.append(r)
    return out


def drop_self_answering(rows):
    """Questions that contain their own answer.

    千代田 is a place inside 千代田区, and Aruba is a country with one state
    also called Aruba. "Which ward is Chiyoda in" is not a question about
    geography. 36 rows, and dropping them is cheaper than explaining in the
    card why a model scores a few points for free.

    Matched on the label in either language, not on the relation: the country
    cases are EQ and the ward cases are not, so filtering by RCC8 would catch
    one kind and leave the other.
    """
    return [r for r in rows
            if not (r["child_en"] and r["child_en"] == r["parent_en"])
            and not (r["child_ja"] and r["child_ja"] == r["parent_ja"])]


def probe_candidates(rows, layer_features):
    """How many parents a level is choosing between, for the chance rate.

    Each level chooses among every feature of its parent layer: 1,740
    municipalities and 47 prefectures. Deliberately generous, since a chance
    rate that is too high makes a result look less impressive than it is,
    which is the safe direction.
    """
    counts = {}
    for level in sorted({r["level"] for r in rows}):
        if level == "place-in-municipality":
            counts[level] = layer_features["jp-muni"]
        elif level == "municipality-in-prefecture":
            counts[level] = layer_features["jp-pref"]
        elif level == "state-in-country":
            # Every country in the layer, not only those that appear in
            # relations.tsv: three of the 258 are disjoint from everything
            # else and so are absent there, but a model naming a country is
            # still choosing among all of them.
            counts[level] = layer_features["ne-admin0"]
        else:
            raise SystemExit(f"no candidate count defined for {level}")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relations", required=True,
                    help="the oracle's relations.tsv")
    ap.add_argument("--oracle", required=True,
                    help="the oracle's manifest.json, for provenance")
    ap.add_argument("--graphs", nargs="*", default=None,
                    help="the oracle's .ttl files; default is beside the manifest")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    oracle = json.load(open(a.oracle, encoding="utf-8"))
    oracle_dir = os.path.dirname(os.path.abspath(a.oracle))
    graphs = a.graphs or [os.path.join(oracle_dir, s["ttl_file"])
                          for s in oracle["sources"]]

    # The oracle's manifest names a digest for each graph. Building against a
    # graph it does not describe would put a provenance block in the output
    # that quietly refers to different bytes, which is the one failure this
    # dataset's whole claim rests on not happening.
    check_graphs(oracle, graphs, a.relations)

    rows = read_relations(a.relations)
    iris, labels = feature_labels(graphs)
    verdicts = certify.read_vendored()
    lean = certify.vendored_revision()
    triples = direct_triples(rows, iris, verdicts)
    layer_of = {}
    for r in rows:
        layer_of[r["subject_id"]] = r["subject_layer"]
        layer_of[r["object_id"]] = r["object_layer"]
    composed, per_cell = composed_triples(rows, compared=was_compared(oracle),
                                          layer_of=layer_of)

    wrong = contradictions(composed)
    if wrong:
        first = wrong[0]
        raise SystemExit(
            f"{len(wrong):,} composed triples contradict the geometry, "
            f"for instance {first['a']} {first['r']} {first['b']} {first['s']} "
            f"{first['c']} entails {first['t']} but the oracle read "
            f"{first['observed']}")

    by_id = {}
    for r in rows:
        for side in ("subject", "object"):
            by_id.setdefault(r[f"{side}_id"],
                             (r[f"{side}_source"], r[f"{side}_layer"]))
    for c in composed:
        triples.append({
            "subject_id": c["a"], "subject_iri": iris[c["a"]],
            "subject_name": (labels.get((c["a"], "en"))
                             or labels.get((c["a"], "ja"))),
            "subject_source": by_id[c["a"]][0],
            "subject_layer": by_id[c["a"]][1],
            # Both regions, necessarily: a composed row exists only where RCC8
            # held on both premises, and RCC8 holds only between regions.
            "subject_kind": "area",
            "predicate": vocab.RCC8_TO_SF[c["t"]],
            "object_id": c["c"], "object_iri": iris[c["c"]],
            "object_name": (labels.get((c["c"], "en"))
                            or labels.get((c["c"], "ja"))),
            "object_source": by_id[c["c"]][0],
            "object_layer": by_id[c["c"]][1],
            "object_kind": "area",
            "truth": True,
            # The composition step is the proved part: this cell of the table
            # is one of the 64 LeanGeospatial derives from its own definitions
            # and shows equal to the published table. What is not proved is
            # the RCC8 label of either premise; those are readings the oracle
            # made, and the certificate says which half is which.
            "certification": "certified",
            "certificate": f"rcc8_composition:{c['r']} x {c['s']}",
            "de9im": c["observed_de9im"], "rcc8": c["t"],
            "derivation": "composition",
            "via_id": c["b"], "via_iri": iris[c["b"]],
            # The oracle measures this on a pair it looked at. A composed
            # triple was not looked at, so there is nothing to put here.
            "outside_ratio": None,
            "premise_a_predicate": vocab.RCC8_TO_SF[c["r"]],
            "premise_b_predicate": vocab.RCC8_TO_SF[c["s"]],
        })

    # Composed and observed rows can agree on subject, predicate and object,
    # and two composed rows can agree on everything but the intermediate, so
    # via_id is part of the key. Without it the order of equal rows would be
    # whatever order they were appended in.
    triples.sort(key=lambda t: (t["subject_id"], t["predicate"],
                                t["object_id"], t["derivation"],
                                t["via_id"] or ""))
    held_out = eval_features(triples)
    cpt = cpt_rows(triples, iris, labels, held_out, layer_of)
    probe = mark_split(probe_rows(triples, labels, by_id), held_out)
    layer_features = {s["layer"]: s["features"] for s in oracle["sources"]}

    os.makedirs(a.out, exist_ok=True)
    engine = {
        "oracle": "YuisekinGeoSPARQL",
        # The byte-for-byte claim is about the Parquet files, and Parquet
        # bytes are the writer's as much as the data's. A different pyarrow
        # or a different codec gives the same rows a different digest, so
        # both are recorded beside the digests rather than left implied.
        "parquet_writer": "pyarrow " + pa.__version__,
        "parquet_compression": "zstd",
        "oracle_relations_sha256": oracle["relations"]["sha256"],
        "endpoint": "Apache Jena Fuseki 6.2.0",
        "geometry": "GEOS via shapely, as the oracle computed it",
        "composition_table": "LeanGeospatial data/rcc8_known_table.tsv, "
                             "equal to its proved table by table_eq_published",
        # The oracle observes, the certifier proves, and this build derives.
        # Naming all three keeps a reader from reading a certified row as a
        # measured one or the other way about.
        "certifier": ("LeanGeospatial " + lean) if lean else None,
        "certifier_role": "certifies the step from a DE-9IM matrix to a "
                          "Simple Features predicate, and the composition "
                          "cells; it does not observe geometry and is not "
                          "needed to build this dataset",
        "schema_version": SCHEMA_VERSION,
    }
    dataset = ";".join(sorted(
        {s["dataset"] + "@" + s["revision"] for s in oracle["sources"]}))
    for t in triples:
        t["source_dataset"] = dataset
        t["engine_version"] = engine["endpoint"]
        t["schema_version"] = SCHEMA_VERSION

    write(triples, TRIPLE_FIELDS, os.path.join(a.out, "triples.parquet"), pa, pq)
    write(cpt, CPT_FIELDS, os.path.join(a.out, "cpt.parquet"), pa, pq)
    write(probe, PROBE_FIELDS, os.path.join(a.out, "probe.parquet"), pa, pq)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "licence": oracle["licence"],
        "sources": [{"dataset": s["dataset"], "layer": s["layer"],
                     "revision": s["revision"],
                     "licence": s["licence"], "features": s["features"]}
                    for s in oracle["sources"]],
        "triples": {
            "rows": len(triples),
            "pairs": len(rows),
            "predicates": list(vocab.SF),
            "true": sum(1 for t in triples if t["truth"]),
            "false": sum(1 for t in triples if not t["truth"]),
            "by_derivation": dict(collections.Counter(
                t["derivation"] for t in triples)),
            # A derived row whose conclusion the oracle also measured, against
            # one it never formed a pair for. The second kind is where the
            # composition table says something the geometry was never asked.
            "derived_beyond_what_was_measured": sum(
                1 for t in triples
                if t["derivation"] == "composition" and not t["de9im"]),
            "by_certification": dict(collections.Counter(
                t["certification"] for t in triples)),
            "true_by_predicate": dict(collections.Counter(
                t["predicate"] for t in triples if t["truth"])),
            "by_rcc8": dict(collections.Counter(
                r["rcc8_raw"] for r in rows)),
            "sha256": digest(os.path.join(a.out, "triples.parquet")),
        },
        "cpt": {
            "rows": len(cpt),
            "by_form": dict(collections.Counter(r["form"] for r in cpt)),
            "by_derivation": dict(collections.Counter(
                r["derivation"] for r in cpt)),
            "characters": sum(len(r["text"]) for r in cpt),
            "characters_by_form": {
                f: sum(len(r["text"]) for r in cpt if r["form"] == f)
                for f in vocab.FORMS},
            "by_topic": dict(collections.Counter(r["topic"] for r in cpt)),
            "by_pair": dict(collections.Counter(
                r["pair"] for r in cpt if r["form"] == "ja")),
            "by_topic_and_form": {
                f"{topic} {form}": n for (topic, form), n
                in sorted(collections.Counter(
                    (r["topic"], r["form"]) for r in cpt).items())},
            "held_out_rows": sum(1 for r in cpt if r["holdout"]),
            "held_out_features": len(held_out),
            "holdout_rule": (
                f"one feature in {EVAL_SHARE} of {' and '.join(EVAL_LAYERS)}, "
                "chosen by sha256 of its id. A row is held out when it names "
                "any of them, in any position"),
            "sha256": digest(os.path.join(a.out, "cpt.parquet")),
        },
        "probe": {
            "rows": len(probe),
            "by_level": dict(collections.Counter(r["level"] for r in probe)),
            "by_split": dict(collections.Counter(r["split"] for r in probe)),
            "by_level_and_split": {
                f"{level} {split}": n for (level, split), n
                in sorted(collections.Counter(
                    (r["level"], r["split"]) for r in probe).items())},
            "answerable_ja": sum(1 for r in probe
                                 if r["child_ja"] and r["parent_ja"]),
            "answerable_en": sum(1 for r in probe
                                 if r["child_en"] and r["parent_en"]),
            "answer_in_the_name": {
                lang: sum(1 for r in probe if r["answer_in_child_" + lang])
                for lang in ("ja", "en")},
            "candidates": probe_candidates(probe, layer_features),
            "by_rcc8": dict(collections.Counter(r["rcc8"] for r in probe)),
            "contamination": "every answer appears in the cpt table; this "
                             "set measures recall of that table, not "
                             "generalisation to places outside it",
            "sha256": digest(os.path.join(a.out, "probe.parquet")),
        },
        "composition_cells_used": {f"{r} x {s}": n
                                   for (r, s), n in sorted(per_cell.items())},
    }
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"{len(triples):,} triples  "
          f"({manifest['triples']['true']:,} true, "
          f"{manifest['triples']['false']:,} false)")
    for k, n in sorted(manifest["triples"]["by_derivation"].items()):
        print(f"    {k:12} {n:8,}")
    print(f"{len(cpt):,} cpt rows, {manifest['cpt']['characters']:,} characters")
    for k in vocab.FORMS:
        print(f"    {k:12} {manifest['cpt']['by_form'].get(k, 0):8,}")
    print(f"{len(cpt):,} cpt rows, {manifest['cpt']['characters']:,} characters"
          if False else "", end="")
    for k, n in sorted(manifest["cpt"]["by_topic_and_form"].items()):
        print(f"    {k:22} {n:9,}")
    print(f"{manifest['cpt']['held_out_rows']:,} cpt rows held out of "
          f"training ({manifest['cpt']['held_out_rows'] / len(cpt):.1%}), "
          f"naming {len(held_out):,} features")
    print(f"{len(probe):,} probe questions "
          f"({manifest['probe']['answerable_ja']:,} answerable in Japanese)")
    for k, n in sorted(manifest["probe"]["by_level"].items()):
        print(f"    {k:28} {n:6,}  "
              f"1 in {manifest['probe']['candidates'][k]:,}")
    for k, n in sorted(manifest["probe"]["by_level_and_split"].items()):
        print(f"      {k:34} {n:6,}")
    print(f"licence: {oracle['licence']['name']}")
    return 0


def check_graphs(oracle, graphs, relations_path):
    """Stop unless the oracle's manifest describes the files being read.

    A rebuild rewrites the graphs and relations.tsv at different moments, so
    a manifest left over from an earlier run is the ordinary way this goes
    wrong, not an exotic one.

    The graphs checked are the ones that will be read, not the ones beside
    the manifest, because --graphs can point somewhere else and a check of
    files nobody opened would be worse than no check.
    """
    want = {s["ttl_file"]: s["ttl_sha256"] for s in oracle["sources"]}
    stale = []
    for path in graphs:
        name = os.path.basename(path)
        if name not in want:
            stale.append(f"{name} (the manifest names no such graph)")
        elif digest(path) != want[name]:
            stale.append(name)
    if digest(relations_path) != oracle["relations"]["sha256"]:
        stale.append(os.path.basename(relations_path))
    if stale:
        raise SystemExit(
            "the oracle's manifest does not describe " + ", ".join(stale)
            + "; re-run the oracle before building")


# Declared rather than inferred. A column that is empty in one build and not
# in the next would otherwise change type, and two Parquet files that differ
# only in a column's type are not comparable byte for byte.
TRIPLE_FIELDS = (
    ("subject_id", "string"), ("subject_iri", "string"),
    ("subject_name", "string"), ("subject_source", "string"),
    ("subject_layer", "string"), ("subject_kind", "string"),
    ("predicate", "string"),
    ("object_id", "string"), ("object_iri", "string"),
    ("object_name", "string"), ("object_source", "string"),
    ("object_layer", "string"), ("object_kind", "string"),
    ("truth", "bool"), ("de9im", "string"), ("rcc8", "string"),
    ("certification", "string"), ("certificate", "string"),
    ("derivation", "string"), ("via_id", "string"), ("via_iri", "string"),
    ("outside_ratio", "float64"),
    ("source_dataset", "string"), ("engine_version", "string"),
    ("schema_version", "string"),
)

CPT_FIELDS = (
    ("text", "string"), ("form", "string"),
    ("subject_id", "string"), ("predicate", "string"),
    ("object_id", "string"),
    ("derivation", "string"), ("via_id", "string"),
    ("de9im", "string"), ("rcc8", "string"),
    ("certification", "string"), ("certificate", "string"),
    ("holdout", "bool"), ("topic", "string"), ("pair", "string"),
)

PROBE_FIELDS = (
    ("child_id", "string"), ("child_iri", "string"),
    ("child_en", "string"), ("child_ja", "string"),
    ("parent_id", "string"), ("parent_iri", "string"),
    ("parent_en", "string"), ("parent_ja", "string"),
    ("level", "string"),
    ("child_layer", "string"), ("parent_layer", "string"),
    ("rcc8", "string"),
    ("answer_in_child_ja", "bool"), ("answer_in_child_en", "bool"),
    ("split", "string"),
)

TRIPLE_COLUMNS = tuple(n for n, _ in TRIPLE_FIELDS)
CPT_COLUMNS = tuple(n for n, _ in CPT_FIELDS)


def write(rows, fields, path, pa, pq):
    types = {"string": pa.string(), "bool": pa.bool_(),
             "float64": pa.float64()}
    schema = pa.schema([pa.field(n, types[t]) for n, t in fields])
    cols = [pa.array([r.get(n) for r in rows], type=types[t])
            for n, t in fields]
    pq.write_table(pa.Table.from_arrays(cols, schema=schema), path,
                   compression="zstd")


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
