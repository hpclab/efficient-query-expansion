import pattern.en
import numpy as np

from normalize_text import normalize_text
from pattern_matching.segmenter import PySegmenter


class QueryExpansionSupport(object):
    def __init__(
        self,
        expansion_support,
        good_unary_terms,
        stopwords,
        segment_to_phrase_freq,
        segment_to_and_freq
    ):
        assert isinstance(expansion_support, dict)
        assert all(entry in expansion_support for entry in ["segment_id_to_segment", "entity_id_to_tags_segment_id_list", "segment_id_to_entity_id_tags_list", "segment_id_to_meaning_id_list", "meaning_id_to_pos_segment_id_list"])
        assert all(isinstance(term, str) for term in good_unary_terms)
        assert all(isinstance(term, str) for term in stopwords)
        assert all(isinstance(segment, str) and isinstance(df, (int, long)) for segment, df in segment_to_phrase_freq.iteritems())
        assert all(isinstance(segment, str) and isinstance(df, (int, long)) for segment, df in segment_to_and_freq.iteritems())

        self._good_unary_terms = frozenset(good_unary_terms)
        segments_thesaurus = frozenset(expansion_support["segment_id_to_segment"])
        self._query_segmenter = PySegmenter(
            set(
                segment
                for segment in segment_to_phrase_freq
                if " " in segment and segment in segments_thesaurus
            ),
            segment_to_phrase_freq,
            segment_to_and_freq,
            -1.0,  # min_probability
            100  # min document frequency
        )
        self._stopwords = frozenset(stopwords)

        self._segment_id_to_segment = expansion_support["segment_id_to_segment"]
        self._segment_to_segment_id = dict((segment, segment_id) for segment_id, segment in enumerate(self._segment_id_to_segment))
        self._entity_id_to_tags_segment_id_list = expansion_support["entity_id_to_tags_segment_id_list"]
        self._segment_id_to_entity_id_tags_list = expansion_support["segment_id_to_entity_id_tags_list"]
        self._segment_id_to_meaning_id_list = expansion_support["segment_id_to_meaning_id_list"]
        self._meaning_id_to_pos_segment_id_list = expansion_support["meaning_id_to_pos_segment_id_list"]

        self._pos_to_lemma_to_segment_id_set = self._get_pos_to_lemma_to_segment_id_set()
        self._collapsed_segment_to_segment_id_list = self._get_collapsed_segment_to_segment_id_list()

    @staticmethod
    def _term_to_lemma(term, pos):
        if " " in term:
            lemma = ' '.join(pattern.en.lemma(t) or t for t in term.split())
        else:
            lemma = pattern.en.lemma(term)
        return str(lemma).strip()

    @staticmethod
    def _term_to_plural(term, pos):
        return str(pattern.en.pluralize(term, pos)).strip()

    def _get_pos_to_lemma_to_segment_id_set(self):
        pos_to_lemma_to_segment_id_set = {'adj': {}, 'adv': {}, 'noun': {}, 'verb': {}}

        for segment_id, meaning_id_list in self._segment_id_to_meaning_id_list.iteritems():
            term = self._segment_id_to_segment[segment_id]
            # iterate over the possible meanings and take the pos tags
            for pos in set(self._meaning_id_to_pos_segment_id_list[meaning_id][0] for meaning_id in meaning_id_list):
                if pos not in pos_to_lemma_to_segment_id_set:
                    continue
                lemma = QueryExpansionSupport._term_to_lemma(term, pos)

                # update the dictionaries
                if lemma in pos_to_lemma_to_segment_id_set[pos]:
                    pos_to_lemma_to_segment_id_set[pos][lemma].add(segment_id)
                else:
                    pos_to_lemma_to_segment_id_set[pos][lemma] = set([segment_id])
        return pos_to_lemma_to_segment_id_set

    def _get_collapsed_segment_to_segment_id_list(self):
        collapsed_segment_to_segment_id_list = dict()

        for segment_id in xrange(max(
            len(self._segment_id_to_entity_id_tags_list),
            1
        )):
            segment = self._segment_id_to_segment[segment_id]
            if " " in segment:
                new_segment = segment.replace(" ", "")
                if new_segment in self._segment_to_segment_id:
                    continue

                if new_segment in collapsed_segment_to_segment_id_list:
                    collapsed_segment_to_segment_id_list[new_segment] += (segment_id,)
                else:
                    collapsed_segment_to_segment_id_list[new_segment] = (segment_id,)
        return collapsed_segment_to_segment_id_list

    def _is_good_expansion(self, expansion):
        if " " in expansion:
            return all((term in self._good_unary_terms) for term in expansion.split())
        else:
            return expansion in self._good_unary_terms

    @staticmethod
    def _filter_expansions(term_tags_list, query_terms_set):
        return [
            term_tags
            for term_tags in term_tags_list
            if term_tags[0] not in query_terms_set
        ]

    @staticmethod
    def _group_or_terms(or_term):
        term_to_tags = dict()
        for term, tags in or_term:
            if term not in term_to_tags:
                term_to_tags[term] = tags
            else:
                term_to_tags[term] += tuple(tag for tag in tags if tag not in term_to_tags[term])

        return list(term_to_tags.iteritems())

    @staticmethod
    def _get_source_term(term):
        return (term,)

    def _get_thesaurus_expansions_part1(self, term):
        pos_set = self._pos_to_lemma_to_segment_id_set.keys()

        # get the LEMMA for each possible pos tag
        pos_to_lemma = dict(
            (pos, QueryExpansionSupport._term_to_lemma(term, pos))
            for pos in pos_set
        )
        # filtering unlikely lemmas
        if False:
            for pos in pos_set:
                if pos_to_lemma[pos] not in self._pos_to_lemma_to_segment_id_set[pos]:
                    del pos_to_lemma[pos]

        # use the lemma only if this term doesn't appear in our segments
        if False and term in self._segment_to_segment_id:
            segment_id = self._segment_to_segment_id[term]
            if segment_id in self._segment_id_to_meaning_id_list:
                meaning_pos_set = set(
                    self._meaning_id_to_pos_segment_id_list[meaning_id][0]
                    for meaning_id in self._segment_id_to_meaning_id_list[segment_id]
                )
            else:
                meaning_pos_set = set()

            segment_id_set = set([segment_id])
            pos_to_normalized_segment_id_set = dict(
                (pos, segment_id_set if pos in meaning_pos_set else set())
                for pos in pos_set
            )
        else:
            # find possible NORMALIZED versions of the lemma for each pos tag
            pos_to_normalized_segment_id_set = dict(
                #(pos, self._pos_to_lemma_to_segment_id_set[pos][lemma])
                (pos, self._pos_to_lemma_to_segment_id_set[pos][lemma] if lemma in self._pos_to_lemma_to_segment_id_set[pos] else set())
                for (pos, lemma) in pos_to_lemma.iteritems()
            )

        pos_to_normalized_term_set = dict(
            (pos, set(self._segment_id_to_segment[segment_id] for segment_id in normalized_segment_id_list))
            for (pos, normalized_segment_id_list) in pos_to_normalized_segment_id_set.iteritems()
        )

        # get the SYNONYMS of each normalized version
        pos_to_synset = dict()
        for pos, normalized_segment_id_list in pos_to_normalized_segment_id_set.iteritems():
            pos_to_synset[pos] = set(
                self._segment_id_to_segment[segment_id]
                for normalized_segment_id in normalized_segment_id_list
                for meaning_id in self._segment_id_to_meaning_id_list[normalized_segment_id]
                    if normalized_segment_id in self._segment_id_to_meaning_id_list
                    and pos == self._meaning_id_to_pos_segment_id_list[meaning_id][0]
                for segment_id in self._meaning_id_to_pos_segment_id_list[meaning_id][1]
                    if (" " + self._segment_id_to_segment[normalized_segment_id] + " ") not in (" " + self._segment_id_to_segment[segment_id] + " ")  # discard synonyms that extend the starting term with additional terms
            )

        # get the PLURALS of the normalized terms and their synonyms (which should be in the singular form)
        terms_to_pluralize = set()
        if "noun" in pos_to_normalized_term_set:
            terms_to_pluralize.update(pos_to_normalized_term_set["noun"])
        if "noun" in pos_to_synset:
            terms_to_pluralize.update(pos_to_synset["noun"])

        noun_plurals = set(
            QueryExpansionSupport._term_to_plural(new_term, "noun")
            for new_term in terms_to_pluralize
        )

        # put all togheter
        res = QueryExpansionSupport._group_or_terms(
            [
                (lemma, (pos, "Lem"))
                for (pos, lemma) in pos_to_lemma.iteritems()
            ] + [
                (normalized_term, (pos, "Norm"))
                for (pos, normalized_terms_set) in pos_to_normalized_term_set.iteritems()
                for normalized_term in normalized_terms_set
            ] + [
                (synonym, (pos, "Syn"))
                for (pos, synonyms_set) in pos_to_synset.iteritems()
                for synonym in synonyms_set
            ] + [
                (noun_plural, ("noun", "Plu"))
                for noun_plural in noun_plurals
            ]
        )

        return [
            (synonym, tags)
            for (synonym, tags) in res
            if (" " + term + " ") not in (" " + synonym + " ")  # remove synonyms that contains the original term
                and self._is_good_expansion(synonym)
        ]

    def _get_thesaurus_expansions_part2_support(self, segment_id):
        if segment_id >= len(self._segment_id_to_entity_id_tags_list):
            return []

        res = [
            (self._segment_id_to_segment[new_segment_id], self._entity_id_to_tags_segment_id_list[entity_id][0] + tags)
            for entity_id, tags in self._segment_id_to_entity_id_tags_list[segment_id]
            for new_segment_id in self._entity_id_to_tags_segment_id_list[entity_id][1]
        ]

        segment_src = self._segment_id_to_segment[segment_id]
        return [
            (segment, tags)
            for segment, tags in res
            if (" " + segment_src + " ") not in (" " + segment + " ")  # remove synonyms that contains the original term
        ]

    def _get_thesaurus_expansions_part2(self, segment):
        segment_id = self._segment_to_segment_id.get(segment, None)

        if segment_id is None:
            if " " not in segment and segment in self._collapsed_segment_to_segment_id_list:
                # TEMP CODE
                return sum([
                    self._get_thesaurus_expansions_part2_support(new_segment_id)
                    for new_segment_id in self._collapsed_segment_to_segment_id_list[segment]
                ], [])

            return []

        return self._get_thesaurus_expansions_part2_support(segment_id)

    def _remove_stopwords(self, query):
        # create a backup of the query
        query_backup = query
        # remove the stop words according to if they belong to some entity or not
        query = filter(
            (lambda x: x not in self._stopwords),
            (self._query_segmenter.segment(query) if self._query_segmenter else query.split())
        )

        # ACK: if the query is composed only by stopwords use all the terms as query
        if len(query) == 0:
            query = query_backup
        else:
            # discard the previous segmentation
            query = " ".join(query)

        return query

    def get_all_theraurus_expansions(self, query):
        # normalize the text
        query = normalize_text(query)

        # remove the stop words according to if they belong to some entity or not
        query = self._remove_stopwords(query)

        # segment the query using the thesaurus
        query_terms = self._query_segmenter.segment(query)

        query_terms_set = frozenset(query_terms)

        base_query = map(
            (lambda t: [QueryExpansionSupport._get_source_term(t)]),
            query_terms
        )

        # create the synsets
        candidates = map(
            (lambda t: QueryExpansionSupport._filter_expansions(
                QueryExpansionSupport._group_or_terms(
                    self._get_thesaurus_expansions_part1(t) + self._get_thesaurus_expansions_part2(t)),
                    query_terms_set
                )
            ),
            query_terms
        )

        # the expanded query is composed only of one segmentation
        # In case of more segmentations the two arrays contain more CNF queries
        return [base_query], [candidates]
