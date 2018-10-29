#include <assert.h>
#include <sstream>
#include <sys/time.h>
#include "buffered_stream/BufferedReader.hpp"
#include "buffered_stream/BufferedWriter.hpp"
#include "pattern_matching/PatternMatcher.hpp"
#include "CollectionStats.hpp"


template<typename T=uint32_t>
void testCollectionStatsDump_Basic() {
    T key(12);
    KeyPair<T> keyPair(12,15);
    KeyTriple<T> keyTriple(12,15,18);
    StatsKey statsKey(1,2,4);
    StatsKeyPair statsKeyPair(1,1,2,4,1);
    StatsKeyTriple statsKeyTriple(2,2,4,8,0);

    std::stringstream sstream;
    BufferedWriter<false> bufferedWriter(&sstream, 4096);
    bufferedWriter.put<T>(key);
    bufferedWriter.put<KeyPair<T>>(keyPair);
    bufferedWriter.put<KeyTriple<T>>(keyTriple);
    bufferedWriter.put<StatsKey>(statsKey);
    bufferedWriter.put<StatsKeyPair>(statsKeyPair);
    bufferedWriter.put<StatsKeyTriple>(statsKeyTriple);
    bufferedWriter.flush();

    BufferedReader<true> bufferedReader(&sstream, 4096, 0);
    bufferedReader.increase_num_bytes_constraint(
            sizeof(T) +
            sizeof(KeyPair<T>) +
            sizeof(KeyTriple<T>)
    );

    assert(std::equal_to<T>()(key, bufferedReader.get<T>()));
    assert(std::equal_to<KeyPair<T>>()(keyPair, bufferedReader.get<KeyPair<T>>()));
    assert(std::equal_to<KeyTriple<T>>()(keyTriple, bufferedReader.get<KeyTriple<T>>()));

    bufferedReader.increase_num_bytes_constraint(
            sizeof(statsKey) +
            sizeof(statsKeyPair) +
            sizeof(statsKeyTriple)
    );

    StatsKey statsKey2(bufferedReader.get<StatsKey>());
    StatsKeyPair statsKeyPair2(bufferedReader.get<StatsKeyPair>());
    StatsKeyTriple statsKeyTriple2(bufferedReader.get<StatsKeyTriple>());

    assert(statsKey.document_frequency == statsKey2.document_frequency);
    assert(statsKey.frequency == statsKey2.frequency);
    assert(statsKey.frequency_square == statsKey2.frequency_square);

    assert(statsKeyPair.document_frequency == statsKeyPair2.document_frequency);
    assert(statsKeyPair.window_document_frequency == statsKeyPair2.window_document_frequency);
    assert(statsKeyPair.window_frequency == statsKeyPair2.window_frequency);
    assert(statsKeyPair.window_frequency_square == statsKeyPair2.window_frequency_square);
    assert(statsKeyPair.window_min_dist == statsKeyPair2.window_min_dist);

    assert(statsKeyTriple.document_frequency == statsKeyTriple2.document_frequency);
    assert(statsKeyTriple.window_document_frequency == statsKeyTriple2.window_document_frequency);
    assert(statsKeyTriple.window_frequency == statsKeyTriple2.window_frequency);
    assert(statsKeyTriple.window_frequency_square == statsKeyTriple2.window_frequency_square);
    assert(statsKeyTriple.window_min_dist == statsKeyTriple2.window_min_dist);
}


template<typename Key, typename Value, class _HASH, class _PRED>
void _add_testCollectionStats(
        Key key,
        Value value,
        std::unordered_map<Key, Value, _HASH, _PRED> &stats
) {
    auto entry_it = stats.find(key);
    if (entry_it != stats.end()) {
        entry_it->second += value;
    } else {
        stats.insert({key, value});
    }
};


template<bool B_DISABLE_UNWINDOWED, bool B_RESTRICTED, typename T>
void _test_testCollectionStats(
        const CollectionStats<T, B_DISABLE_UNWINDOWED, B_RESTRICTED> &stats,
        const std::unordered_map<T, size_t> &stats_key,
        const std::unordered_map<KeyPair<T>, size_t> &stats_key_pair,
        const std::unordered_map<KeyTriple<T>, size_t> &stats_key_triple,
        size_t pairs_to_exclude, size_t triples_to_exclude,
        size_t key_frequency_sum, size_t key_pair_frequency_sum, size_t key_triple_frequency_sum,
        size_t round
) {
    assert(stats.get_num_docs() == round);

    if (B_RESTRICTED || round > 0) {
        assert(stats.get_num_keys() == stats_key.size());
        assert(stats.get_num_key_pairs() == (stats_key_pair.size() - pairs_to_exclude));
        assert(stats.get_num_key_triples() == (stats_key_triple.size() - triples_to_exclude));
    } else {
        assert(stats.get_num_keys() == 0);
        assert(stats.get_num_key_pairs() == 0);
        assert(stats.get_num_key_triples() == 0);
    }

    for (auto stats_key_it: stats_key) {
        const auto &statsTerm = stats.get_stats_key(stats_key_it.first);
        assert(statsTerm.document_frequency == round * (stats_key_it.second > 0));
        assert(statsTerm.frequency == round * stats_key_it.second);
    }

    for (auto stats_key_pair_it: stats_key_pair) {
        const auto &statsTermPair = stats.get_stats_key_pair(stats_key_pair_it.first);
        if (B_DISABLE_UNWINDOWED) {
            assert(statsTermPair.document_frequency == 0);
        } else {
            assert(statsTermPair.document_frequency == round);
        }
        assert(statsTermPair.window_document_frequency == round * (stats_key_pair_it.second > 0));
        assert(statsTermPair.window_frequency == round * stats_key_pair_it.second);
        assert(statsTermPair.window_frequency_square == round * stats_key_pair_it.second * stats_key_pair_it.second);
    }

    for (auto stats_key_triple_it: stats_key_triple) {
        const auto &statsTermTriple = stats.get_stats_key_triple(stats_key_triple_it.first);
        if (B_DISABLE_UNWINDOWED) {
            assert(statsTermTriple.document_frequency == 0);
        } else {
            assert(statsTermTriple.document_frequency == round);
        }
        assert(statsTermTriple.window_document_frequency == round * (stats_key_triple_it.second > 0));
        assert(statsTermTriple.window_frequency == round * stats_key_triple_it.second);
        assert(statsTermTriple.window_frequency_square ==
               round * stats_key_triple_it.second * stats_key_triple_it.second);
    }

    assert(stats.get_key_frequency_sum() == round * key_frequency_sum);
    assert(stats.get_key_pair_window_co_occ_sum() == round * key_pair_frequency_sum);
    assert(stats.get_key_triple_window_co_occ_sum() == round * key_triple_frequency_sum);
}


template<bool B_DISABLE_UNWINDOWED, bool B_RESTRICTED, typename T>
void testCollectionStats_impl(
        const distance_t window_size_key_pairs_co_occ,
        const distance_t window_size_key_triples_co_occ,
        const std::string &text,
        const PatternMatcher<T> &matcher,
        const size_t num_match_tests = 5,
        const std::unordered_set<T> &key_constraints = std::unordered_set<T>({}),
        const std::unordered_set<KeyPair<T>> &key_pair_constraints = std::unordered_set<KeyPair<T>>({}),
        const std::unordered_set<KeyTriple<T>> &key_triple_constraints = std::unordered_set<KeyTriple<T>>({})
) {
    using _CollectionStats = CollectionStats<T, B_DISABLE_UNWINDOWED, B_RESTRICTED>;
    using _CollectionStatsFiller = CollectionStatsFiller<T, B_DISABLE_UNWINDOWED, B_RESTRICTED>;

    // check the configuration before all
    assert(window_size_key_pairs_co_occ >= 0);
    assert(window_size_key_triples_co_occ >= 0);
    assert(!text.empty());
    assert(!matcher.get_pattern_set().empty());
    if (B_RESTRICTED) {
        assert (key_constraints.size() + key_pair_constraints.size() + key_triple_constraints.size() > 0);
    } else {
        assert (key_constraints.empty());
        assert (key_pair_constraints.empty());
        assert (key_triple_constraints.empty());
    }

    // create a list of matches and the map pattern to length
    PatternMatches<T> matches(true);
    matcher.find_patterns(text, matches);

    // test collection with and without restrictions
    _CollectionStats stats(window_size_key_pairs_co_occ, window_size_key_triples_co_occ);
    _CollectionStatsFiller filler(&stats, &matcher, 0, 1);
    if (B_RESTRICTED) {
        for (T key: key_constraints) {
            filler.add_restriction(key);
        }
        for (KeyPair<T> keyPair: key_pair_constraints) {
            filler.add_restriction(keyPair);
        }
        for (KeyTriple<T> keyTriple: key_triple_constraints) {
            filler.add_restriction(keyTriple);
        }
    }

    // start position of each match
    std::vector<size_t> start_positions(matches.size());
    for (size_t l = 0, end = matches.size(); l < end; ++l) {
        start_positions[l] = matches[l].end_pos + 1 - matcher.get_pattern_length(matches[l].pattern);
    }

    // fill local maps to check with the current implementation
    // EXHAUSTIVE SEARCH
    std::unordered_map<T, size_t> stats_key;
    std::unordered_map<KeyPair<T>, size_t> stats_key_pair;
    std::unordered_map<KeyTriple<T>, size_t> stats_key_triple;

    if (B_RESTRICTED) {
        for (T key: key_constraints) {
            _add_testCollectionStats(key, static_cast<size_t>(0), stats_key);
        }
        for (KeyPair<T> keyPair: key_pair_constraints) {
            _add_testCollectionStats(keyPair, static_cast<size_t>(0), stats_key_pair);
            if (keyPair.first() == keyPair.second()) {
                _add_testCollectionStats(keyPair.first(), static_cast<size_t>(0), stats_key);
            }
        }
        for (KeyTriple<T> keyTriple: key_triple_constraints) {
            _add_testCollectionStats(keyTriple, static_cast<size_t>(0), stats_key_triple);
            if (keyTriple.first() == keyTriple.second() == keyTriple.third()) {
                _add_testCollectionStats(keyTriple.first(), static_cast<size_t>(0), stats_key);
            } else if (keyTriple.first() == keyTriple.second()) {
                _add_testCollectionStats(KeyPair<T>(keyTriple.first(), keyTriple.third()), static_cast<size_t>(0),
                                         stats_key_pair);
            } else if (keyTriple.second() == keyTriple.third()) {
                _add_testCollectionStats(KeyPair<T>(keyTriple.second(), keyTriple.first()), static_cast<size_t>(0),
                                         stats_key_pair);
            }
        }
    }

    size_t key_frequency_sum = 0;
    size_t key_pair_frequency_sum = 0;
    size_t key_triple_frequency_sum = 0;

    for (size_t l = 0, end = matches.size(); l < end; ++l) {
        if (!B_RESTRICTED || stats_key.count(matches[l].pattern)) {
            _add_testCollectionStats(matches[l].pattern, static_cast<size_t>(1), stats_key);
            key_frequency_sum += 1;
        }

        for (size_t r = l + 1; r < end; ++r) {
            if (matches[l].end_pos >= start_positions[r]) { // end_pos is always included
                continue;
            }

            size_t window_size = matches[r].end_pos + 1 - start_positions[l];
            if (window_size <= window_size_key_pairs_co_occ) {
                KeyPair<T> keyPair(matches[l].pattern, matches[r].pattern);
                if (!B_RESTRICTED || stats_key_pair.count(keyPair)) {
                    _add_testCollectionStats(keyPair, static_cast<size_t>(1), stats_key_pair);
                    key_pair_frequency_sum += 1;
                }
            }

            for (size_t m = l + 1; m < r; ++m) {
                if (matches[l].end_pos >= start_positions[m] || matches[m].end_pos >= start_positions[r]) {
                    continue;
                }
                if (window_size <= window_size_key_triples_co_occ) {
                    KeyTriple<T> keyTriple(matches[l].pattern, matches[m].pattern, matches[r].pattern);
                    if (!B_RESTRICTED || stats_key_triple.count(keyTriple)) {
                        _add_testCollectionStats(keyTriple, static_cast<size_t>(1), stats_key_triple);
                        key_triple_frequency_sum += 1;
                    }
                }
            }
        }
    }

    // add all the possible pairs and triples to remember about the document frequency
    if (!B_DISABLE_UNWINDOWED) {
        for (size_t l = 0, end = matches.size(); l < end; ++l) {
            for (size_t r = l + 1; r < end; ++r) {
                KeyPair<T> keyPair(matches[l].pattern, matches[r].pattern);
                if (!B_RESTRICTED || stats_key_pair.count(keyPair)) {
                    // add a fake entry to remember that this pair appears inside the text
                    _add_testCollectionStats(keyPair, static_cast<size_t>(0), stats_key_pair);
                }

                for (size_t m = l + 1; m < r; ++m) {
                    KeyTriple<T> keyTriple(keyPair, matches[m].pattern);
                    if (!B_RESTRICTED || stats_key_triple.count(keyTriple)) {
                        // add a fake entry to remember that this pair appears inside the text
                        _add_testCollectionStats(keyTriple, static_cast<size_t>(0), stats_key_triple);
                    }
                }
            }
        }
    }
    // count the number of pair/triples with identical elements
    size_t pairs_to_exclude = 0;
    size_t triples_to_exclude = 0;
    if (!B_RESTRICTED) {
        for (auto pair_entry: stats_key_pair) {
            // exclude the pairs that would be automatically included because in some window
            if (pair_entry.second > 0) {
                continue;
            }
            if (pair_entry.first.first() == pair_entry.first.second()) {
                pairs_to_exclude += 1;
            }
        }
        for (auto triple_entry: stats_key_triple) {
            // exclude the triples that would be automatically included because in some window
            if (triple_entry.second > 0) {
                continue;
            }
            if (triple_entry.first.first() == triple_entry.first.second() ||
                triple_entry.first.second() == triple_entry.first.third()) {
                triples_to_exclude += 1;
            }
        }
    }

//    stats.reserve(100, 1000, 10000);
    _test_testCollectionStats(
            stats,
            stats_key, stats_key_pair, stats_key_triple,
            pairs_to_exclude, triples_to_exclude,
            key_frequency_sum, key_pair_frequency_sum, key_triple_frequency_sum,
            0
    );

    for (size_t round = 1; round <= num_match_tests; ++round) {
        //stats.update(matcher.get_pattern_length_map(), &matches, (&matches) + 1);
        for (size_t i=0; i < round; ++i) {
            filler.update({text});
        }
        filler.flush();

        _test_testCollectionStats(
                stats,
                stats_key, stats_key_pair, stats_key_triple,
                pairs_to_exclude, triples_to_exclude,
                key_frequency_sum, key_pair_frequency_sum, key_triple_frequency_sum,
                round * (round + 1) / 2
        );
    }

    // dump - load test
    std::stringstream sstream;
    BufferedWriter<false> bufferedWriter(&sstream, 4096);
    stats.dumps(bufferedWriter);
    bufferedWriter.flush();

    BufferedReader<true> bufferedReader(&sstream, 4096, 0);
    _CollectionStats *stats_p = _CollectionStats::loads(bufferedReader);

    _test_testCollectionStats(
            *stats_p,
            stats_key, stats_key_pair, stats_key_triple,
            pairs_to_exclude, triples_to_exclude,
            key_frequency_sum, key_pair_frequency_sum, key_triple_frequency_sum,
            num_match_tests * (num_match_tests + 1) / 2
    );

    delete stats_p;

    // clear test
    stats.clear();
    assert(stats.get_num_docs() == 0);
    assert(stats.get_num_keys() == 0);
    assert(stats.get_num_key_pairs() == 0);
    assert(stats.get_num_key_triples() == 0);
    assert(stats.get_key_frequency_sum() == 0);
    assert(stats.get_key_pair_window_co_occ_sum() == 0);
    assert(stats.get_key_triple_window_co_occ_sum() == 0);
}


template<typename T=uint16_t>
void testCollectionStats() {
    const size_t num_chars = 10;
    const size_t seq_n_repetitions = 3 * 3;
    const size_t num_restrictions = 3;

    static_assert(num_chars > 0, "num_chars must be greater than 0");
    static_assert(seq_n_repetitions > 0, "seq_n_repetitions must be greater than 0");
    static_assert(seq_n_repetitions % 3 == 0, "seq_n_repetitions must be a multiple of 3");
    static_assert(num_restrictions > 0, "num_restrictions must be greater than 0");

    // create a list of matches and the map pattern to length
    char text[seq_n_repetitions * num_chars * 2];
    for (size_t n = 0, i = 0; n < seq_n_repetitions; ++n) {
        for (size_t ci = 0; ci < num_chars; ++ci, i += 2) {
            text[i] = 'a' + (char) ci;
            text[i + 1] = ' ';
        }
    }
    PatternMatcher<T> matcher;
    for (T ci = 0; ci < num_chars; ++ci) {
        char str[2] = {(char) ('a' + ci), '\0'};
        matcher.add_pattern(ci, std::string(str));
    }
    matcher.compile();

    // build restrictions
    std::unordered_set<T> key_constraints;
    std::unordered_set<KeyPair<T>> key_pair_constraints;
    std::unordered_set<KeyTriple<T>> key_triple_constraints;

    for (T i = 0; i < std::min(num_restrictions, num_chars); ++i) {
        T ci = (ci) % num_chars;
        T cj = ((ci + 3) % num_chars);
        T ck = ((ci + 5) % num_chars);

        key_constraints.insert(ck);
        key_pair_constraints.insert({ci, ck});
        key_triple_constraints.insert({ci, cj, ck});

        key_pair_constraints.insert({ci, cj});
        key_triple_constraints.insert({ci, cj, ci});
        key_triple_constraints.insert({ci, cj, cj});
    }

    std::cout << "\rTest w/o constraints 1 " << std::flush;
    testCollectionStats_impl<false, false>(num_chars * 2, num_chars * 3, text, matcher, 5);
    std::cout << "\rTest w/o constraints 2 " << std::flush;
    testCollectionStats_impl<true, false>(num_chars * 2, num_chars * 3, text, matcher, 5);
    std::cout << "\rTest w/o constraints 3 " << std::flush;
    testCollectionStats_impl<false, false>(12, 15, text, matcher, 5);
    std::cout << "\rTest w/o constraints 4 " << std::flush;
    testCollectionStats_impl<true, false>(12, 15, text, matcher, 5);
    std::cout << "\rTest w/o constraints 5 " << std::flush;
    testCollectionStats_impl<false, false>(12, 0, text, matcher, 5);
    std::cout << "\rTest w/o constraints 6 " << std::flush;
    testCollectionStats_impl<true, false>(12, 0, text, matcher, 5);
    std::cout << "\rTest w/o constraints 7 " << std::flush;
    testCollectionStats_impl<false, false>(0, 15, text, matcher, 5);
    std::cout << "\rTest w/o constraints 8 " << std::flush;
    testCollectionStats_impl<true, false>(0, 15, text, matcher, 5);
    std::cout << "\rTest w/o constraints 9 " << std::flush;
    testCollectionStats_impl<false, false>(0, 0, text, matcher, 5);
    std::cout << "\rTest w/o constraints 10 " << std::flush;
    testCollectionStats_impl<true, false>(0, 0, text, matcher, 5);

    std::cout << "\rTesting w constraints 1 " << std::flush;
    testCollectionStats_impl<false, true>(num_chars * 2, num_chars * 3, text, matcher, 5, key_constraints, key_pair_constraints, key_triple_constraints);
    std::cout << "\rTesting w constraints 2 " << std::flush;
    testCollectionStats_impl<true, true>(num_chars * 2, num_chars * 3, text, matcher, 5, key_constraints, key_pair_constraints, key_triple_constraints);
    std::cout << "\rTesting w constraints 3 " << std::flush;
    testCollectionStats_impl<false, true>(12, 15, text, matcher, 5, key_constraints, key_pair_constraints, key_triple_constraints);
    std::cout << "\rTesting w constraints 4 " << std::flush;
    testCollectionStats_impl<true, true>(12, 15, text, matcher, 5, key_constraints, key_pair_constraints, key_triple_constraints);
    std::cout << "\rTesting w constraints 5 " << std::flush;
    testCollectionStats_impl<false, true>(12, 0, text, matcher, 5, key_constraints, key_pair_constraints);
    std::cout << "\rTesting w constraints 6 " << std::flush;
    testCollectionStats_impl<true, true>(12, 0, text, matcher, 5, key_constraints, key_pair_constraints);
    std::cout << "\rTesting w constraints 7 " << std::flush;
    testCollectionStats_impl<false, true>(0, 15, text, matcher, 5, key_constraints, std::unordered_set<KeyPair<T>>({}), key_triple_constraints);
    std::cout << "\rTesting w constraints 8 " << std::flush;
    testCollectionStats_impl<true, true>(0, 15, text, matcher, 5, key_constraints, std::unordered_set<KeyPair<T>>({}), key_triple_constraints);
    std::cout << "\rTesting w constraints 9 " << std::flush;
    testCollectionStats_impl<false, true>(0, 0, text, matcher, 5, key_constraints);
    std::cout << "\rTesting w constraints 10 " << std::flush;
    testCollectionStats_impl<true, true>(0, 0, text, matcher, 5, key_constraints);
    std::cout << "\r                        \r";
}


int main(int argc, char **argv) {
    std::cout << "1) testCollectionStatsDump_Basic" << std::endl;
    testCollectionStatsDump_Basic();
    std::cout << "2) testCollectionStats" << std::endl;
    testCollectionStats();

    // TODO test dumps and loads

    return 0;
}
