#ifndef COLLECTION_STATS_HPP
#define COLLECTION_STATS_HPP

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <parallel/algorithm>

#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

#include "pattern_matching/AhoCorasickAutomaton.hpp"
#include "buffered_stream/BufferedReader.hpp"
#include "buffered_stream/BufferedWriter.hpp"
#include "pattern_matching/PatternMatcher.hpp"

typedef uint64_t key_frequency_t;
typedef uint32_t document_frequency_t;
typedef uint16_t distance_t;


template<typename KeyType>
class KeyPair {
private:
    KeyType _first;
    KeyType _second;

public:
    KeyPair(
            const KeyType &first,
            const KeyType &second
    ) {
        if (first < second) {
            this->_first = first;
            this->_second = second;
        } else {
            this->_first = second;
            this->_second = first;
        }
    }

    KeyPair(
            const KeyPair &keyPair
    ) :
            _first(keyPair._first),
            _second(keyPair._second) {
    }

    inline const KeyType &
    first() const {
        return this->_first;
    }

    inline const KeyType &
    second() const {
        return this->_second;
    }
};


template<typename KeyType>
class KeyTriple {
private:
    KeyType _first;
    KeyType _second;
    KeyType _third;

public:
    KeyTriple(
            const KeyType &first,
            const KeyType &second,
            const KeyType &third
    ) {
        if (first < second) {
            // _first < _second
            if (second < third) {
                // _first < _second < _third
                this->_first = first;
                this->_second = second;
                this->_third = third;
            } else if (first < third) {
                // _first < _third < _second
                this->_first = first;
                this->_second = third;
                this->_third = second;
            } else {
                // _third < _first < _second
                this->_first = third;
                this->_second = first;
                this->_third = second;
            }
        } else {
            // _second < _first
            if (first < third) {
                // _second < _first < _third
                this->_first = second;
                this->_second = first;
                this->_third = third;
            } else if (second < third) {
                // _second < _third < _first
                this->_first = second;
                this->_second = third;
                this->_third = first;
            } else {
                // _third < _second < _first
                this->_first = third;
                this->_second = second;
                this->_third = first;
            }
        }
    }

    KeyTriple(
            const KeyPair<KeyType> &keyPair,
            const KeyType &other
    ) {
        if (other < keyPair.first()) {
            this->_first = other;
            this->_second = keyPair.first();
            this->_third = keyPair.second();
        } else if (keyPair.second() < other) {
            this->_first = keyPair.first();
            this->_second = keyPair.second();
            this->_third = other;
        } else {
            this->_first = keyPair.first();
            this->_second = other;
            this->_third = keyPair.second();
        }
    }

    KeyTriple(
            const KeyTriple &keyTriple
    ) :
            _first(keyTriple._first),
            _second(keyTriple._second),
            _third(keyTriple._third) {
    }

    inline const KeyType &
    first() const {
        return this->_first;
    }

    inline const KeyType &
    second() const {
        return this->_second;
    }

    inline const KeyType &
    third() const {
        return this->_third;
    }
};

// extend the "hash", "equal_to" and "less" implementations to KeyPair and KeyTriple
namespace std {
    template<typename _Tp>
    struct hash<KeyPair<_Tp>> : public __hash_base<size_t, KeyPair<_Tp>> {
        size_t
        operator()(const KeyPair<_Tp> &tp) const noexcept {
            return (std::hash<_Tp>()(tp.first()) ^ (std::hash<_Tp>()(tp.second()) << 1));
        }
    };

    template<typename _Tp>
    struct hash<KeyTriple<_Tp>> : public __hash_base<size_t, KeyTriple<_Tp>> {
        size_t
        operator()(const KeyTriple<_Tp> &tt) const noexcept {
            return ((std::hash<_Tp>()(tt.first()) ^ (std::hash<_Tp>()(tt.second()) << 1)) >> 1) ^
                   (std::hash<_Tp>()(tt.third()) << 1);
        }
    };

    template<typename _Tp>
    struct equal_to<KeyPair<_Tp>> : public binary_function<KeyPair<_Tp>, KeyPair<_Tp>, bool> {
        bool
        operator()(const KeyPair<_Tp> &tp1, const KeyPair<_Tp> &tp2) const noexcept {
            return std::equal_to<_Tp>()(tp1.first(), tp2.first()) && std::equal_to<_Tp>()(tp1.second(), tp2.second());
        }
    };

    template<typename _Tp>
    struct equal_to<KeyTriple<_Tp>> : public binary_function<KeyTriple<_Tp>, KeyTriple<_Tp>, bool> {
        bool
        operator()(const KeyTriple<_Tp> &tt1, const KeyTriple<_Tp> &tt2) const noexcept {
            return std::equal_to<_Tp>()(tt1.first(), tt2.first()) &&
                   std::equal_to<_Tp>()(tt1.second(), tt2.second()) &&
                   std::equal_to<_Tp>()(tt1.third(), tt2.third());
        }
    };

    template<typename _Tp>
    struct less<KeyPair<_Tp>> : public binary_function<KeyPair<_Tp>, KeyPair<_Tp>, bool> {
        bool
        operator()(const KeyPair<_Tp> &tp1, const KeyPair<_Tp> &tp2) const noexcept {
            return std::less<_Tp>()(tp1.first(), tp2.first()) ||
                   (std::equal_to<_Tp>()(tp1.first(), tp2.first()) &&
                    std::less<_Tp>()(tp1.second(), tp2.second()));
        }
    };

    template<typename _Tp>
    struct less<KeyTriple<_Tp>> : public binary_function<KeyTriple<_Tp>, KeyTriple<_Tp>, bool> {
        bool
        operator()(const KeyTriple<_Tp> &tt1, const KeyTriple<_Tp> &tt2) const noexcept {
            return std::less<_Tp>()(tt1.first(), tt2.first()) ||
                   (std::equal_to<_Tp>()(tt1.first(), tt2.first()) &&
                    (std::less<_Tp>()(tt1.second(), tt2.second()) ||
                     (std::equal_to<_Tp>()(tt1.second(), tt2.second()) &&
                      std::less<_Tp>()(tt1.third(), tt2.third()))));
        }
    };
}


class StatsKey {
public:
    document_frequency_t document_frequency;
    key_frequency_t frequency;
    key_frequency_t frequency_square;

    StatsKey() {
        this->document_frequency = 0;
        this->frequency = this->frequency_square = 0;
    }

    StatsKey(const StatsKey &other) {
        this->document_frequency = other.document_frequency;
        this->frequency = other.frequency;
        this->frequency_square = other.frequency_square;
    }

    StatsKey(
            document_frequency_t document_frequency,
            key_frequency_t frequency,
            key_frequency_t frequency_square
    ) {
        this->document_frequency = document_frequency;
        this->frequency = frequency;
        this->frequency_square = frequency_square;
    }

    inline void
    update(const StatsKey &other) {
        this->document_frequency += other.document_frequency;
        this->frequency += other.frequency;
        this->frequency_square += other.frequency_square;
    }
};


class StatsKeyPair {
public:
    document_frequency_t document_frequency;
    document_frequency_t window_document_frequency;
    key_frequency_t window_frequency;
    key_frequency_t window_frequency_square;
    distance_t window_min_dist;

    StatsKeyPair() {
        this->document_frequency = this->window_document_frequency = 0;
        this->window_frequency = this->window_frequency_square = 0;
        this->window_min_dist = (distance_t) -1;
    };

    StatsKeyPair(const StatsKeyPair &other) {
        this->document_frequency = other.document_frequency;
        this->window_document_frequency = other.window_document_frequency;
        this->window_frequency = other.window_frequency;
        this->window_frequency_square = other.window_frequency_square;
        this->window_min_dist = other.window_min_dist;
    }

    StatsKeyPair(
            document_frequency_t document_frequency,
            document_frequency_t window_document_frequency,
            key_frequency_t window_frequency,
            key_frequency_t window_frequency_square,
            distance_t window_min_dist
    ) {
        this->document_frequency = document_frequency;
        this->window_document_frequency = window_document_frequency;
        this->window_frequency = window_frequency;
        this->window_frequency_square = window_frequency_square;
        this->window_min_dist = window_min_dist;
    }

    inline void
    update(const StatsKeyPair &other) {
        this->document_frequency += other.document_frequency;
        this->window_document_frequency += other.window_document_frequency;
        this->window_frequency += other.window_frequency;
        this->window_frequency_square += other.window_frequency_square;
        if (this->window_min_dist > other.window_min_dist) {
            this->window_min_dist = other.window_min_dist;
        }
    }
};


class StatsKeyTriple {
public:
    document_frequency_t document_frequency;
    document_frequency_t window_document_frequency;
    key_frequency_t window_frequency;
    key_frequency_t window_frequency_square;
    distance_t window_min_dist;

    StatsKeyTriple() {
        this->document_frequency = this->window_document_frequency = 0;
        this->window_frequency = this->window_frequency_square = 0;
        this->window_min_dist = (distance_t) -1;
    }

    StatsKeyTriple(const StatsKeyTriple &other) {
        this->document_frequency = other.document_frequency;
        this->window_document_frequency = other.window_document_frequency;
        this->window_frequency = other.window_frequency;
        this->window_frequency_square = other.window_frequency_square;
        this->window_min_dist = other.window_min_dist;
    }

    StatsKeyTriple(
            document_frequency_t document_frequency,
            document_frequency_t window_document_frequency,
            key_frequency_t window_frequency,
            key_frequency_t window_frequency_square,
            distance_t window_min_dist
    ) {
        this->document_frequency = document_frequency;
        this->window_document_frequency = window_document_frequency;
        this->window_frequency = window_frequency;
        this->window_frequency_square = window_frequency_square;
        this->window_min_dist = window_min_dist;
    }

    inline void
    update(const StatsKeyTriple &other) {
        this->document_frequency += other.document_frequency;
        this->window_document_frequency += other.window_document_frequency;
        this->window_frequency += other.window_frequency;
        this->window_frequency_square += other.window_frequency_square;
        if (this->window_min_dist > other.window_min_dist) {
            this->window_min_dist = other.window_min_dist;
        }
    }
};


template<
        typename KeyType,
        bool B_DISABLE_UNWINDOWED = false,
        bool B_RESTRICTED = true,
        bool B_BUFFERED_WORKER = false,
        bool B_BUFFERED_COLLECTOR = false
>
class CollectionStatsFiller;


/**
 * Collection Stats class, used to collect statistic about collections and query them
 * @tparam KeyType The elements type
 * @tparam B_RESTRICTED A boolean saying if the statistics must be restricted to only certain keys
 */
template<
        typename KeyType,
        bool B_DISABLE_UNWINDOWED = false,
        bool B_RESTRICTED = true
>
class CollectionStats {
private:
    using _Key = KeyType;
    using _KeyPair = KeyPair<KeyType>;
    using _KeyTriple = KeyTriple<KeyType>;

public:
/**
 * Object fields
 */
public:
    const distance_t window_size_key_pairs_co_occ;
    const distance_t window_size_key_triples_co_occ;

private:
    // the following fields are used as zero elements
    const StatsKey zero_stats_key = StatsKey();
    const StatsKeyPair zero_stats_key_pair = StatsKeyPair();
    const StatsKeyTriple zero_stats_key_triple = StatsKeyTriple();

    document_frequency_t num_docs;  // number of documents
    key_frequency_t key_frequency_sum;  // sum of the key frequencies
    key_frequency_t key_pair_window_co_occ_sum;  // sum of the windowed pair co_occ
    key_frequency_t key_triple_window_co_occ_sum;  // sum of the windowed triple co_occ

    std::unordered_map<_Key, StatsKey> stats_key;  // key to stats_key
    std::unordered_map<_KeyPair, StatsKeyPair> stats_key_pair;  // key_pair to stats_key_pair
    std::unordered_map<_KeyTriple, StatsKeyTriple> stats_key_triple;  // key_triple to stats_key_triple

public:
    friend class CollectionStatsFiller<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED, false, false>;
    friend class CollectionStatsFiller<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED, false, true>;
    friend class CollectionStatsFiller<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED, true, false>;
    friend class CollectionStatsFiller<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED, true, true>;

    CollectionStats(
            distance_t window_size_key_pairs_co_occ = 12,
            distance_t window_size_key_triples_co_occ = 15
    ) :
            window_size_key_pairs_co_occ(window_size_key_pairs_co_occ),
            window_size_key_triples_co_occ(window_size_key_triples_co_occ),
            num_docs(0),
            key_frequency_sum(0),
            key_pair_window_co_occ_sum(0),
            key_triple_window_co_occ_sum(0) {};

    /**
     * Reset the collection stats and all the restrictions
     */
    void clear() {
        this->num_docs = 0;
        this->key_frequency_sum = 0;
        this->key_pair_window_co_occ_sum = 0;
        this->key_triple_window_co_occ_sum = 0;

        this->stats_key.clear();
        this->stats_key_pair.clear();
        this->stats_key_triple.clear();
    }

    void
    dump(
            const std::string &filename
    ) const {
        std::ofstream outfile(filename, std::fstream::trunc | std::fstream::binary);
        try {
            this->dumps(&outfile);
            outfile.close();
        } catch (...) {
            outfile.close();
            throw;
        }
    }

    void
    dumps(
            std::ostream *os
    ) const {
        BufferedWriter<false> writer(os, 8192);
        this->dumps(writer);
        writer.flush();
    }

    void
    dumps(
            BufferedWriter<false> &writer
    ) const {
        if (is_pointer<_Key>::value) {
            throw std::runtime_error("Unable to serialize this CollectionStats type");
        }

        // write the size of the type
        writer.put<size_t>(sizeof(_Key));
        // write if this collection is restricted
        writer.put<bool>(B_DISABLE_UNWINDOWED);
        // write if this collection is restricted
        writer.put<bool>(B_RESTRICTED);
        // write pair window size
        writer.put<distance_t>(this->window_size_key_pairs_co_occ);
        // write triple window size
        writer.put<distance_t>(this->window_size_key_triples_co_occ);
        // write num_docs
        writer.put<document_frequency_t>(this->num_docs);
        // write key_frequency_sum
        writer.put<key_frequency_t>(this->key_frequency_sum);
        // write key_pair_window_co_occ_sum
        writer.put<key_frequency_t>(this->key_pair_window_co_occ_sum);
        // write key_triple_window_co_occ_sum
        writer.put<key_frequency_t>(this->key_triple_window_co_occ_sum);

        // write stats_key, which type is unordered_map<_Key, StatsKey>
        writer.put<size_t>(this->stats_key.size());
        for (auto it: this->stats_key) {
            writer.put<_Key>(it.first);
            writer.put<StatsKey>(it.second);
        }
        // write stats_key_pair, which type is unordered_map<_KeyPair, StatsKeyPair>
        writer.put<size_t>(this->stats_key_pair.size());
        for (auto it: this->stats_key_pair) {
            writer.put<_KeyPair>(it.first);
            writer.put<StatsKeyPair>(it.second);
        }
        // write stats_key_triple, which type is unordered_map<_KeyTriple, StatsKeyTriple>
        writer.put<size_t>(this->stats_key_triple.size());
        for (auto it: this->stats_key_triple) {
            writer.put<_KeyTriple>(it.first);
            writer.put<StatsKeyTriple>(it.second);
        }
    }

    static CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> *
    load(
            const std::string &filename
    ) {
        std::ifstream infile(filename, std::ifstream::binary);
        if (infile.fail() or !infile.is_open()) {
            throw std::runtime_error("The file cannot be opened");
        }
        try {
            BufferedReader<false> reader(&infile, 8 * 1024 * 1024);
            CollectionStats<_Key, B_DISABLE_UNWINDOWED, B_RESTRICTED> *result = CollectionStats<_Key, B_DISABLE_UNWINDOWED, B_RESTRICTED>::loads(reader);
            infile.close();
            return result;
        } catch (...) {
            infile.close();
            throw;
        }
    }

    static CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> *
    loads(
            std::istream *is
    ) {
        BufferedReader<true> reader(is, 8 * 1024 * 1024, 0);
        return CollectionStats::loads(reader);
    };

    template<bool use_read_constraint = true>
    static CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> *
    loads(
            BufferedReader<use_read_constraint> &reader
    ) {
        if (is_pointer<_Key>::value) {
            throw std::runtime_error("Unable to deserialize this CollectionStats type");
        }

        // temporary size and read buffer used in the following code
        size_t tmp_size;
        if (use_read_constraint) {
            const size_t initial_buffer_limit =
                    sizeof(size_t)  // type size
                    + sizeof(bool)  // disable unwindowed type
                    + sizeof(bool)  // restriction type
                    + sizeof(distance_t)  // window_size_key_pairs_co_occ
                    + sizeof(distance_t)  // window_size_key_triples_co_occ
                    + sizeof(document_frequency_t)  // num_docs
                    + sizeof(key_frequency_t)  // key_frequency_sum
                    + sizeof(key_frequency_t)  // key_pair_window_co_occ_sum
                    + sizeof(key_frequency_t)  // key_triple_window_co_occ_sum
                    + sizeof(size_t)  // stats_key size
                    + sizeof(size_t)  // stats_key_pair size
                    + sizeof(size_t);  // stats_key_triple size
            reader.increase_num_bytes_constraint(initial_buffer_limit);
        }

        // read the size of the type
        tmp_size = reader.template get<size_t>();
        if (tmp_size != sizeof(_Key)) {
            throw std::runtime_error("The type of the collection to load is not compatible with the one given");
        }
        bool restricted = reader.template get<bool>();
        if (restricted != B_DISABLE_UNWINDOWED) {
            throw std::runtime_error("The collection to load is has not the same type B_DISABLE_UNWINDOWED of this one");
        }
        restricted = reader.template get<bool>();
        if (restricted != B_RESTRICTED) {
            throw std::runtime_error("The collection to load is has not the same type B_RESTRICTED of this one");
        }
        // create the collection stats to return
        CollectionStats<_Key, B_DISABLE_UNWINDOWED, B_RESTRICTED> *result = new CollectionStats<_Key, B_DISABLE_UNWINDOWED, B_RESTRICTED>(
                reader.template get<distance_t>(),
                reader.template get<distance_t>()
        );

        // read num_docs
        result->num_docs = reader.template get<document_frequency_t>();
        // read key_frequency_sum
        result->key_frequency_sum = reader.template get<key_frequency_t>();
        // read key_pair_window_co_occ_sum
        result->key_pair_window_co_occ_sum = reader.template get<key_frequency_t>();
        // read key_triple_window_co_occ_sum
        result->key_triple_window_co_occ_sum = reader.template get<key_frequency_t>();

        // read stats_key, which type is unordered_map<_Key, StatsKey>
        tmp_size = reader.template get<size_t>();
        if (use_read_constraint) {
            reader.increase_num_bytes_constraint(tmp_size * (sizeof(_Key) + sizeof(StatsKey)));
        }
        result->stats_key.reserve(tmp_size);

        for (size_t i = 0; i < tmp_size; ++i) {
            _Key key(reader.template get<_Key>());
            StatsKey value(reader.template get<StatsKey>());

            result->stats_key.insert({key, value});
        }

        // read stats_key_pair, which type is unordered_map<_KeyPair, StatsKeyPair>
        tmp_size = reader.template get<size_t>();
        if (use_read_constraint) {
            reader.increase_num_bytes_constraint(tmp_size * (sizeof(_KeyPair) + sizeof(StatsKeyPair)));
        }
        result->stats_key_pair.reserve(tmp_size);

        for (size_t i = 0; i < tmp_size; ++i) {
            _KeyPair key(reader.template get<_KeyPair>());
            StatsKeyPair value(reader.template get<StatsKeyPair>());

            result->stats_key_pair.insert({key, value});
        }

        // read stats_key_triple, which type is unordered_map<_KeyTriple, StatsKeyTriple>
        tmp_size = reader.template get<size_t>();
        if (use_read_constraint) {
            reader.increase_num_bytes_constraint(tmp_size * (sizeof(_KeyTriple) + sizeof(StatsKeyTriple)));
        }
        result->stats_key_triple.reserve(tmp_size);

        for (size_t i = 0; i < tmp_size; ++i) {
            _KeyTriple key(reader.template get<_KeyTriple>());
            StatsKeyTriple value(reader.template get<StatsKeyTriple>());

            result->stats_key_triple.insert({key, value});
        }

        return result;
    }

    document_frequency_t
    get_num_docs() const noexcept {
        return this->num_docs;
    }

    size_t
    get_num_keys() const noexcept {
        return this->stats_key.size();
    }

    size_t
    get_num_key_pairs() const noexcept {
        return this->stats_key_pair.size();
    }

    size_t
    get_num_key_triples() const noexcept {
        return this->stats_key_triple.size();
    }

    key_frequency_t
    get_key_frequency_sum() const noexcept {
        return this->key_frequency_sum;
    }

    key_frequency_t
    get_key_pair_window_co_occ_sum() const noexcept {
        return this->key_pair_window_co_occ_sum;
    }

    key_frequency_t
    get_key_triple_window_co_occ_sum() const noexcept {
        return this->key_triple_window_co_occ_sum;
    }

    StatsKey
    get_stats_key(
            const KeyType &key
    ) const {
        auto stats_1_key_it = this->stats_key.find(key);
        if (stats_1_key_it == this->stats_key.end()) {
            if (B_RESTRICTED) {
//                throw std::runtime_error("The given key is not among the restrictions");
            }
            return this->zero_stats_key;
        } else {
            return stats_1_key_it->second;
        }
    }

    StatsKeyPair
    get_stats_key_pair(
            const KeyPair<KeyType> &keyPair
    ) const {
        auto stats_2_key_it = this->stats_key_pair.find(keyPair);
        if (B_DISABLE_UNWINDOWED) {
            if (stats_2_key_it == this->stats_key_pair.end()) {
                if (B_RESTRICTED) {
//                    throw std::runtime_error("The given pair of keys is not among the restrictions");
                }
                return this->zero_stats_key_pair;
            } else {
                return stats_2_key_it->second;
            }
        } else {
            if (stats_2_key_it == this->stats_key_pair.end()) {
                if (B_RESTRICTED) {
//                    throw std::runtime_error("The given pair of keys is not among the restrictions");
                }
                document_frequency_t document_co_occ = 0;
                if (std::equal_to<_Key>()(keyPair.first(), keyPair.second())) {
                    document_co_occ = this->get_stats_key(keyPair.first()).document_frequency;
                }

                if (document_co_occ == 0) {
                    return this->zero_stats_key_pair;
                } else {
                    StatsKeyPair statsKeyPair = this->zero_stats_key_pair;
                    statsKeyPair.document_frequency = document_co_occ;
                    return statsKeyPair;
                }
            } else {
                StatsKeyPair statsKeyPair = stats_2_key_it->second;
                if (std::equal_to<_Key>()(keyPair.first(), keyPair.second())) {
                    statsKeyPair.document_frequency = this->get_stats_key(keyPair.first()).document_frequency;
                }
                return statsKeyPair;
            }
        }
    }

    StatsKeyPair
    get_stats_key_pair(
            const KeyType &first,
            const KeyType &second
    ) const {
        return this->get_stats_key_pair(_KeyPair(first, second));
    }

    StatsKeyTriple
    get_stats_key_triple(
            const KeyTriple<KeyType> &keyTriple
    ) const {
        auto stats_3_key_it = this->stats_key_triple.find(keyTriple);
        if (B_DISABLE_UNWINDOWED) {
            if (stats_3_key_it == this->stats_key_triple.end()) {
                if (B_RESTRICTED) {
//                    throw std::runtime_error("The given triple of keys is not among the restrictions");
                }
                return this->zero_stats_key_triple;
            } else {
                return stats_3_key_it->second;
            }
        } else {
            if (stats_3_key_it == this->stats_key_triple.end()) {
                if (B_RESTRICTED) {
//                    throw std::runtime_error("The given triple of keys is not among the restrictions");
                }
                document_frequency_t document_co_occ = 0;
                if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.third())) {
                    // all the keys are the same, because they are ordered
                    document_co_occ = this->get_stats_key(keyTriple.first()).document_frequency;
                } else if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.second())) {
                    // the first key is equal to the second one, but it is different from the third one
                    document_co_occ = this->get_stats_key_pair(
                            _KeyPair(keyTriple.first(), keyTriple.third())).document_frequency;
                } else if (std::equal_to<_Key>()(keyTriple.second(), keyTriple.third())) {
                    // the second key is equal to the third one, but it is different from the first one
                    document_co_occ = this->get_stats_key_pair(
                            _KeyPair(keyTriple.second(), keyTriple.first())).document_frequency;
                }

                if (document_co_occ == 0) {
                    return this->zero_stats_key_triple;
                } else {
                    StatsKeyTriple statsKeyTriple = this->zero_stats_key_triple;
                    statsKeyTriple.document_frequency = document_co_occ;
                    return statsKeyTriple;
                }
            } else {
                StatsKeyTriple statsKeyTriple = stats_3_key_it->second;

                if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.third())) {
                    // all the keys are the same, because they are ordered
                    statsKeyTriple.document_frequency = this->get_stats_key(keyTriple.first()).document_frequency;
                } else if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.second())) {
                    // the first key is equal to the second one, but it is different from the third one
                    statsKeyTriple.document_frequency = this->get_stats_key_pair(
                            _KeyPair(keyTriple.first(), keyTriple.third())).document_frequency;
                } else if (std::equal_to<_Key>()(keyTriple.second(), keyTriple.third())) {
                    // the second key is equal to the third one, but it is different from the first one
                    statsKeyTriple.document_frequency = this->get_stats_key_pair(
                            _KeyPair(keyTriple.second(), keyTriple.first())).document_frequency;
                }
                return statsKeyTriple;
            }
        }
    }

    StatsKeyTriple
    get_stats_key_triple(
            const KeyType &first,
            const KeyType &second,
            const KeyType &third
    ) const {
        return this->get_stats_key_triple(_KeyTriple(first, second, third));
    }

    void
    update(
            const CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> &other
    ) {
        if (this->window_size_key_pairs_co_occ != other.window_size_key_pairs_co_occ ||
            this->window_size_key_triples_co_occ != other.window_size_key_triples_co_occ) {
            throw std::runtime_error("The two collection stats must be based on the same windows");
        }

        // update num_docs, key_frequency_sum, key_pair_window_co_occ_sum, key_triple_window_co_occ_sum
        this->num_docs += other.num_docs;
        this->key_frequency_sum += other.key_frequency_sum;
        this->key_pair_window_co_occ_sum += other.key_pair_window_co_occ_sum;
        this->key_triple_window_co_occ_sum += other.key_triple_window_co_occ_sum;

        // update stats_key
        for (auto other_stats_key_it: other.stats_key) {
            auto stats_key_it = this->stats_key.find(other_stats_key_it.first);
            if (stats_key_it != this->stats_key.end()) {
                stats_key_it->second.update(other_stats_key_it.second);
            } else if (!B_RESTRICTED) {
                this->stats_key.insert(other_stats_key_it);
            }
        }
        // update stats_key_pair
        for (auto other_stats_key_pair_it: other.stats_key_pair) {
            auto stats_key_pair_it = this->stats_key_pair.find(other_stats_key_pair_it.first);
            if (stats_key_pair_it != this->stats_key_pair.end()) {
                stats_key_pair_it->second.update(other_stats_key_pair_it.second);
            } else if (!B_RESTRICTED) {
                this->stats_key_pair.insert(other_stats_key_pair_it);
            }
        }
        // update stats_key_triple
        for (auto other_stats_key_triple_it: other.stats_key_triple) {
            auto stats_key_triple_it = this->stats_key_triple.find(other_stats_key_triple_it.first);
            if (stats_key_triple_it != this->stats_key_triple.end()) {
                stats_key_triple_it->second.update(other_stats_key_triple_it.second);
            } else if (!B_RESTRICTED) {
                this->stats_key_triple.insert(other_stats_key_triple_it);
            }
        }
    }

private:
    template<typename _T>
    struct is_pointer {
        static const bool value = false;
    };

    template<typename _T>
    struct is_pointer<_T *> {
        static const bool value = true;
    };
};


/**
 * Collection Stats Filler class, used to fill a CollectionStats object from texts and maches
 * @tparam KeyType The elements type
 * @tparam B_RESTRICTED A boolean saying if the statistics must be restricted to only certain keys
 */
template<
        typename KeyType,
        bool B_DISABLE_UNWINDOWED,
        bool B_RESTRICTED,
        bool B_BUFFERED_WORKER,
        bool B_BUFFERED_COLLECTOR
>
class CollectionStatsFiller {
private:
    using _Key = KeyType;
    using _KeyPair = KeyPair<KeyType>;
    using _KeyTriple = KeyTriple<KeyType>;

    using KeyEntry = std::pair<_Key, StatsKey>;
    using KeyPairEntry = std::pair<_KeyPair, StatsKeyPair>;
    using KeyTripleEntry = std::pair<_KeyTriple, StatsKeyTriple>;

    /**
     * Struct used by sort algorithm to internally sort a buffer of pairs
     */
    template<typename _Key>
    struct PositionsLessThanPred : public std::binary_function<size_t, size_t, bool> {
    private:
        const char *buffer;
    public:
        PositionsLessThanPred(const char *buffer) : buffer(buffer) {}

        bool
        operator()(const size_t l, const size_t r) const {
            return std::less<_Key>()(*(_Key *) (buffer + l), *(_Key *) (buffer + r));
        }
    };

    /**
     * Struct used by sort algorithm to internally sort a buffer of pairs or triple
     */
    template<typename _Key, typename _Value>
    struct PositionsKeyValueLessThanPred : public std::binary_function<size_t, size_t, bool> {
    private:
        const char *buffer;
    public:
        PositionsKeyValueLessThanPred(const char *buffer) : buffer(buffer) {}

        bool
        operator()(const size_t l, const size_t r) const {
            return std::less<_Key>()(
                    ((std::pair<_Key, _Value> *) (buffer + l))->first,
                    ((std::pair<_Key, _Value> *) (buffer + r))->first
            );
        }
    };

/**
* Object fields
*/
private:
    CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> *collection_stats;
    const PatternMatcher<KeyType> *pattern_matcher;
    const distance_t max_window_size_co_occ;
    bool add_restrictions_enabled;

    std::vector<std::thread> threads;
    std::queue<std::vector<std::string>> job_queue;
    size_t job_queue_limit;
    std::mutex job_queue_mutex;
    std::condition_variable job_queue_condition_variable;
    std::condition_variable job_queue_wait_condition_variable;

    uint32_t job_queue_num_working_threads;
    std::condition_variable job_queue_num_working_threads_condition_variable;

    std::vector<char> buffer_stats;
    std::size_t buffer_stats_end;
    std::size_t buffer_stats_remaining;

    std::vector<std::size_t> buffer_stats_keys_positions;
    std::vector<std::size_t> buffer_stats_key_pairs_positions;
    std::vector<std::size_t> buffer_stats_key_triples_positions;

    volatile bool buffer_stats_busy = false;
    std::mutex buffer_stats_mutex;
    std::condition_variable buffer_stats_condition_variable;

    // suitable keys/pairs for the restricted version of this class
    std::unordered_map<_Key, char> suitable_keys;  // key to bit mask.
    std::unordered_map<_KeyPair, char> suitable_key_pairs;  // key_pair to bit mask.
    // mask used by the mappings above
    static const char SUITABLE_FOR_TERM_MASK = (1 << 0);
    static const char SUITABLE_FOR_TERM_PAIR_MASK = (1 << 1);
    static const char SUITABLE_FOR_TERM_TRIPLE_MASK = (1 << 2);

public:
    CollectionStatsFiller(
            CollectionStats<KeyType, B_DISABLE_UNWINDOWED, B_RESTRICTED> *collection_stats,
            const PatternMatcher<KeyType> *pattern_matcher,
            std::size_t buffer_size_in_bytes,
            uint32_t num_threads = 1,
            uint32_t queue_max_size = 1
    ) :
            collection_stats(collection_stats),
            pattern_matcher(pattern_matcher),
            max_window_size_co_occ(std::max(collection_stats->window_size_key_pairs_co_occ,
                                            collection_stats->window_size_key_triples_co_occ)),
            add_restrictions_enabled(collection_stats->num_docs == 0),
            job_queue_limit(queue_max_size),
            job_queue_num_working_threads(num_threads) {
        if (num_threads <= 0) {
            throw std::runtime_error("num_threads must be greater than 0");
        }
        if (queue_max_size <= 0) {
            throw std::runtime_error("queue_max_size must be greater than 0");
        }

        if (B_BUFFERED_COLLECTOR) {
            std::size_t max_entry_size = std::max(sizeof(KeyEntry),
                                                  std::max(sizeof(KeyPairEntry), sizeof(KeyTripleEntry)));
            if (buffer_size_in_bytes < max_entry_size) {
                throw std::runtime_error("buffer_size_in_bytes is too small");
            }
        } else {
            if (buffer_size_in_bytes > 0) {
                throw std::runtime_error(
                        "buffer_size_in_bytes must be 0 when used with the UNbuffered version of the filler");
            }
        }

        // update the suitable hash maps according to the content of the collection stats
        if (B_RESTRICTED) {
            for (auto stats_key_it: collection_stats->stats_key) {
                this->update_suitable<false>(stats_key_it.first);
            }

            for (auto stats_key_pair_it: collection_stats->stats_key_pair) {
                this->update_suitable<false>(stats_key_pair_it.first);
            }

            for (auto stats_key_triple_it: collection_stats->stats_key_triple) {
                this->update_suitable<false>(stats_key_triple_it.first);
            }
        }

        // compute the buffers dimension
        if (B_BUFFERED_COLLECTOR) {
            std::size_t min_entry_size = std::min(sizeof(KeyEntry),
                                                  std::min(sizeof(KeyPairEntry), sizeof(KeyTripleEntry)));

            // the constraints to compute the buffer size (bs) and buffer_stats_positions size (bps) are the following:
            // bs + bps = buffer_size_in_bytes && bs / min_entry_size <= bps / sizeof(size_t)
            // (implies) bps * min_entry_size / sizeof(size_t) + bps = buffer_size_in_bytes
            // (implies) bps * min_entry_size + bps * sizeof(size_t) = buffer_size_in_bytes * sizeof(size_t)
            // (implies) bps * (min_entry_size + sizeof(size_t)) = buffer_size_in_bytes * sizeof(size_t)
            // (implies) bps = buffer_size_in_bytes * sizeof(size_t) / (min_entry_size + sizeof(size_t))
            std::size_t bps = sizeof(std::size_t) + buffer_size_in_bytes * sizeof(std::size_t) / (min_entry_size +
                                                                                                  sizeof(std::size_t)); // the plus sizeof(std::size_t) is to round up bps instead of bs
            std::size_t bs = buffer_size_in_bytes - bps;

            this->buffer_stats.resize(bs);
            this->buffer_stats_end = 0;
            this->buffer_stats_remaining = bs;
        } else {
            this->buffer_stats_end = 0;
            this->buffer_stats_remaining = 0;
        }

        for (uint32_t i = 0; i < num_threads; ++i) {
            threads.push_back(std::thread(&CollectionStatsFiller::update_worker_loop, this));
        }
    }

    ~CollectionStatsFiller() {
        // send an exit message to all threads
        {
            std::unique_lock<std::mutex> lock(this->job_queue_mutex);
            for (uint32_t i = 0; i < this->threads.size(); ++i) {
                this->job_queue.push({});
            }
            this->job_queue_condition_variable.notify_all();
        }

        this->flush();

        // join all threads
        for (uint32_t i = 0; i < this->threads.size(); ++i) {
            this->threads[i].join();
        }
    }

    void
    add_restriction(
            const KeyType &key
    ) {
        check_add_restriction();

        this->update_suitable<true>(key);
    }

    void
    add_restriction(
            const KeyPair<KeyType> &keyPair
    ) {
        check_add_restriction();

        this->update_suitable<true>(keyPair);
    }

    void
    add_restriction(
            const KeyType &first,
            const KeyType &second
    ) {
        this->add_restriction(_KeyPair(first, second));
    }

    void
    add_restriction(
            const KeyTriple<KeyType> &keyTriple
    ) {
        check_add_restriction();

        this->update_suitable<true>(keyTriple);
    }

    void
    add_restriction(
            const KeyType &first,
            const KeyType &second,
            const KeyType &third
    ) {
        this->add_restriction(_KeyTriple(first, second, third));
    }

    void
    update(
            const std::vector<std::string> &doc_fields
    ) {
        this->add_restrictions_enabled = false;
        if (doc_fields.size() == 0)
            return;

        {
            std::unique_lock<std::mutex> lock(this->job_queue_mutex);
            this->job_queue.push(doc_fields);
            this->job_queue_condition_variable.notify_one();
        }
    }

    void
    update(
            std::vector<std::string> &doc_fields
    ) {
        this->add_restrictions_enabled = false;
        if (doc_fields.size() == 0)
            return;

        {
            std::unique_lock<std::mutex> lock(this->job_queue_mutex);
            while (this->job_queue.size() > this->job_queue_limit) {
                this->job_queue_wait_condition_variable.wait(lock);
            }
            this->job_queue.push({});
            std::swap(this->job_queue.back(), doc_fields);
            this->job_queue_condition_variable.notify_one();
        }
    }

    void
    flush() {
        {
            // wait until no jobs are available
            std::unique_lock<std::mutex> lock(this->job_queue_mutex);
            while (this->job_queue.size() > 0 || this->job_queue_num_working_threads > 0) {
                this->job_queue_num_working_threads_condition_variable.wait(lock);
            }

            // lock before ending
            if (B_BUFFERED_COLLECTOR) {
                this->update_lock();
            }
        }

        if (B_BUFFERED_COLLECTOR) {
            // flush the internal buffer
            this->flush_impl();
            this->update_unlock();
        }
    }

private:
    inline void
    add_key_into_buffer(
            const _Key &key,
            const StatsKey &statsKey
    ) {
        // THIS FUNCTION MUST BE CALLED INSIDE A THREAD SAFE CODE
        this->add_into_buffer<_Key, StatsKey>(key, statsKey, this->buffer_stats_keys_positions);
    }

    inline void
    add_key_pair_into_buffer(
            const _KeyPair &keyPair,
            const StatsKeyPair &statsKeyPair
    ) {
        // THIS FUNCTION MUST BE CALLED INSIDE A THREAD SAFE CODE
        this->add_into_buffer<_KeyPair, StatsKeyPair>(keyPair, statsKeyPair, this->buffer_stats_key_pairs_positions);
    }

    inline void
    add_key_triple_into_buffer(
            const _KeyTriple &keyTriple,
            const StatsKeyTriple &statsKeyTriple
    ) {
        // THIS FUNCTION MUST BE CALLED INSIDE A THREAD SAFE CODE
        this->add_into_buffer<_KeyTriple, StatsKeyTriple>(keyTriple, statsKeyTriple,
                                                          this->buffer_stats_key_triples_positions);
    }

    template<typename Key, typename Value>
    inline void
    add_into_buffer(
            const Key &key,
            const Value &stats,
            std::vector<size_t> &positions
    ) {
        if (this->buffer_stats_remaining < sizeof(Key) + sizeof(Value)) {
            this->flush_impl();
        }

        // update this key - value inside the buffer
        std::pair<Key, Value> *dest = (std::pair<Key, Value> *) (this->buffer_stats.data() + this->buffer_stats_end);
        dest->first = key;
        dest->second = stats;
        // update the positions vector
        positions.push_back(this->buffer_stats_end);
        // update buffer_stats properties
        this->buffer_stats_end += sizeof(Key) + sizeof(Value);
        this->buffer_stats_remaining -= sizeof(Key) + sizeof(Value);
    }

    inline bool
    add_key(
            const _Key &key,
            const StatsKey &statsKey
    ) {
        if (this->add(key, statsKey, this->collection_stats->stats_key)) {
            this->collection_stats->key_frequency_sum += statsKey.frequency;
            return true;
        } else {
            return false;
        }
    }

    inline bool
    add_key_pair(
            const _KeyPair &keyPair,
            const StatsKeyPair &statsKeyPair
    ) {
        if (this->add(keyPair, statsKeyPair, this->collection_stats->stats_key_pair)) {
            this->collection_stats->key_pair_window_co_occ_sum += statsKeyPair.window_frequency;
            return true;
        } else {
            return false;
        }
    }

    inline bool
    add_key_triple(
            const _KeyTriple &keyTriple,
            const StatsKeyTriple &statsKeyTriple
    ) {
        if (this->add(keyTriple, statsKeyTriple, this->collection_stats->stats_key_triple)) {
            this->collection_stats->key_triple_window_co_occ_sum += statsKeyTriple.window_frequency;
            return true;
        } else {
            return false;
        }
    }

    template<typename Key, typename Value>
    inline bool
    add(
            const Key &key,
            const Value &value,
            std::unordered_map<Key, Value> &stats
    ) {
        // update this key inside the stats
        typename std::unordered_map<Key, Value>::iterator stats_it = stats.find(key);
        if (stats_it != stats.end()) {
            stats_it->second.update(value);
            return true;
        } else if (!B_RESTRICTED) {
            stats.insert({key, value});
            return true;
        }
        return false;
    }

    inline void
    check_add_restriction() const {
        if (!B_RESTRICTED) {
            throw std::runtime_error("Operation not permitted when the CollectionStats is not restricted");
        }
        if (this->collection_stats->num_docs > 0 or not this->add_restrictions_enabled) {
            throw std::runtime_error("Operation not permitted when the CollectionStats has been already updated");
        }
    }

    void
    flush_impl() {
        // THIS CODE MUST BE CALLED INSIDE A THREAD SAFE AREA

        // map/reduce style
        this->collection_stats->key_frequency_sum += this->flush_impl_reduce(
                this->buffer_stats.data(),
                this->buffer_stats_keys_positions,
                this->collection_stats->stats_key
        );

        this->collection_stats->key_pair_window_co_occ_sum += this->flush_impl_reduce(
                this->buffer_stats.data(),
                this->buffer_stats_key_pairs_positions,
                this->collection_stats->stats_key_pair
        );

        this->collection_stats->key_triple_window_co_occ_sum += this->flush_impl_reduce(
                this->buffer_stats.data(),
                this->buffer_stats_key_triples_positions,
                this->collection_stats->stats_key_triple
        );

        this->buffer_stats_keys_positions.clear();
        this->buffer_stats_key_pairs_positions.clear();
        this->buffer_stats_key_triples_positions.clear();
        this->buffer_stats_end = 0;
        this->buffer_stats_remaining = this->buffer_stats.size();
    }

    template<typename Key, typename Value>
    size_t
    flush_impl_reduce(
            const char *buffer,
            std::vector<size_t> &buffer_positions,
            std::unordered_map<Key, Value> &stats
    ) {
        __gnu_parallel::sort(
                buffer_positions.begin(),
                buffer_positions.end(),
                PositionsKeyValueLessThanPred<Key, StatsKey>(buffer),
                __gnu_parallel::multiway_mergesort_exact_tag()
        );

        size_t acc = 0;
        for (std::size_t l = 0, r = 0, end = buffer_positions.size(); l < end; l = r) {
            const std::pair<Key, Value> *l_pair = buffer_get<std::pair<Key, Value>>(buffer_positions[l], buffer);

            r = l + 1;
            while (r < end) {
                const std::pair<Key, Value> *r_pair = buffer_get<std::pair<Key, Value>>(buffer_positions[r], buffer);
                if (!std::equal_to<Key>()(l_pair->first, r_pair->first)) {
                    break;
                }
                ++r;
            }

            // update this key inside the stats
            typename std::unordered_map<Key, Value>::iterator stats_it = stats.find(l_pair->first);
            if (stats_it != stats.end() || !B_RESTRICTED) {
                Value value = l_pair->second;
                ++l;
                while (l < r) {
                    const std::pair<Key, Value> *r_pair = buffer_get<std::pair<Key, Value>>(buffer_positions[l++],
                                                                                            buffer);
                    value.update(r_pair->second);
                }

                acc += get_frequency(value);

                // insert or update the key-value
                if (stats_it != stats.end()) {
                    stats_it->second.update(value);
                } else {
                    stats.insert({l_pair->first, value});
                }
            }
            // end key update
        }
        return acc;
    };

    template<bool _INSERT = true>
    inline void
    update_suitable(
            const _Key &key
    ) {
        if (_INSERT) {
            // the insert doesn't modify the map if the key is already in
            this->collection_stats->stats_key.insert({key, this->collection_stats->zero_stats_key});
        }

        this->update_suitable_key(key, SUITABLE_FOR_TERM_MASK);
    }

    template<bool _INSERT = true>
    inline void
    update_suitable(
            const _KeyPair &keyPair
    ) {
        if (_INSERT) {
            // the insert doesn't modify the map if the key is already in
            this->collection_stats->stats_key_pair.insert({keyPair, this->collection_stats->zero_stats_key_pair});
        }

        this->update_suitable_key(keyPair.first(), SUITABLE_FOR_TERM_PAIR_MASK);
        this->update_suitable_key(keyPair.second(), SUITABLE_FOR_TERM_PAIR_MASK);

        this->update_suitable_key_pair(keyPair, SUITABLE_FOR_TERM_PAIR_MASK);

        if (std::equal_to<_Key>()(keyPair.first(), keyPair.second())) {
            this->update_suitable(keyPair.first());
        }
    }

    template<bool _INSERT = true>
    inline void
    update_suitable(
            const _KeyTriple &keyTriple
    ) {
        if (_INSERT) {
            // the insert doesn't modify the map if the key is already in
            this->collection_stats->stats_key_triple.insert({keyTriple, this->collection_stats->zero_stats_key_triple});
        }

        this->update_suitable_key(keyTriple.first(), SUITABLE_FOR_TERM_TRIPLE_MASK);
        this->update_suitable_key(keyTriple.second(), SUITABLE_FOR_TERM_TRIPLE_MASK);
        this->update_suitable_key(keyTriple.third(), SUITABLE_FOR_TERM_TRIPLE_MASK);

        this->update_suitable_key_pair(_KeyPair(keyTriple.first(), keyTriple.second()),
                                       SUITABLE_FOR_TERM_TRIPLE_MASK);
        this->update_suitable_key_pair(_KeyPair(keyTriple.first(), keyTriple.third()),
                                       SUITABLE_FOR_TERM_TRIPLE_MASK);
        this->update_suitable_key_pair(_KeyPair(keyTriple.second(), keyTriple.third()),
                                       SUITABLE_FOR_TERM_TRIPLE_MASK);

        if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.third())) {
            // all the keys are the same, because they are ordered
            this->update_suitable(keyTriple.first());
        } else if (std::equal_to<_Key>()(keyTriple.first(), keyTriple.second())) {
            // the first key is equal to the second one, but it is different from the first one
            this->update_suitable(_KeyPair(keyTriple.first(), keyTriple.third()));
        } else if (std::equal_to<_Key>()(keyTriple.second(), keyTriple.third())) {
            // the second key is equal to the third one, but it is different from the first one
            this->update_suitable(_KeyPair(keyTriple.second(), keyTriple.first()));
        }
    }

    inline void
    update_suitable_key(
            const _Key &key,
            char mask
    ) {
        auto found = this->suitable_keys.find(key);
        if (found != this->suitable_keys.end()) {
            found->second |= mask;
        } else {
            this->suitable_keys[key] = mask;
        }
    }

    inline void
    update_suitable_key_pair(
            const _KeyPair &key,
            char mask
    ) {
        auto found = this->suitable_key_pairs.find(key);
        if (found != this->suitable_key_pairs.end()) {
            found->second |= mask;
        } else {
            this->suitable_key_pairs[key] = mask;
        }
    }

    void
    update_worker_loop() {
        // element of the job_queue
        std::vector<std::string> doc_fields;

        // map with the patterns length
        const std::unordered_map<KeyType, uint16_t> &pattern_to_length = this->pattern_matcher->get_pattern_length_map();

        // local buffers and data structures
        PatternMatches<KeyType> matches(true);
        std::vector<size_t> matches_start_pos;

        std::unordered_map<_Key, size_t> local_stats_key;
        std::unordered_map<_KeyPair, std::pair<size_t, distance_t>> local_stats_key_pair;
        std::unordered_map<_KeyTriple, std::pair<size_t, distance_t>> local_stats_key_triple;
        if (!B_BUFFERED_WORKER) {
            local_stats_key.reserve(1024);
            local_stats_key_pair.reserve(2048);
            local_stats_key_triple.reserve(4096);
        }

        // buffered version
        size_t local_buffer_end = 0;
        size_t local_buffer_size = 0;
        std::vector<char> local_buffer;

        std::vector<size_t> local_keys_positions;
        std::vector<size_t> local_key_pairs_positions;
        std::vector<size_t> local_key_triples_positions;
        if (B_BUFFERED_WORKER) {
            local_buffer_size = 8192;
            local_buffer.resize(local_buffer_size);
            local_keys_positions.reserve(1024);
            local_key_pairs_positions.reserve(2048);
            local_key_triples_positions.reserve(4096);
        }

        // MAIN LOOP
        while (true) {
            {
                std::unique_lock<std::mutex> lock(this->job_queue_mutex);

                // wait untill a job is available
                while (this->job_queue.empty()) {
                    if (--this->job_queue_num_working_threads == 0) {
                        this->job_queue_num_working_threads_condition_variable.notify_all();
                    }
                    this->job_queue_condition_variable.wait(lock);
                    ++this->job_queue_num_working_threads;
                }

                // swap the two vectors to avoid the copy
                std::swap(this->job_queue.front(), doc_fields);
                // remove the element from the job_queue
                this->job_queue.pop();

                // END CONDITION
                if (doc_fields.size() == 0) {
                    if (--this->job_queue_num_working_threads == 0) {
                        this->job_queue_num_working_threads_condition_variable.notify_all();
                    }
                    break;
                }

                // notify who is waiting the queue
                this->job_queue_wait_condition_variable.notify_one();
            }

            // iterate over the matches and aggregate the matchings into the local buffers
            for (size_t i = 0, end = doc_fields.size(); i < end; ++i) {
                // find the patterns
                this->pattern_matcher->find_patterns(doc_fields[i], matches);

                // initialize starting positions
                matches_start_pos.resize(matches.size());
                for (size_t i = 0, i_end = matches.size(); i < i_end; ++i) {
                    const PatternMatch<_Key> match = matches.at(i);
                    matches_start_pos[i] = match.end_pos + 1 - pattern_to_length.at(match.pattern);
                }

                // update the buffer
                this->update_fill_local_structures(
                        &matches, pattern_to_length, matches_start_pos,
                        local_buffer, local_buffer_end, local_buffer_size,
                        local_keys_positions, local_key_pairs_positions, local_key_triples_positions,
                        local_stats_key, local_stats_key_pair, local_stats_key_triple
                );

                // clear the matches buffer
                matches.clear();
            }

            // update from the local buffers
            if (B_BUFFERED_WORKER) {
                this->update_from_local_buffer(
                        local_buffer, local_buffer_end, local_buffer_size,
                        local_keys_positions, local_key_pairs_positions, local_key_triples_positions
                );
            } else {
                this->update_from_local_maps(
                        local_stats_key, local_stats_key_pair, local_stats_key_triple
                );
            }

            // clear all the local buffers
            doc_fields.clear();
            if (B_BUFFERED_WORKER) {
                local_buffer_end = 0;
                local_keys_positions.clear();
                local_key_pairs_positions.clear();
                local_key_triples_positions.clear();
            } else {
                local_stats_key.clear();
                local_stats_key_pair.clear();
                local_stats_key_triple.clear();
            }
        }
    }

    inline void
    update_fill_local_structures(
            const PatternMatches<_Key> *matches_it,
            const std::unordered_map<_Key, uint16_t> &pattern_to_length,
            const std::vector<size_t> &matches_start_pos,

            std::vector<char> &local_buffer,
            size_t &local_buffer_end,
            size_t &local_buffer_size,
            std::vector<size_t> &local_keys_positions,
            std::vector<size_t> &local_key_pairs_positions,
            std::vector<size_t> &local_key_triples_positions,

            std::unordered_map<_Key, size_t> &local_stats_key,
            std::unordered_map<_KeyPair, std::pair<size_t, distance_t>> &local_stats_key_pair,
            std::unordered_map<_KeyTriple, std::pair<size_t, distance_t>> &local_stats_key_triple
    ) const {
        const size_t match_size = matches_it->size();

        // left delimiter loop
        for (size_t l = 0, min_m = 1; l < match_size; ++l) {
            const PatternMatch<_Key> l_match = matches_it->at(l);

            // compute the mask related to the key l
            char l_mask = ~0;
            if (B_RESTRICTED) {
                auto st_it = this->suitable_keys.find(l_match.pattern);
                if (st_it != this->suitable_keys.end()) {
                    l_mask = st_it->second;
                } else {
                    l_mask = 0;
                }
            }

            // put the key into a buffer if it can be used as singleton
            //if (!B_RESTRICTED || l_mask & SUITABLE_FOR_TERM_MASK) {
            // UPDATE: put the key inside the buffer if it can partecipate to some count
            // then it will be ignored if it isn't helpful to any key, pair or triple
            if (!B_RESTRICTED || l_mask) {
                if (B_BUFFERED_WORKER) {
                    this->update_fill_local_buffer_push(
                            l_match.pattern,
                            local_keys_positions,
                            local_buffer, &local_buffer_end, &local_buffer_size
                    );
                } else {
                    auto stats_entry_it = local_stats_key.find(l_match.pattern);
                    if (stats_entry_it == local_stats_key.end()) {
                        local_stats_key.insert({l_match.pattern, 1});
                    } else {
                        stats_entry_it->second += 1;
                    }
                }
            }

            // check if there is at least one pair or triple that can be updated
            if (B_RESTRICTED && !(l_mask & (SUITABLE_FOR_TERM_PAIR_MASK | SUITABLE_FOR_TERM_TRIPLE_MASK))) {
                continue;
            }

            while (min_m < match_size && l_match.end_pos >= matches_start_pos[min_m]) {
                ++min_m;
            }
            // right delimiter loop
            // NOTE: the matches are ordered by increasing end_position, and when a tie occurs by decreasing pattern length
            for (size_t r = min_m; r < match_size; ++r) {
                // check if the two patterns overlap
                if (l_match.end_pos >= matches_start_pos[r]) {
                    continue;
                }
                const PatternMatch<_Key> r_match = matches_it->at(r);

                // compute the window size
                const size_t window_size =
                        r_match.end_pos - matches_start_pos[l] + 1;  // the + 1 is because the end_pos is included

                // nothing to windowize
                if (window_size > this->max_window_size_co_occ) {
                    break;
                }

                // compute the mask related to the pair of keys l, r
                char r_mask = ~0;
                if (B_RESTRICTED) {
                    _KeyPair keyPair(l_match.pattern, r_match.pattern);
                    auto st_it = this->suitable_key_pairs.find(keyPair);
                    if (st_it != this->suitable_key_pairs.end()) {
                        r_mask = st_it->second;
                    } else {
                        r_mask = 0;
                    }
                }

                // update doc_key_pairs
                if (window_size <= this->collection_stats->window_size_key_pairs_co_occ &&
                    (!B_RESTRICTED || (r_mask & SUITABLE_FOR_TERM_PAIR_MASK))) {
                    const distance_t pair_gap = (distance_t) (matches_start_pos[r] - l_match.end_pos - 1);
                    // the - 1 is because the end_pos is included and we want to know the number of words in the middle
                    _KeyPair keyPair(l_match.pattern, r_match.pattern);

                    if (B_BUFFERED_WORKER) {
                        this->update_fill_local_buffer_push<std::pair<_KeyPair, distance_t>>(
                                {keyPair, pair_gap},
                                local_key_pairs_positions,
                                local_buffer, &local_buffer_end, &local_buffer_size
                        );
                    } else {
                        auto stats_entry_it = local_stats_key_pair.find(keyPair);
                        if (stats_entry_it == local_stats_key_pair.end()) {
                            local_stats_key_pair.insert({keyPair, {1, pair_gap}});
                        } else {
                            stats_entry_it->second.first += 1;
                            if (pair_gap < stats_entry_it->second.second) {
                                stats_entry_it->second.second = pair_gap;
                            }
                        }
                    }
                }

                // update doc_key_triples
                if (window_size <= this->collection_stats->window_size_key_triples_co_occ &&
                    (!B_RESTRICTED || (r_mask & SUITABLE_FOR_TERM_TRIPLE_MASK))) {
                    _KeyPair keyPair(l_match.pattern, r_match.pattern);

                    // middle indicator loop
                    for (size_t m = min_m; m < r; ++m) {
                        const PatternMatch<_Key> m_match = matches_it->at(m);
                        // check if part of the pattern overlaps the siblings
                        // The condition matches_start_pos[m] <= l_match.end_pos is already satisfied by starting from min_m
                        if (l_match.end_pos >= matches_start_pos[m]) {
                            continue;
                        }
                        if (m_match.end_pos >= matches_start_pos[r]) {
                            break;
                        }

                        const distance_t triple_gap = (distance_t) ((matches_start_pos[r] - m_match.end_pos) +
                                                                    (matches_start_pos[m] - l_match.end_pos) - 2);
                        // the - 2 is because the end_pos is included and we want to know the number of words in the middle

                        // this triple will be checked at the end
                        _KeyTriple keyTriple(keyPair, m_match.pattern);
                        if (B_BUFFERED_WORKER) {
                            this->update_fill_local_buffer_push<std::pair<_KeyTriple, distance_t>>(
                                    {_KeyTriple(keyPair, m_match.pattern), triple_gap},
                                    local_key_triples_positions,
                                    local_buffer, &local_buffer_end, &local_buffer_size
                            );
                        } else {
                            auto stats_entry_it = local_stats_key_triple.find(keyTriple);
                            if (stats_entry_it == local_stats_key_triple.end()) {
                                local_stats_key_triple.insert({keyTriple, {1, triple_gap}});
                            } else {
                                stats_entry_it->second.first += 1;
                                if (triple_gap < stats_entry_it->second.second) {
                                    stats_entry_it->second.second = triple_gap;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    inline void
    update_from_local_maps(
            std::unordered_map<_Key, size_t> &local_stats_key,
            std::unordered_map<_KeyPair, std::pair<size_t, distance_t>> &local_stats_key_pair,
            std::unordered_map<_KeyTriple, std::pair<size_t, distance_t>> &local_stats_key_triple
    ) {
        // NOTE: doc_keys and doc_key_pairs are already filtered using the suitable dictionary


        // update key pairs and triples according to the presence inside the document
        if (!B_DISABLE_UNWINDOWED) {
            for (auto l_it = local_stats_key.begin(), end = local_stats_key.end(); l_it != end; ++l_it) {
                for (auto r_it = l_it; ++r_it != end;) {
                    _KeyPair keyPair(l_it->first, r_it->first);

                    if (B_RESTRICTED) {
                        auto st_it = this->suitable_key_pairs.find(keyPair);
                        if (st_it != this->suitable_key_pairs.end()) {
                            char mask = st_it->second;
                            if (mask & SUITABLE_FOR_TERM_PAIR_MASK) {
                                if (local_stats_key_pair.find(keyPair) == local_stats_key_pair.end()) {
                                    local_stats_key_pair.insert({keyPair, {0, (distance_t) -1}});
                                }
                            }
                            if (mask & SUITABLE_FOR_TERM_TRIPLE_MASK) {
                                for (auto m_it = l_it; ++m_it != r_it;) {
                                    _KeyTriple keyTriple(keyPair, m_it->first);
                                    if (local_stats_key_triple.find(keyTriple) == local_stats_key_triple.end()) {
                                        local_stats_key_triple.insert({keyTriple, {0, (distance_t) -1}});
                                    }
                                }
                            }
                        } else {
                            continue;
                        }
                    } else {
                        if (local_stats_key_pair.find(keyPair) == local_stats_key_pair.end()) {
                            local_stats_key_pair.insert({keyPair, {0, (distance_t) -1}});
                        }
                        for (auto m_it = l_it; ++m_it != r_it;) {
                            _KeyTriple keyTriple(keyPair, m_it->first);
                            if (local_stats_key_triple.find(keyTriple) == local_stats_key_triple.end()) {
                                local_stats_key_triple.insert({keyTriple, {0, (distance_t) -1}});
                            }
                        }
                    }
                }
            }
        }

        // update keys
        this->update_lock();
        this->collection_stats->num_docs += 1;
        {
            for (auto stats_entry_it: local_stats_key) {
                key_frequency_t kf = stats_entry_it.second;
                StatsKey statsKey(1, kf, kf * kf);

                // I must check if this key should be considered after the update in the policy used to fill doc_keys
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_into_buffer(stats_entry_it.first, statsKey);
                } else {
                    this->add_key(stats_entry_it.first, statsKey);
                }
            }
        }

        // update key pairs
        {
            for (auto stats_entry_it: local_stats_key_pair) {
                key_frequency_t window_co_occ = stats_entry_it.second.first;
                StatsKeyPair statsKeyPair(
                        (B_DISABLE_UNWINDOWED ? 0 : 1),
                        (window_co_occ > 0 ? 1 : 0),
                        window_co_occ,
                        window_co_occ * window_co_occ,
                        stats_entry_it.second.second
                );

                // I don't need to check if this keyPair must be considered because I know this from r_mask
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_pair_into_buffer(stats_entry_it.first, statsKeyPair);
                } else {
                    this->add_key_pair(stats_entry_it.first, statsKeyPair);
                }
            }
        }

        // update key triples
        {
            for (auto stats_entry_it: local_stats_key_triple) {
                key_frequency_t window_co_occ = stats_entry_it.second.first;
                StatsKeyTriple statsKeyTriple(
                        (B_DISABLE_UNWINDOWED ? 0 : 1),
                        (window_co_occ > 0 ? 1 : 0),
                        window_co_occ,
                        window_co_occ * window_co_occ,
                        stats_entry_it.second.second
                );

                // I must check if this triple should be considered, because from r_mask I know only that two of its keys partecipate to some triple, no more
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_triple_into_buffer(stats_entry_it.first, statsKeyTriple);
                } else {
                    this->add_key_triple(stats_entry_it.first, statsKeyTriple);
                }
            }
        }
        this->update_unlock();
    }

    inline void
    update_from_local_buffer(
            std::vector<char> &local_buffer,
            size_t &local_buffer_end,
            size_t &local_buffer_size,
            std::vector<size_t> &local_keys_positions,
            std::vector<size_t> &local_key_pairs_positions,
            std::vector<size_t> &local_key_triples_positions
    ) {
        // NOTE: doc_keys and doc_key_pairs are already filtered using the suitable dictionary

        // update keys
        std::sort(
                local_keys_positions.begin(),
                local_keys_positions.end(),
                PositionsLessThanPred<_Key>(local_buffer.data())
        );
        size_t cursor_end = 0;
        this->update_lock();
        this->collection_stats->num_docs += 1;

        {
            const char *data = local_buffer.data();
            for (size_t l = 0, r = 0, end = local_keys_positions.size(); l < end; l = r) {
                const _Key *l_key = buffer_get<_Key>(local_keys_positions[l], data);
                r = l + 1;
                while (r < end) {
                    const _Key *r_key = buffer_get<_Key>(local_keys_positions[r], data);
                    if (!std::equal_to<_Key>()(*l_key, *r_key)) {
                        break;
                    }
                    ++r;
                }
                // move the unique keys at the beginning of this buffer
                local_keys_positions[cursor_end++] = local_keys_positions[l];

                key_frequency_t kf = r - l;
                StatsKey statsKey(1, kf, kf * kf);

                // I must check if this key should be considered after the update in the policy used to fill doc_keys
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_into_buffer(*l_key, statsKey);
                } else {
                    this->add_key(*l_key, statsKey);
                }
            }
        }
        this->update_unlock();
        // update key pairs and triples according to the presence inside the document
        if (!B_DISABLE_UNWINDOWED) {
            const StatsKeyPair statsKeyPair(1, 0, 0, 0, 0);
            const StatsKeyTriple statsKeyTriple(1, 0, 0, 0, 0);
            for (size_t l = 0; l < cursor_end; ++l) {
                for (size_t r = l + 1; r < cursor_end; ++r) {
                    _KeyPair keyPair(
                            *buffer_get<_Key>(local_keys_positions[l], local_buffer.data()),
                            *buffer_get<_Key>(local_keys_positions[r], local_buffer.data())
                    );

                    if (B_RESTRICTED) {
                        auto st_it = this->suitable_key_pairs.find(keyPair);
                        if (st_it != this->suitable_key_pairs.end()) {
                            char mask = st_it->second;
                            if (mask & SUITABLE_FOR_TERM_PAIR_MASK) {
                                this->update_fill_local_buffer_push<std::pair<_KeyPair, distance_t>>(
                                        {keyPair, (distance_t) -1},
                                        local_key_pairs_positions,
                                        local_buffer, &local_buffer_end, &local_buffer_size
                                );
                            }
                            if (mask & SUITABLE_FOR_TERM_TRIPLE_MASK) {
                                for (size_t m = l + 1; m < r; ++m) {
                                    const _Key *_m = buffer_get<_Key>(local_keys_positions[m], local_buffer.data());
                                    this->update_fill_local_buffer_push<std::pair<_KeyTriple, distance_t>>(
                                            {_KeyTriple(keyPair, *_m), (distance_t) -1},
                                            local_key_triples_positions,
                                            local_buffer, &local_buffer_end, &local_buffer_size
                                    );
                                }
                            }
                        } else {
                            continue;
                        }
                    } else {
                        this->update_fill_local_buffer_push<std::pair<_KeyPair, distance_t>>(
                                {keyPair, (distance_t) -1},
                                local_key_pairs_positions,
                                local_buffer, &local_buffer_end, &local_buffer_size
                        );
                        for (size_t m = l + 1; m < r; ++m) {
                            const _Key *_m = buffer_get<_Key>(local_keys_positions[m], local_buffer.data());
                            this->update_fill_local_buffer_push<std::pair<_KeyTriple, distance_t>>(
                                    {_KeyTriple(keyPair, *_m), (distance_t) -1},
                                    local_key_triples_positions,
                                    local_buffer, &local_buffer_end, &local_buffer_size
                            );
                        }
                    }
                }
            }
        }

        // update key pairs
        std::sort(
                local_key_pairs_positions.begin(),
                local_key_pairs_positions.end(),
                PositionsKeyValueLessThanPred<_KeyPair, distance_t>(local_buffer.data())
        );
        this->update_lock();
        {
            const char *data = local_buffer.data();

            for (size_t l = 0, r = 0, end = local_key_pairs_positions.size(); l < end; l = r) {
                const std::pair<_KeyPair, distance_t> *l_pair = buffer_get<std::pair<_KeyPair, distance_t>>(
                        local_key_pairs_positions[l], data);
                r = l + 1;

                distance_t min_gap = l_pair->second;
                while (r < end) {
                    const std::pair<_KeyPair, distance_t> *r_pair = buffer_get<std::pair<_KeyPair, distance_t>>(
                            local_key_pairs_positions[r], data);
                    if (!std::equal_to<_KeyPair>()(l_pair->first, r_pair->first)) {
                        break;
                    }
                    if (min_gap > r_pair->second) {
                        min_gap = r_pair->second;
                    }
                    ++r;
                }

                // I don't need to check if this keyPair must be considered because I know this from r_mask
                key_frequency_t window_co_occ = r - l;
                // if the two components are different, one of the occurrences has been added to count the document_frequency by the previous code
                if (l_pair->first.first() != l_pair->first.second()) {
                    window_co_occ -= 1;
                }
                StatsKeyPair statsKeyPair(
                        (B_DISABLE_UNWINDOWED ? 0 : 1),
                        (window_co_occ > 0 ? 1 : 0),
                        window_co_occ,
                        window_co_occ * window_co_occ,
                        min_gap
                );
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_pair_into_buffer(l_pair->first, statsKeyPair);
                } else {
                    this->add_key_pair(l_pair->first, statsKeyPair);
                }
            }
        }
        this->update_unlock();

        // update key triples
        std::sort(
                local_key_triples_positions.begin(),
                local_key_triples_positions.end(),
                PositionsKeyValueLessThanPred<_KeyTriple, distance_t>(local_buffer.data())
        );
        this->update_lock();
        {
            const char *data = local_buffer.data();
            for (size_t l = 0, r = 0, end = local_key_triples_positions.size(); l < end; l = r) {
                const std::pair<_KeyTriple, distance_t> *l_pair = buffer_get<std::pair<_KeyTriple, distance_t>>(
                        local_key_triples_positions[l], data);
                r = l + 1;

                distance_t min_gap = l_pair->second;
                while (r < end) {
                    const std::pair<_KeyTriple, distance_t> *r_pair = buffer_get<std::pair<_KeyTriple, distance_t>>(
                            local_key_triples_positions[r], data);
                    if (!std::equal_to<_KeyTriple>()(l_pair->first, r_pair->first)) {
                        break;
                    }
                    if (min_gap > r_pair->second) {
                        min_gap = r_pair->second;
                    }
                    ++r;
                }

                key_frequency_t window_co_occ = r - l;
                // if the three components are different, one of the occurrences has been added to count the document_frequency by the previous code
                if (l_pair->first.first() != l_pair->first.second() &&
                    l_pair->first.second() != l_pair->first.third()) {
                    window_co_occ -= 1;
                }
                StatsKeyTriple statsKey(
                        (B_DISABLE_UNWINDOWED ? 0 : 1),
                        (window_co_occ > 0 ? 1 : 0),
                        window_co_occ,
                        window_co_occ * window_co_occ,
                        min_gap
                );
                // I must check if this triple should be considered, because from r_mask I know only that two of its keys partecipate to some triple, no more
                if (B_BUFFERED_COLLECTOR) {
                    this->add_key_triple_into_buffer(l_pair->first, statsKey);
                } else {
                    this->add_key_triple(l_pair->first, statsKey);
                }
            }
        }
        this->update_unlock();
    }

    template<typename _T>
    inline void
    update_fill_local_buffer_push(
            const _T &value,
            std::vector<size_t> &positions,
            std::vector<char> &local_buffer,
            size_t *local_buffer_end,
            size_t *local_buffer_size
    ) const {
        // increase the buffer size if it is not enough
        while (*local_buffer_end + sizeof(_T) > *local_buffer_size) {
            *local_buffer_size = *local_buffer_size * 2;
            local_buffer.resize(*local_buffer_size);
        }
        // put the value into the buffer
        *(_T *) (local_buffer.data() + *local_buffer_end) = value;
        // put the position into the related position vector
        positions.push_back(*local_buffer_end);
        // increase the end position
        *local_buffer_end += sizeof(_T);
    }

    inline void
    update_lock() {
        std::unique_lock<std::mutex> lock(this->buffer_stats_mutex);
        while (this->buffer_stats_busy) {
            this->buffer_stats_condition_variable.wait(lock);
        }
        this->buffer_stats_busy = true;
    }

    inline void
    update_unlock() {
        std::unique_lock<std::mutex> lock(this->buffer_stats_mutex);
        this->buffer_stats_busy = false;
        this->buffer_stats_condition_variable.notify_one();
    }

    template<typename _T>
    inline const _T *
    buffer_get(
            size_t position,
            const char *data
    ) {
        return (_T *) (data + position);
    }
};

template<typename Value>
size_t
get_frequency(
        const Value &statsKey
);

template<>
size_t
get_frequency<StatsKey>(
        const StatsKey &statsKey
) {
    return statsKey.frequency;
}

template<>
size_t
get_frequency<StatsKeyPair>(
        const StatsKeyPair &statsKeyPair
) {
    return statsKeyPair.window_frequency;
}

template<>
size_t
get_frequency<StatsKeyTriple>(
        const StatsKeyTriple &statsKeyTriple
) {
    return statsKeyTriple.window_frequency;
}

#endif //COLLECTION_STATS_HPP
