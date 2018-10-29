def query_repr_to_sql_query(query_repr, uniq_repr=True):
    join_fun = (
        lambda l, m, r, it: "{}{}{}".format(
            l,
            m.join(sorted(set(it)) if uniq_repr else it),
            r
        )
    )

    return \
        join_fun("(", ") | (", ")", (
            join_fun("(", ") (", ")", (
                join_fun("", " | ", "", (
                    "\"{}\"".format(syn_tag[0]) if " " in syn_tag[0] else syn_tag[0]
                    for syn_tag in synset
                ))
                for synset in and_query
            ))
            for and_query in query_repr
        ))


def sql_query_to_query_repr(sql_query):
    assert sql_query[:2] == "((" and sql_query[-2:] == "))"

    query_repr = \
        [
            [
                [
                    (syn[1:-1] if (syn[0] == syn[-1] == "\"") else syn, )
                    for syn in synset.split(" | ")
                ]
                for synset in and_query.split(") (")
            ]
            for and_query in sql_query[2:-2].split(")) | ((")
        ]

    assert all(
        " " not in syn_tag[0] or syn_tag[0].find("\"", 1, -1) == -1
        for and_query in query_repr
        for synset in and_query
        for syn_tag in synset
    )
    return query_repr
