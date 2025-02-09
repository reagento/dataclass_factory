from textwrap import dedent

from adaptix._internal.code_tools.code_gen_tree import (
    CodeBlock,
    DictKeyValue,
    DictLiteral,
    LinesWriter,
    ListLiteral,
    MappingUnpack,
    RawExpr,
    Statement,
)


def assert_string(stmt: Statement, string: str) -> str:
    writer = LinesWriter()
    stmt.write_lines(writer)
    assert writer.make_string() == dedent(string).strip()


def test_dict_literal():
    assert_string(
        DictLiteral(
            [
                DictKeyValue(RawExpr("a"), RawExpr("1")),
                DictKeyValue(RawExpr("b"), RawExpr("2")),
                MappingUnpack(RawExpr("c")),
            ],
        ),
        """
        {
            a: 1,
            b: 2,
            **c,
        }
        """,
    )


def test_dict_literal_nested():
    assert_string(
        DictLiteral(
            [
                DictKeyValue(RawExpr("a"), RawExpr("1")),
                DictKeyValue(
                    RawExpr("b"),
                    DictLiteral(
                        [
                            DictKeyValue(RawExpr("c"), RawExpr("2")),
                            DictKeyValue(RawExpr("d"), RawExpr("3")),
                        ],
                    ),
                ),
                MappingUnpack(RawExpr("e")),
            ],
        ),
        """
        {
            a: 1,
            b: {
                c: 2,
                d: 3,
            },
            **e,
        }
        """,
    )


def test_list_literal():
    assert_string(
        ListLiteral(
            [
                RawExpr("a"),
                RawExpr("b"),
            ],
        ),
        """
        [
            a,
            b,
        ]
        """,
    )


def test_code_block():
    assert_string(
        CodeBlock(
            """
            if <condition>:
                <true_case>
            else:
                <false_case>
            """,
            condition=RawExpr("zzz"),
            true_case=ListLiteral(
                [
                    RawExpr("a"),
                    RawExpr("b"),
                ],
            ),
            false_case=DictLiteral(
                [
                    DictKeyValue(RawExpr("a"), RawExpr("1")),
                    DictKeyValue(RawExpr("b"), RawExpr("2")),
                ],
            ),
        ),
        """
        if zzz:
            [
                a,
                b,
            ]
        else:
            {
                a: 1,
                b: 2,
            }
        """,
    )
