"""cif_references: the references a CIF block states, as ReferenceRecord objects."""

import logging
import unicodedata
from pathlib import Path

import pytest
from httk.core import PersonRecord, ReferenceRecord
from httk.core.report import collect_reports

from httk.atomistic import CIFStructure, cif_references
from httk.atomistic.io.cif import CIFBlock, CIFDataItem, read_cif

COD = Path(__file__).parent / "fixtures" / "cod"
LOGGER = "httk.atomistic.io.cif.cif_references"


def _block(text: str) -> CIFBlock:
    return read_cif(text.splitlines(), lossless=True)[0]


def _refs(text: str) -> tuple[ReferenceRecord, ...]:
    return cif_references(_block(text))


def _person(name: str) -> PersonRecord:
    last, first = name.split(", ")
    return PersonRecord(name, firstname=first, lastname=last)


def test_publication_with_author_loop_and_multiline_title():
    (ref,) = _refs("""data_x
loop_
_publ_author_name
'Smith, John'
'Doe, Jane'
_publ_section_title
;
 A   long
 title   over lines
;
_journal_name_full 'Journal of Tests'
_journal_name_abbrev 'J. Tests'
_journal_volume 12
_journal_issue 3
_journal_year 2001
_journal_page_first 007
_journal_page_last 019
_journal_paper_doi 10.1000/xyz
_journal_coden_ASTM JTSTAB
""")
    assert ref == ReferenceRecord(
        title="A long title over lines",
        journal="Journal of Tests",
        volume="12",
        number="3",
        year="2001",
        pages="007-019",
        doi="10.1000/xyz",
        bib_type="article",
        authors=(_person("Smith, John"), _person("Doe, Jane")),
    )


def test_single_author_abbrev_fallback_and_partial_pages():
    (ref,) = _refs("""data_x
_publ_author_name 'Smith, John'
_journal_name_full ?
_journal_name_abbrev 'J. Tests'
_journal_page_last 19
""")
    assert ref == ReferenceRecord(journal="J. Tests", pages="19", bib_type="article", authors=(_person("Smith, John"),))
    (ref,) = _refs("data_x\n_publ_section_title 'Untitled'\n_journal_page_first e123\n")
    assert ref == ReferenceRecord(title="Untitled", pages="e123")  # no journal: no bib_type


def test_citation_loop_with_author_and_editor_loops():
    refs = _refs("""data_x
_publ_section_title 'The paper'
loop_
_citation_id
_citation_title
_citation_journal_full
_citation_journal_abbrev
_citation_journal_volume
_citation_journal_issue
_citation_year
_citation_page_first
_citation_page_last
_citation_DOI
_citation_language
_citation_journal_id_ASTM
primary 'First cited' 'Acta Tests' 'Acta T.' 5 2 1999 10 20 10.1/a English ACTAB
2 'A book chapter' ? ? ? ? 2005 . . ? ? ?
3 ? . 'Short J.' 7 ? ? 33 ? ? ? ?
loop_
_citation_book_title
_citation_book_publisher
_citation_book_publisher_city
_citation_special_details
_citation_id
'The Book' 'Big Press' 'Cambridge' 'second print' 4
loop_
_citation_author_citation_id
_citation_author_name
_citation_author_ordinal
primary 'Alpha, A.' 1
2 'Beta, B.' 1
primary 'Gamma, G.' 2
loop_
_citation_editor_citation_id
_citation_editor_name
2 'Editor, E.'
""")
    assert refs == (
        ReferenceRecord(title="The paper"),
        ReferenceRecord(
            title="First cited",
            journal="Acta Tests",
            volume="5",
            number="2",
            year="1999",
            pages="10-20",
            doi="10.1/a",
            bib_type="article",
            authors=(_person("Alpha, A."), _person("Gamma, G.")),
        ),
        ReferenceRecord(
            title="A book chapter", year="2005", authors=(_person("Beta, B."),), editors=(_person("Editor, E."),)
        ),
        ReferenceRecord(journal="Short J.", volume="7", pages="33", bib_type="article"),
        ReferenceRecord(booktitle="The Book", publisher="Big Press", address="Cambridge", note="second print"),
    )


def test_single_row_citation_and_people_without_citation_row():
    refs = _refs("""data_x
_citation_title 'Only citation'
_citation_publisher 'Data Centre'
_citation_author_name 'Solo, S.'
loop_
_citation_editor_citation_id
_citation_editor_name
9 'Orphan, O.'
""")
    assert refs == (
        ReferenceRecord(title="Only citation", publisher="Data Centre", authors=(_person("Solo, S."),)),
        ReferenceRecord(editors=(_person("Orphan, O."),)),
    )


def test_person_names_and_placeholders():
    (ref,) = _refs("""data_x
loop_
_publ_author_name
'Smith, John'
'John Smith'
'Smith, John, Jr.'
'Smith,'
?
.
''
'  van  der   Berg ,  Anna  '
""")
    assert ref.authors == (
        _person("Smith, John"),
        PersonRecord("John Smith"),
        PersonRecord("Smith, John, Jr."),
        PersonRecord("Smith,"),
        PersonRecord("van der Berg , Anna", firstname="Anna", lastname="van der Berg"),
    )
    assert _refs("data_x\n_publ_section_title ?\n_journal_year .\n_publ_author_name ?\n") == ()


ACCENTS = {
    "\\'e": "é",
    "\\`a": "à",
    "\\^o": "ô",
    '\\"u': "ü",
    "\\~n": "ñ",
    "\\,c": "ç",
    "\\;a": "ą",
    "\\>o": "ő",
    "\\=a": "ā",
    "\\.z": "ż",
    "\\<c": "č",
    "\\(g": "ğ",
    "\\'E": "É",
}
SPECIAL_LETTERS = {
    "\\%a": "å",
    "\\%A": "Å",
    "\\/o": "ø",
    "\\/O": "Ø",
    "\\/l": "ł",
    "\\/L": "Ł",
    "\\/d": "đ",
    "\\/D": "Đ",
    "\\?i": "ı",
    "\\&s": "ß",
}
GREEK = dict(zip("abgdezhqiklmnxoprstufcyw", "αβγδεζηθικλμνξοπρστυφχψω", strict=True))
# Not codes: kept exactly as written.
UNDEFINED = [
    "\\^2",
    "\\'.",
    "\\j",
    "\\v",
    "\\%u",
    "\\%\u00e9",
    "\\\\db",
    "\\\\langle",
    "\\'",
    "\\&t",
    "x~2~",
    "y^3^",
    "--",
    "+-",
    "\\",
]


@pytest.mark.parametrize(
    ("written", "decoded"),
    [
        *ACCENTS.items(),
        *SPECIAL_LETTERS.items(),
        *((f"\\{code}", letter) for code, letter in GREEK.items()),
        *((f"\\{code.upper()}", letter.upper()) for code, letter in GREEK.items()),
        ("\\%", "\u00b0"),
        ("10\\%.", "10\u00b0."),
        *((code, code) for code in UNDEFINED),
    ],
)
def test_special_character_codes(written, decoded):
    (ref,) = _refs(f"data_x\n_publ_section_title\n;\nA {written} Z\n;\n")
    assert ref.title == f"A {decoded} Z"


def test_decoding_in_context_and_not_in_dois():
    (ref,) = _refs("""data_x
_publ_section_title 'Ni\\<zn\\'y: \\a-Fe~2~O~3~ at 10\\% \\\\db x\\^2 caf\\'\u00e9 caf\\'e\u0301 \\'\\a \\'\u0142'
_publ_author_name 'R\\%u\\<zi\\<cka, Ale\\<s'
_journal_paper_doi '10.1000/a\\b'
""")
    # An accent combines only with an unaccented base letter that no combining mark follows.
    assert ref.title == "Nižný: α-Fe~2~O~3~ at 10° \\\\db x\\^2 caf\\'\u00e9 caf\\'e\u0301 \\'α \u0142\u0301"
    assert ref.authors == (PersonRecord("R\\%užička, Aleš", firstname="Aleš", lastname="R\\%užička"),)
    assert ref.authors is not None and ref.authors[0].name == unicodedata.normalize("NFC", ref.authors[0].name)
    assert ref.doi == "10.1000/a\\b"


def test_exact_duplicates_are_removed():
    refs = _refs("""data_x
_publ_section_title 'Same'
_journal_name_full 'J'
loop_
_citation_id
_citation_title
_citation_journal_full
1 'Same' 'J'
2 'Same' 'J'
3 'Other' 'J'
""")
    assert refs == (
        ReferenceRecord(title="Same", journal="J", bib_type="article"),
        ReferenceRecord(title="Other", journal="J", bib_type="article"),
    )


def test_empty_reference_is_logged_with_its_data(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        refs = _refs("""data_blk
_journal_coden_ASTM JTSTAB
_journal_year ?
loop_
_citation_id
_citation_journal_id_ASTM
_citation_language
c1 ACTAB English
""")
    assert refs == ()
    messages = [record.getMessage() for record in caplog.records]
    assert all(vars(record)["context"] == "cif" and record.levelno == logging.INFO for record in caplog.records)
    assert len(messages) == 2
    assert "'blk'" in messages[0] and "could not be represented" in messages[0]
    assert "_journal_coden_ASTM = 'JTSTAB'" in messages[0] and "_journal_year" not in messages[0]
    assert "_citation_id = 'c1'; _citation_journal_id_ASTM = 'ACTAB'; _citation_language = 'English'" in messages[1]


def test_unrepresentable_citations_do_not_raise():
    # A CIF2 list as a title and a repeated publication item; the rest is still returned.
    block = CIFBlock(
        "x",
        (
            CIFDataItem("_publ_section_title", "One"),
            CIFDataItem("_publ_section_title", "Two"),
            CIFDataItem("_citation_title", ("a", "b")),
            CIFDataItem("_citation_year", "1990"),
        ),
        dialect="mcif",
    )
    with collect_reports("info") as reports:
        assert cif_references(block) == ()
    messages = [record.getMessage() for record in reports.records]
    assert len(messages) == 2
    assert "_publ_section_title is given 2 times" in messages[0]
    assert "_publ_section_title = 'One'; _publ_section_title = 'Two'" in messages[0]
    assert "_citation_title = \"('a', 'b')\"; _citation_year = '1990'" in messages[1]
    assert all(vars(record)["context"] == "cif" for record in reports.records)


def test_cod_entries(caplog):
    (cs,) = [CIFStructure.from_cif(COD / "1509425.cif")]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert cif_references(cs) == (
            ReferenceRecord(
                title=(
                    "Supercritical ammonia synthesis and characterization of four new alkali metal silver "
                    "antimony sulfides: M Ag2 Sb S4 and M2 Ag Sb S4 (M=K, Rb)"
                ),
                journal="Journal of Solid State Chemistry",
                volume="123",
                year="1996",
                pages="277-284",
                doi="10.1006/jssc.1996.0179",
                bib_type="article",
                authors=tuple(map(_person, ["Schimek, G.L.", "Wood, P.T.", "Pennington, W.T.", "Kolis, J.W."])),
            ),
        )
    # COD's lone _citation_journal_id_ASTM fits no field.
    (message,) = [record.getMessage() for record in caplog.records]
    assert "_citation_journal_id_ASTM = 'JSSCBI'" in message

    assert cif_references(CIFStructure.from_cif(COD / "4326344.cif")) == (
        ReferenceRecord(
            title=(
                "Tetrylenes Chelated by Hybrid Amido-Amino Ligand: Derivatives of 2-[(N,N-Dimethylamino)methyl]aniline"
            ),
            journal="Inorganic Chemistry",
            volume="50",
            year="2011",
            pages="9454-9464",
            doi="10.1021/ic2011056",
            bib_type="article",
            authors=tuple(
                map(
                    PersonRecord,
                    [
                        "Hana Vaňkátová",
                        "Lies Broeckaert",
                        "Frank De Proft",
                        "Roman Olejník",
                        "Jan Turek",
                        "Zdeňka Padělková",
                        "Aleš R\\%užička",  # \%u is not a CIF 1.1 code
                    ],
                )
            ),
        ),
    )

    (ref,) = cif_references(CIFStructure.from_cif(COD / "7219855.cif"))
    assert ref.title is not None and ref.title.startswith("Effect of π--π stacking interactions")
    assert (ref.journal, ref.volume, ref.number, ref.year, ref.pages, ref.doi) == (
        "CrystEngComm",
        "17",
        "19",
        "2015",
        "3659",
        "10.1039/C5CE00010F",
    )
    assert ref.authors is not None
    assert ref.authors[1] == PersonRecord("San Sebastián, Eider", firstname="Eider", lastname="San Sebastián")


def test_people_without_id_and_several_citations_without_id():
    # Id-less people go with an id-less citation only when it is the only one.
    text = """data_x
loop_
_citation_title
'First'
'Second'
_citation_author_name 'Nobody, N.'
"""
    assert _refs(text) == (
        ReferenceRecord(title="First"),
        ReferenceRecord(title="Second"),
        ReferenceRecord(authors=(_person("Nobody, N."),)),
    )
