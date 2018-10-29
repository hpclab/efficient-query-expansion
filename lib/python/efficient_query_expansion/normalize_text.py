# This Python file uses the following encoding: utf-8
import unicodedata
import re

_rex_control_characters = re.compile(
    r'[\x00-\x09\x0E-\x19][\x00-\x09\x0E-\x20]*')
_rex_spaces = re.compile(
    r'[ ]{2,}')
_rex_new_lines = re.compile(
    r'[ ]?[\n\x0A-\x0D][\n\x0A-\x0D ]*')
_rex_alphanumeric_characters = re.compile(
    r'[ ]?[^ 0-9a-zA-Z][^0-9a-zA-Z]*')
_rex_hyphens = re.compile(
    ur'[\-\_\.\•]')


def normalize_text_step_1(text):
    if isinstance(text, unicode):
        # normalize the UTF8 characters
        text = unicodedata.normalize('NFD', text)

    # remove non ascii characters
    text = text.encode('ascii', 'ignore')
    # remove ascii control characters
    text = _rex_control_characters.sub(' ', text)
    # reduce the number of spaces
    text = _rex_spaces.sub(' ', text)
    # remove trailing space from the lines and reduce the number of new lines
    text = _rex_new_lines.sub('\n', text)
    # remove trailing spaces from the first and the last line
    return text.strip()


def normalize_text_step_2(text):
    # remove non alphanumeric characters collapsing near spaces
    text = _rex_alphanumeric_characters.sub(' ', text)
    # make the text lower and remove trailing spaces
    return text.lower().strip()


def normalize_text(text):
    return normalize_text_step_2(normalize_text_step_1(text))


def normalize_hyphens(text):
    return normalize_text(
        _rex_hyphens.sub('', normalize_text_step_1(text))
    )


def normalize_multiword(text):
    text = normalize_text_step_1(text)
    return normalize_text_step_2(
        ''.join(
            (" " + letter) if (i>0 and letter.isupper() and text[i-1].islower()) else letter
            for i, letter
            in enumerate(text)
        )
    )


def normalize_aliases_raw(aliases_raw, ampersand=True, hyphens=True, multiword=True, acronyms=True):
    # result
    aliases = set()

    # support set
    alias_raw_support = set()
    alias_support = set()
    # normalize all the aliases separately
    for alias_raw in aliases_raw:
        alias_raw_support.clear()
        # raw version
        alias_raw_support.add(alias_raw)

        # raw versione where the character & has been replaced by some alternatives
        if ampersand and "&" in alias_raw:
            alias_raw_splitted = alias_raw.strip().split("&")
            alias_raw_support.update([
                " ".join(alias_raw_splitted),  # because the normalize text removes the & we must overcome this issue
                "".join(alias_raw_splitted),  # because the normalize text removes the & we must overcome this issue
                " and ".join(alias_raw_splitted),
                " n ".join(alias_raw_splitted),
                "n".join(alias_raw_splitted)
            ])

        alias_support.clear()
        # normalization
        for alias_raw in alias_raw_support:
            alias_support.add(normalize_text(alias_raw))
            if hyphens:
                alias_support.add(normalize_hyphens(alias_raw))
            if multiword:
                alias_support.add(normalize_multiword(alias_raw))

        # integration of these aliases into the result set
        aliases.update(alias_support)

    # add acronyms syns
    if acronyms:
        acronyms_support = set()
        for alias in aliases:
            if " " not in alias:
                continue
            # get the initial letters
            initials = [
                letter
                for i, letter in enumerate(alias)
                if i==0 or (alias[i-1] == ' ' and letter != ' ')
            ]
            # create the two acronyms using the initial letters only
            acronyms = (
                ''.join(initials),
                ' '.join(initials)
            )

            # if only one of the two acronyms appear among the aliases I add the other one
            if acronyms[0] in aliases:
                if acronyms[1] not in aliases:
                    acronyms_support.add(acronyms[1])
            else:
                if acronyms[1] in aliases:
                    acronyms_support.add(acronyms[0])

        aliases.update(acronyms_support)

    # discard empty aliases if they are there
    aliases.discard("")

    # assertions
    assert all(("  " not in alias) for alias in aliases)

    # return the Entity object
    return aliases


def get_stopword_set():
    import nltk
    stopwords = set(normalize_text(word)
                    for word in nltk.corpus.stopwords.words('english'))
    stopwords.discard("")
    return stopwords
