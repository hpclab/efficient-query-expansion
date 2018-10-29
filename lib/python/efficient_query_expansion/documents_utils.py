import codecs
import gzip
import nltk
import os.path
import sys

from normalize_text import normalize_text, normalize_text_step_1
from parallel_stream.utils import get_emitter_from_iterable


class Doc(object):
    def __init__(self, id=None, title=None, content_lines=None):
        self.id = id
        self.title = title or ""
        self.content_lines = content_lines or []


class ExtendedDoc(Doc):
    def __init__(self, id=None, title=None, content_lines=None,
                 url=None, keywords=None, description=None):
        super(ExtendedDoc, self).__init__(id, title, content_lines)
        self.url = url or ""
        self.keywords = keywords or ""
        self.description = description or ""


def _custom_reader_to_doc_generator(reader):
    doc_id = None
    doc_title = None
    doc_content_lines = []

    step = 0
    for line in reader:
        if step == 0:  # document start

            if line.startswith("<doc ") or line.rstrip().isdigit():
                doc_id = line.strip()
                step = 1
            else:
                if line.strip() != "":
                    raise Exception("A <doc> tag or a number was expected, instead a \"{}\" has been found".format(
                        line.rstrip("\n")))

        elif step == 1:  # document title

            doc_title = line.strip()
            step = 2

        elif step == 2:  # document content/end

            line = line.rstrip("\n")

            if line != "":
                # document content
                doc_content_lines.append(line)

            else:
                # document end
                yield(Doc(doc_id, doc_title, doc_content_lines))
                doc_id = None
                doc_content_lines = []
                step = 0

        continue

    if doc_id is not None:
        raise Exception("A content was expected before the end of file")


def _wiki_extractor_reader_to_doc_generator(reader):
    doc_id = None
    doc_title = None
    doc_content_lines = []

    step = 0
    for line in reader:
        if step == 0:  # document start

            if line.startswith("<doc "):
                doc_id = line.strip()
                step = 1
            else:
                if line.strip() != "":
                    raise Exception("A <doc> tag was expected, instead a \"{}\" has been found".format(
                        line.rstrip("\n")))

        elif step == 1:  # document title

            doc_title = line.strip()
            step = 2

        elif step == 2:  # empty line

            if line != "\n":
                raise Exception("An empty line was expected")
            step = 3

        elif step == 3:  # document content/end

            if not line.startswith("</doc>"):
                # document content
                doc_content_lines.append(line.rstrip("\n"))

            else:
                # document end
                yield(Doc(doc_id, doc_title, doc_content_lines))
                doc_id = None
                doc_content_lines = []
                step = 0

        continue

    if doc_id is not None:
        raise Exception("A content was expected before the end of file")


def _xml_extractor_reader_to_doc_generator(reader):
    doc = None

    step = 0
    for original_line in reader:
        line = original_line.strip()

        if step == 0:  # document start
            if line.startswith("<sphinx:document id='"):
                doc = ExtendedDoc(line[(line.find("'") + 1):line.rfind("'")])
                step = 1
            elif len(line) != 0:
                raise Exception(
                    "A <sphinx:document> tag was expected, instead a \"{}\" has been found".format(line))

        elif step == 1:  # document tags
            content = line[(line.find(">") + 1):(line.rfind("</"))]

            if line.startswith("<url>"):
                doc.url = content
            if line.startswith("<title>"):
                doc.title = content
            elif line.startswith("<keywords>"):
                doc.keywords = content
            elif line.startswith("<description>"):
                doc.description = content
            elif line == "<content>":
                step = 2
            elif line.startswith("</sphinx:document"):
                raise Exception(
                    "A <title> tag was expected, instead a \"{}\" has been found".format(line))

        elif step == 2:  # document content

            if line == "</content>":
                # document end
                yield(doc)
                doc = None
                step = 4
            else:
                # document content
                doc.content_lines.append(original_line.rstrip("\n"))

        elif step == 4:  # document end

            if line.startswith("</sphinx:document"):
                step = 0

        else:
            raise Exception("Programming error: step not recognized")

    if doc is not None:
        raise Exception("A content was expected before the end of file")


def get_reader(infilename, encoding=None):
    if infilename == "-":
        reader = sys.stdin
    else:
        if not os.path.isfile(infilename):
            raise Exception("File {} doesn't exist".format(infilename))

        if infilename.endswith(".gz"):
            reader = gzip.open(infilename, "rb")
        else:
            reader = open(infilename, "r")

    if encoding is None or encoding.upper() == "ASCII":
        return reader

    return codecs.getreader(encoding)(reader)


def doc_generator_from_file(infilenames, encoding=None, file_format="custom"):
    if isinstance(infilenames, (str, unicode)):
        infilenames = [infilenames]
    elif hasattr(infilenames, "__iter__"):
        pass
    else:
        raise Exception(
            "The first parameter must be a filename or an iterator/generator of filenames")

    if file_format == "custom":
        generator = _custom_reader_to_doc_generator
    elif file_format == "wiki":
        generator = _wiki_extractor_reader_to_doc_generator
    elif file_format == "xml":
        generator = _xml_extractor_reader_to_doc_generator
    else:
        raise Exception(
            "The file_format parameter must be one between 'custom' or 'wiki'")

    for infilename in infilenames:
        with get_reader(infilename, encoding) as reader:
            for doc in generator(reader):
                yield doc


def sentence_generator_from_doc_file(*args, **kwargs):
    for doc in doc_generator_from_file(*args, **kwargs):
        yield(doc.title)
        for line in doc.content_lines:
            yield(line)


def get_doc_emitter_from_files(*args, **kwargs):
    return get_emitter_from_iterable(doc_generator_from_file(*args, **kwargs))


def get_doc_normalizer_worker():
    def worker(worker_id, job_iterator, outqueue):
        for doc in job_iterator:
            # clean the id to put it into an ASCII file and normalize the title
            doc.id = normalize_text_step_1(doc.id).replace('\n', ' ')
            doc.title = normalize_text(doc.title).replace('\n', ' ') or "_"

            # split the content in sentences and normalize the text
            # then remove empty lines
            doc.content_lines = filter(len,
                                       [normalize_text(sent)
                                        for line in doc.content_lines
                                        for sent in nltk.sent_tokenize(line)])

            # put the result on the output queue
            outqueue.put('\n'.join([doc.id, doc.title] + doc.content_lines))

    return worker
