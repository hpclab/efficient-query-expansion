import codecs
import gzip
import os.path
import sys


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


def get_writer(outfilename, encoding=None, check_if_exists=True):
    if outfilename == "-":
        writer = sys.stdout
    else:
        if check_if_exists and os.path.isfile(outfilename):
            raise Exception("File {} already exists".format(outfilename))

        if outfilename.endswith(".gz"):
            writer = gzip.open(outfilename, "wb")
        else:
            writer = open(outfilename, "w")

    if encoding is None or encoding.upper() == "ASCII":
        return writer

    return codecs.getwriter(encoding)(writer)


def get_emitter_from_generator(generator):
    assert hasattr(generator, "__iter__")

    def emitter(outqueue):
        for job in generator:
            outqueue.put(job)

    return emitter


def get_collector_pairs_to_dict(dict_object=None):
    def collector(inqueue):
        if dict_object is None:
            dict_object = dict()
        for key, value in inqueue:
            dict_object[key] = value
        return dict_object

    return collector


def get_text_collector_to_file(outfilename,
                               encoding=None,
                               separator="\n",
                               buffer_size=64 * 1024):
    def collector(job_iterator):
        buffer = []
        buffer_length = 0

        with get_writer(outfilename, encoding) as writer:
            for text in job_iterator:
                # put the text into the buffer
                buffer.append(text)
                buffer_length += len(text)

                # write only when the buffer is full above a certain threshold
                if buffer_length >= buffer_size:
                    writer.write(separator.join(buffer) + separator)
                    buffer = []
                    buffer_length = 0

            # empty the buffer since we have finished
            if buffer_length > 0:
                writer.write(separator.join(buffer) + separator)
            writer.flush()

    return collector
