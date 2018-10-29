import cPickle
import collections
import json
import socket
import struct

from utils import query_repr_to_sql_query


QueryPerformanceSubset = collections.namedtuple(
    "QueryPerformanceSubset",
    ["num_ret", "exe_time"]
)
QueryPerformance = collections.namedtuple(
    "QueryPerformance",
    ["num_ret", "num_rel", "num_rel_ret", "exe_time"]
)


class SocketChannel(object):
    _length_format = "<I"
    _length_size = struct.calcsize(_length_format)

    def __init__(self, host, port):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))

    def close(self):
        self._sock.close()

    def send_request(self, request):
        assert isinstance(request, dict)
        self._send_msg(json.dumps(request))

    def receive_reply(self):
        return json.loads(self._recv_msg())

    def _recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = b''
        while len(data) < n:
            packet = self._sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _send_msg(self, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack(SocketChannel._length_format, len(msg)) + msg
        self._sock.sendall(msg)

    def _recv_msg(self):
        # Read message length and unpack it into an integer
        raw_msglen = self._recvall(SocketChannel._length_size)
        if not raw_msglen:
            return None
        msglen = struct.unpack(SocketChannel._length_format, raw_msglen)[0]
        # Read the message data
        return self._recvall(msglen)


class IndexCursor(object):
    def __init__(self, index_cache, db_cursor):
        self._index_cache = index_cache
        self._db_cursor = db_cursor

    def close(self):
        self._db_cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def get_performance(self, query_repr, document_id_list=None, document_id_list_key=None,
                         include_time=True, force=False):
        # check parameters
        assert isinstance(query_repr, (list, tuple))
        assert document_id_list is None or (isinstance(document_id_list, (list, tuple)) and len(document_id_list) > 0 and all(isinstance(doc_id, (int, long)) for doc_id in document_id_list))
        assert document_id_list_key is None or isinstance(document_id_list_key, (int, long))
        assert (document_id_list_key is None) == (document_id_list is None)
        assert isinstance(include_time, bool)
        assert isinstance(force, bool)

        zero_document_id_list = (document_id_list is None) or (len(document_id_list) == 0)

        # transform the query representation in a query string
        sql_str = query_repr_to_sql_query(query_repr)

        # get the entry from the cache
        key = sql_str if zero_document_id_list else (sql_str, document_id_list_key)
        if not force:
            query_performance = self._index_cache._get(key)
            if query_performance is not None and (not include_time or query_performance.exe_time is not None):
                return query_performance

        # transform the document_id_list
        document_id_list = [] if document_id_list is None else list(set(document_id_list))

        request = {
            "query": sql_str,
            "query_type": "cnf"
        }
        if not zero_document_id_list:
            request["rel"] = document_id_list
        self._db_cursor.send_request(request)

        result = self._db_cursor.receive_reply()
        if "error" in result:
            raise Exception(result["error"])

        # compose the resulting object
        if zero_document_id_list:
            query_performance = QueryPerformanceSubset(
                num_ret=int(result["num_ret"]),
                exe_time=float(result["exe_time"])
            )
        else:
            query_performance = QueryPerformance(
                num_ret=int(result["num_ret"]),
                num_rel=int(result["num_rel"]),
                num_rel_ret=int(result["num_rel_ret"]),
                exe_time=float(result["exe_time"])
            )

        # put the result into the cache
        self._index_cache._put(key, query_performance)
        if not zero_document_id_list:
            qps = self._index_cache._get(key[0])
            if qps is None or (include_time and qps.exe_time is None):
                qps = QueryPerformanceSubset(
                    num_ret=query_performance.num_ret,
                    exe_time=query_performance.exe_time
                )
                self._index_cache._put(key[0], qps)

        # return
        return query_performance


class IndexCache(object):
    def __init__(self, host, port):
        assert isinstance(host, str)
        assert isinstance(port, int)

        self._host = host
        self._port = port
        self._cache = dict()

    @staticmethod
    def load(file_path):
        host, port, cache = cPickle.load(open(file_path, "rb"))
        index_cache = IndexCache(host, port)
        index_cache._cache = cache
        return index_cache

    def dump(self, file_path):
        cPickle.dump(
            (self._host, self._port, self._cache),
            open(file_path, "wb"),
            protocol=cPickle.HIGHEST_PROTOCOL
        )

    def __len__(self):
        return len(self._cache)

    def _get(self, key):
        return self._cache.get(key, None)

    def _put(self, key, value):
        self._cache[key] = value

    def cursor(self):
        connection = SocketChannel(host=self._host, port=self._port)
        return IndexCursor(self, connection)
