"""Tests for grok_mccodin.rag."""

from __future__ import annotations

from grok_mccodin.rag import TFIDFIndex, _tokenize, search_codebase


class TestTokenize:
    def test_basic_tokens(self):
        tokens = _tokenize("def hello_world():")
        assert "hello_world" in tokens
        assert "def" in tokens

    def test_camel_case_split(self):
        tokens = _tokenize("getUserName")
        assert "getusername" in tokens
        # Sub-tokens from camelCase
        assert "get" in tokens
        assert "user" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_and_symbols(self):
        tokens = _tokenize("x = 123 + y")
        assert "x" in tokens
        assert "y" in tokens


class TestTFIDFIndex:
    def test_index_and_search(self, tmp_path):
        # Create some files
        (tmp_path / "auth.py").write_text(
            "def authenticate_user(username, password):\n"
            "    if username == 'admin':\n"
            "        return True\n"
            "    return check_password(username, password)\n"
        )
        (tmp_path / "db.py").write_text(
            "def query_database(sql):\n"
            "    connection = get_connection()\n"
            "    return connection.execute(sql)\n"
        )
        (tmp_path / "utils.py").write_text(
            "def format_output(data):\n" "    return json.dumps(data, indent=2)\n"
        )

        index = TFIDFIndex()
        count = index.index_folder(tmp_path)
        assert count > 0
        assert index.document_count > 0

        # Search for auth-related code
        results = index.search("authenticate user password")
        assert len(results) > 0
        assert results[0]["path"] == "auth.py"

    def test_search_database(self, tmp_path):
        (tmp_path / "auth.py").write_text("def login(): pass\n")
        (tmp_path / "db.py").write_text(
            "def query_database(sql):\n    connection = get_connection()\n"
        )

        index = TFIDFIndex()
        index.index_folder(tmp_path)

        results = index.search("database query connection")
        assert len(results) > 0
        assert results[0]["path"] == "db.py"

    def test_empty_folder(self, tmp_path):
        index = TFIDFIndex()
        count = index.index_folder(tmp_path)
        assert count == 0

    def test_nonexistent_folder(self):
        index = TFIDFIndex()
        count = index.index_folder("/nonexistent/path")
        assert count == 0

    def test_no_results_for_unrelated_query(self, tmp_path):
        (tmp_path / "math.py").write_text("def add(a, b):\n    return a + b\n")
        index = TFIDFIndex()
        index.index_folder(tmp_path)
        results = index.search("xyzzyplugh")
        assert results == []

    def test_index_text(self):
        index = TFIDFIndex()
        index.index_text("readme", "This project handles user authentication and login flows.")
        results = index.search("authentication login")
        assert len(results) > 0
        assert results[0]["path"] == "readme"

    def test_cosine_similarity_identical(self):
        vec = {"a": 1.0, "b": 2.0}
        sim = TFIDFIndex._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert TFIDFIndex._cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_empty(self):
        assert TFIDFIndex._cosine_similarity({}, {"a": 1.0}) == 0.0


class TestSearchCodebase:
    def test_one_shot(self, tmp_path):
        (tmp_path / "main.py").write_text("def main():\n    print('hello world')\n")
        result = search_codebase(tmp_path, "hello world")
        assert "main.py" in result

    def test_empty_folder(self, tmp_path):
        result = search_codebase(tmp_path, "anything")
        assert "no code files" in result
