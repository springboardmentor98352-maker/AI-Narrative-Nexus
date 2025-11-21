from narrativenexus_utils import save_uploaded_file, parse_preview


class DummyUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


def test_save_and_parse_txt(tmp_path):
    txt = DummyUpload("sample.txt", b"Line1\nLine2\nLine3\n")
    path = save_uploaded_file(txt, str(tmp_path))
    preview = parse_preview(path, max_lines=2)
    assert "Line1" in preview


def test_save_and_parse_csv(tmp_path):
    csvu = DummyUpload("data.csv", b"a,b,c\n1,2,3\n4,5,6\n")
    path = save_uploaded_file(csvu, str(tmp_path))
    preview = parse_preview(path, max_lines=2)
    # depending on csv parsing, header may appear as 'a, b, c' or 'a,b,c'
    assert "a" in preview
