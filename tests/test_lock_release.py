import fcntl

from tools.dk.core import util


def _free(path) -> bool:
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(f, fcntl.LOCK_UN)
        return True


def test_released_gives_the_lock_up_and_takes_it_back(tmp_path):
    lock = tmp_path / "build" / ".dk.lock"
    with util.project_lock(tmp_path):
        assert not _free(lock)
        with util.released():
            assert _free(lock)
        assert not _free(lock)
    assert _free(lock)


def test_released_without_the_lock_is_a_no_op(tmp_path):
    with util.released():
        pass
