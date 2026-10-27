from unittest.mock import Mock
import pytest
from scripts.control_ownership import client


def test_revoked_lease_allows_cancelled_child_to_flush_report(monkeypatch):
    authority=Mock();authority.token='test';authority.renew.side_effect=ValueError('no live authority for this connection')
    context=Mock();context.__enter__=Mock(return_value=authority);context.__exit__=Mock(return_value=False)
    monkeypatch.setattr(client,'Client',lambda *args:context)
    child=Mock();child.poll.side_effect=[None,None];child.returncode=1
    monkeypatch.setattr(client.subprocess,'Popen',lambda *args,**kwargs:child)
    monkeypatch.setattr('sys.argv',['client','run','--owner','h2','--','test-child'])
    with pytest.raises(SystemExit) as stopped:client.main()
    assert stopped.value.code==1
    child.wait.assert_called_once_with(timeout=12.)
    child.terminate.assert_not_called()
