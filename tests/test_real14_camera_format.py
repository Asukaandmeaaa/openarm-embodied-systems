"""Offline format negotiation checks; no camera is opened."""
import pytest
from openarm_kinesthetic_recorder import wrist_camera_node as camera


@pytest.mark.parametrize('format',['MJPG','YUYV'])
def test_requested_format_and_size_before_capture(monkeypatch,format):
    class Capture:
        def __init__(self):self.values={};self.released=False
        def isOpened(self):return True
        def set(self,key,value):self.values[key]=value;return True
        def get(self,key):return self.values.get(key,0)
        def release(self):self.released=True
    capture=Capture()
    monkeypatch.setattr(camera.cv2,'VideoCapture',lambda *args:capture)
    monkeypatch.setattr(camera.threading.Thread,'start',lambda self:None)
    node=camera.CaptureThread('/fake',640,360,30.,format)
    assert capture.values[camera.cv2.CAP_PROP_FOURCC]==camera.cv2.VideoWriter_fourcc(*format)
    assert not capture.released and node.sequence==0


def test_rejected_format_does_not_start_capture_thread(monkeypatch):
    class Capture:
        released=False
        def isOpened(self):return True
        def set(self,*args):return False
        def get(self,key):return 0
        def release(self):self.released=True
    capture=Capture()
    monkeypatch.setattr(camera.cv2,'VideoCapture',lambda *args:capture)
    monkeypatch.setattr(camera.threading.Thread,'start',lambda self:pytest.fail('must not start'))
    with pytest.raises(RuntimeError,match='format negotiation failed'):
        camera.CaptureThread('/fake',640,360,30.,'YUYV')
    assert capture.released
