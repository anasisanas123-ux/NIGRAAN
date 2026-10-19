import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import config
from python_engine.face_quality import quality, FaceEvidence
from python_engine.zone_store import ZoneStore
from python_engine.arcface_recognizer import ArcFaceRecognizer


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).integers(50,200,(200,200,3),dtype=np.uint8)
        self.face = SimpleNamespace(bbox=np.array([10,10,110,110]),det_score=.99,pose=np.zeros(3))

    def test_rejects_blur_dark_tiny_and_profile(self):
        self.assertTrue(quality(self.face,self.image,config))
        self.assertFalse(quality(self.face,np.full_like(self.image,120),config))
        self.assertFalse(quality(self.face,np.zeros_like(self.image),config))
        self.face.bbox = np.array([0,0,40,40])
        self.assertFalse(quality(self.face,self.image,config))
        self.face.bbox = np.array([10,10,110,110]); self.face.pose[1] = 70
        self.assertFalse(quality(self.face,self.image,config))

    def test_temporal_evidence_isolated_reset_and_one_to_one(self):
        evidence = FaceEvidence(); box = [0,0,100,100]
        self.assertEqual(evidence.confirm('a',[(box,'S1')],0,now=0),[False])
        self.assertEqual(evidence.confirm('a',[(box,'S1')],0,now=.4),[False])
        self.assertEqual(evidence.confirm('b',[(box,'S1')],0,now=.5),[False])
        self.assertEqual(evidence.confirm('a',[(box,'S1'),(box,'S1')],0,now=.8),[True,False])
        self.assertEqual(evidence.confirm('a',[(box,'S2')],0,now=1),[False])
        self.assertEqual(evidence.confirm('a',[(box,'S2')],1,now=1.2),[False])
        self.assertEqual(evidence.confirm('a',[(box,'S2')],1,now=4),[False])

    def test_crop_coordinates_and_deduplication(self):
        rec = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
        rec.crop_times = {}; rec.app = Mock()
        cropped = SimpleNamespace(bbox=np.array([10,20,70,80],dtype=float))
        rec.app.get.side_effect = [[],[cropped]]
        faces = rec._detect_faces(self.image,'a',[[80,80,150,190]])
        np.testing.assert_equal(faces[0].bbox,[80,90,140,150])
        rec.app.get.side_effect = [[cropped]]
        self.assertEqual(len(rec._detect_faces(self.image,'a',[[80,80,150,190]])),1)
        self.assertEqual(rec.app.get.call_count,3)


class ZoneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name)/'zones.json')
        self.settings = SimpleNamespace(CAMERA_SOURCES={'a':0,'b':1},DETECTION_ZONE={},BEHAVIOR_RESTRICTED_ZONES={})
        self.store = ZoneStore(self.settings,self.path)
        self.value = {'recognition':[.1,.2,.8,.9],'restricted':[{'id':'door','name':'Door','box':[.2,.3,.4,.5],'dwell_seconds':2}]}

    def test_persistence_isolation_and_clear(self):
        self.store.save('a',self.value)
        self.assertEqual(self.store.get('b'),{'recognition':None,'restricted':[]})
        copy = self.store.get('a'); copy['restricted'].clear()
        self.assertEqual(len(self.store.get('a')['restricted']),1)
        restored = ZoneStore(self.settings,self.path)
        self.assertEqual(restored.get('a'),self.value)
        restored.save('a',{'recognition':None,'restricted':[]})
        self.assertIsNone(self.settings.DETECTION_ZONE['a'])

    def test_invalid_boxes_and_unknown_camera_do_not_write(self):
        for box in ([0,0,2,1],[.5,0,.2,1],[0,0,float('nan'),1],[True,0,1,1]):
            with self.assertRaises(ValueError):
                self.store.save('a',{'recognition':box})
        with self.assertRaises(ValueError): self.store.save('other',self.value)
        self.assertFalse(Path(self.path).exists())

    def test_failed_disk_write_preserves_active_and_saved_configuration(self):
        self.store.save('a',self.value)
        with patch('python_engine.zone_store.os.replace',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError): self.store.save('a',{'recognition':None,'restricted':[]})
        self.assertEqual(self.store.get('a'),self.value)
        self.assertEqual(self.settings.DETECTION_ZONE['a'],self.value['recognition'])
        self.assertEqual(json.loads(Path(self.path).read_text())['a'],self.value)

    def test_api_validation_local_access_and_reset(self):
        from python_engine import engine_api
        with patch.object(engine_api,'zone_store',self.store), patch.object(engine_api.behavior_monitor,'reset_camera') as reset:
            client = engine_api.app.test_client()
            self.assertEqual(client.put('/zones/a',json=self.value).status_code,200)
            reset.assert_called_once_with('a')
            self.assertEqual(client.get('/zones/a').json['data'],self.value)
            self.assertEqual(client.put('/zones/a',json={'recognition':[0,0,4,1]}).status_code,400)
            self.assertEqual(client.put('/zones/a',json=self.value,environ_base={'REMOTE_ADDR':'10.0.0.1'}).status_code,403)


if __name__ == '__main__':
    unittest.main()
