import sys,json,re,unittest,urllib.request,urllib.error,subprocess,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from paste_test import create_server
class PageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  global BASE
  urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
  cls.server=create_server();BASE='http://127.0.0.1:'+str(cls.server.server_address[1]);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  page=urllib.request.urlopen(BASE+'/').read().decode('utf-8');cls.token=re.search(r"const token='([^']+)'",page)[1]
 @classmethod
 def tearDownClass(cls):
  cls.server.shutdown();cls.server.server_close();cls.thread.join()
 def request(self,data,headers=None,path='/api/analyze'):
  raw=json.dumps(data).encode();req=urllib.request.Request(BASE+path,data=raw,headers={'Content-Type':'application/json','X-Session-Token':self.token,**(headers or {})})
  try:
   with urllib.request.urlopen(req) as r:return r.status,json.loads(r.read())
  except urllib.error.HTTPError as e:return e.code,json.loads(e.read())
 def test_origin_and_token(self):
  self.assertEqual(self.request({'text':'x'},{'Origin':'https://example.com'})[0],403)
  self.assertEqual(self.request({'text':'x'},{'X-Session-Token':'wrong'})[0],403)
 def test_no_arbitrary_files(self):
  try:urllib.request.urlopen(BASE+'/../sample_reviews.json')
  except urllib.error.HTTPError as e:self.assertEqual(e.code,404)
  else:self.fail('Arbitrary file exposed')
 def test_input_validation(self):
  for v in [None,123,'','[A01]\n5','x'*300001]:self.assertEqual(self.request({'text':v})[0],400)
  self.assertEqual(self.request([])[0],400)
 def test_bad_host(self):
  req=urllib.request.Request(BASE+'/',headers={'Host':'external.example'})
  with self.assertRaises(urllib.error.HTTPError) as cm:urllib.request.urlopen(req)
  self.assertEqual(cm.exception.code,403)
 def test_cli_parity(self):
  src=ROOT.parent/'nano-now/qwen-app-fast-06.txt'
  code,response=self.request({'text':src.read_text(encoding='utf-8-sig')});self.assertEqual(code,200)
  result=subprocess.run([sys.executable,'-B',str(ROOT/'predict.py'),str(src)],capture_output=True,env={**__import__('os').environ,'PYTHONIOENCODING':'utf-8'},check=True)
  cli=json.loads(result.stdout.decode('utf-8'));self.assertEqual(response,cli)
if __name__=='__main__':unittest.main()

