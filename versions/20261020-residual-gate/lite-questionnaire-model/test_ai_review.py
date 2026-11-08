"""Regression checks for local review isolation, schema and transactional saves.
All writes use TemporaryDirectory; API requests use a local mock server only.
"""
import copy
import hashlib
import http.client
import json
import re
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
import sample_reviews
import result_scoring
from ability import score_points
from classifier import load_references
from sample_reviews import answer_digest
from ai_review import ReviewManager, FILES, VERSION, validate_rows, normalize_endpoint, call_api
import residual_demo

HERE = Path(__file__).resolve().parent
ORIGINAL_RESULT = result_scoring.score_results

class Stub:
    def __init__(self, records):
        self.refs = records
        self.lock = threading.RLock()
        self.ranking = {'prior':'sentinel'}
    def analyze(self, text):
        r={'raw_response':text}
        return {'score':score_points(r)['score'],'result':result_scoring.score_results(r)['score']}
    def status(self):
        return {'samples':len(self.refs),'families':5,'policy_enabled':False,'release':{}}

class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()
        cls.record = next(r for r in cls.refs if r['source']=='gpt6.1sol-03.txt')
        cls.text = cls.record['raw_response']

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE.parent)
        Path(self.temp.name).resolve().relative_to(HERE.parent.resolve())
        self.addCleanup(self.temp.cleanup)
        self.here = Path(self.temp.name)
        for name in FILES:
            shutil.copyfile(HERE/name,self.here/name)
        self.patch_reviews=patch.object(sample_reviews,'REVIEWS_PATH',self.here/'sample_reviews.json')
        self.patch_reviews.start();self.addCleanup(self.patch_reviews.stop)
        self.patch_results=patch.object(result_scoring,'score_results',lambda r,review_path=None:ORIGINAL_RESULT(r,self.here/'result_reviews.json'))
        self.patch_results.start();self.addCleanup(self.patch_results.stop)
        self.service = Stub(self.refs)
        self.rows = []
        content=sample_reviews.review_for(self.record)
        finals={r['question']:r for r in result_scoring.score_results(self.record)['questions']}
        for q,answer in self.record['answers'].items():
            if not q.startswith('A'):continue
            self.rows.append({'question':q,'score':content[q]['score'],
                'final_correct':finals[q]['correct'],
                'cap_fraction':content[q].get('consistency_review',{}).get('cap_fraction',1),
                'reason':'逐题复核依据：'+q,'evidence':answer[:200]})
        self.calls=[]
        def transport(config,messages):
            self.calls.append((copy.deepcopy(config),copy.deepcopy(messages)))
            return {'questions':copy.deepcopy(self.rows)}
        self.manager=ReviewManager(self.service,self.here,transport)
        self.service.reviewer=self.manager

    def configure(self):
        return self.manager.configure({'action':'set','endpoint':'https://example.invalid/v1','model':'test-review','key':'test-memory-key'})
    def draft(self):
        self.configure()
        return self.manager.propose(self.text)
    def payload(self, draft):
        return {'draft_id':draft['draft_id'],'text':self.text,'review':{'questions':copy.deepcopy(draft['questions'])},'confirmed':True,'replace_existing':True}
    def hashes(self):
        return {n:hashlib.sha256((self.here/n).read_bytes()).hexdigest() for n in FILES}

    def test_memory_only_and_no_key_echo(self):
        state=self.configure()
        self.assertNotIn('test-memory-key',json.dumps(state))
        self.assertTrue(state['has_key'])
        self.manager.configure({'action':'set','endpoint':'https://example.invalid/v1','model':'test-review','key':''})
        self.assertEqual(self.manager.config['key'],'test-memory-key')
        with self.assertRaises(ValueError):
            self.manager.configure({'action':'set','endpoint':'https://other.invalid/v1','model':'test-review','key':''})
        self.assertFalse(ReviewManager(self.service,self.here).public_config()['configured'])
        self.manager.configure({'action':'clear'})
        self.assertIsNone(self.manager.config)

    def test_endpoint_validation(self):
        for value in ['file:///secret','http://remote.invalid/v1','https://name:key@host/v1','https://host/v1?key=abc','https://host/#key','https://host:bad/v1']:
            with self.subTest(value=value),self.assertRaises(ValueError):normalize_endpoint(value)
        self.assertEqual(normalize_endpoint('https://host/v1'),'https://host/v1/chat/completions')
        self.assertEqual(normalize_endpoint('http://127.0.0.1:1234/v1/chat/completions'),'http://127.0.0.1:1234/v1/chat/completions')
        self.assertEqual(normalize_endpoint('https://host'),'https://host/v1/chat/completions')

    def test_schema_coverage_quote_and_numeric_guards(self):
        validate_rows({'questions':self.rows},self.record['answers'])
        variants=[]
        r=copy.deepcopy(self.rows);r.pop();variants.append(r)
        r=copy.deepcopy(self.rows);r[-1]['question']='A01';variants.append(r)
        for key,value in [('score',True),('score',float('nan')),('score',1.5),('cap_fraction',.7),('final_correct','true'),('evidence','nonexistent quote'),('reason','')]:
            r=copy.deepcopy(self.rows);r[0][key]=value;variants.append(r)
        r=copy.deepcopy(self.rows);r[0]['final_correct']=False;variants.append(r)
        for rows in variants:
            with self.subTest(rows=rows[:1]),self.assertRaises(ValueError):validate_rows({'questions':rows},self.record['answers'])

    def test_propose_no_writes_and_only_a_sent(self):
        before=self.hashes();d=self.draft()
        self.assertEqual(before,self.hashes())
        self.assertTrue(d['existing_review'])
        self.assertIn('gpt6.1sol-03.txt',d['reference_sources'])
        self.assertEqual(len(d['questions']),12)
        messages=self.calls[-1][1]
        self.assertEqual(messages[0]['role'],'system')
        sent=json.loads(messages[1]['content'].split('\n',1)[1])
        self.assertEqual(set(sent),set('A%02d'%i for i in range(1,13)))
        self.assertNotIn('test-memory-key',json.dumps(d))
        self.assertEqual(d['preview']['combined']['provisional_score'],200)

    def test_manual_needs_no_key(self):
        before=self.hashes();d=self.manager.manual(self.text)
        self.assertEqual(d['provider']['model'],'manual')
        self.assertEqual(d['preview']['combined']['provisional_score'],200)
        self.assertEqual(before,self.hashes());self.assertEqual(self.calls,[])

    def test_changed_text_expired_and_config_changed_rejected(self):
        d=self.draft();p=self.payload(d);p['text']+='\n'
        with self.assertRaisesRegex(ValueError,'答卷已改变'):self.manager.preview(p)
        self.manager.drafts[d['draft_id']]['created']-=3601
        with self.assertRaisesRegex(ValueError,'过期'):self.manager.preview(self.payload(d))
        self.manager.configure({'action':'clear'})
        with self.assertRaisesRegex(ValueError,'过期'):self.manager.apply(self.payload(d))

    def test_confirmation_and_overwrite_required(self):
        d=self.draft();p=self.payload(d);p['confirmed']=False
        with self.assertRaisesRegex(ValueError,'勾选'):self.manager.apply(p)
        p['confirmed']=True;p['replace_existing']=False
        with self.assertRaisesRegex(ValueError,'替换'):self.manager.apply(p)

    def test_edited_draft_requires_new_preview(self):
        d=self.draft();p=self.payload(d);p['review']['questions'][3]['score']=.8
        with self.assertRaisesRegex(ValueError,'预览'):self.manager.apply(p)
        out=self.manager.preview(p)
        self.assertEqual(out['scores']['combined']['provisional_score'],197.72)
        saved=self.manager.apply(p)
        self.assertTrue(saved['saved']);self.assertEqual(saved['analysis']['score'],98.8)
        self.assertEqual(saved['analysis']['result'],100)
        row=next(r for r in json.loads((self.here/'sample_reviews.json').read_text(encoding='utf-8'))['reviews'] if r['answer_sha256']==d['answer_sha256'])
        self.assertEqual(row['audit_version'],VERSION)
        self.assertEqual(row['scores']['A04']['score'],.8)
        self.assertTrue(Path(saved['backup'],'accepted-review.json').is_file())
        self.assertEqual(len(self.service.refs),100)
        self.assertFalse((self.here/'.ai-review-transaction.json').exists())
        self.assertFalse((self.here/'.ai-review-write-lock').exists())
        self.assertNotIn('test-memory-key',''.join(f.read_text(encoding='utf-8') for f in self.here.rglob('*.json')))

    def test_external_review_never_enters_training(self):
        self.configure();text=self.text.replace(self.record['answers']['A01'],self.record['answers']['A01']+'\n额外注释。',1)
        d=self.manager.propose(text);self.assertFalse(d['existing_review']);self.assertEqual(d['reference_sources'],[])
        p=self.payload(d);p['text']=text;p['replace_existing']=False
        self.assertTrue(self.manager.apply(p)['saved']);self.assertEqual(len(self.service.refs),100)

    def test_stale_file_changes_rejected(self):
        d=self.draft();p=self.here/'sample_reviews.json';p.write_bytes(p.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'其他操作'):self.manager.apply(self.payload(d))

    def test_transaction_rollback(self):
        d=self.draft();p=self.payload(d);before=self.hashes();prior=self.service.ranking
        original=self.manager.atomic_write;failed=[False]
        def failing(path,data):
            if Path(path)==self.here/'model-ranking.json' and not failed[0]:
                failed[0]=True;raise OSError('injected failure')
            return original(path,data)
        with patch.object(self.manager,'atomic_write',side_effect=failing),self.assertRaises(OSError):self.manager.apply(p)
        self.assertEqual(before,self.hashes());self.assertIs(self.service.ranking,prior)
        self.assertFalse((self.here/'.ai-review-transaction.json').exists())
        self.assertFalse((self.here/'.ai-review-write-lock').exists())

    def test_crash_journal_recovery(self):
        before=self.hashes();backup=self.here/'ai_review_history/crash';backup.mkdir(parents=True)
        for n in FILES:shutil.copyfile(self.here/n,backup/n)
        (self.here/'.ai-review-transaction.json').write_text(json.dumps({'backup':'ai_review_history/crash','files':before}),encoding='utf-8')
        (self.here/'sample_reviews.json').write_text('interrupted write',encoding='utf-8')
        ReviewManager(self.service,self.here)
        self.assertEqual(before,self.hashes())

    def test_active_writer_not_removed(self):
        import os
        lock=self.here/'.ai-review-write-lock';lock.mkdir();(lock/'owner.json').write_text(json.dumps({'pid':os.getpid()}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'另一个服务'):ReviewManager(self.service,self.here)
        self.assertTrue(lock.exists())

    def test_local_api_transport_and_redirect_refused(self):
        rows=copy.deepcopy(self.rows);seen=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append((self.path,self.headers.get('Authorization'),payload))
                if self.path=='/redirect':self.send_response(307);self.send_header('Location','/must-not-follow');self.end_headers();return
                if self.path=='/error':self.send_response(401);self.end_headers();self.wfile.write(b'test-memory-key');return
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'choices':[{'message':{'content':json.dumps({'questions':rows})}}]}).encode())
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            url='http://127.0.0.1:'+str(server.server_address[1]);cfg={'endpoint':url+'/v1/chat/completions','model':'mock','key':'test-memory-key'}
            value=call_api(cfg,[{'role':'user','content':'data'}]);self.assertEqual(value['questions'],rows)
            self.assertEqual(seen[0][0],'/v1/chat/completions');self.assertEqual(seen[0][1],'Bearer test-memory-key')
            for path in ('/redirect','/error'):
                with self.assertRaises(ValueError) as error:call_api(dict(cfg,endpoint=url+path),[])
                self.assertNotIn('test-memory-key',str(error.exception))
            self.assertNotIn('/must-not-follow',[s[0] for s in seen])
        finally:server.shutdown();server.server_close();thread.join()

    def test_http_review_routes_keep_session_and_origin_guards(self):
        before=self.hashes();server=residual_demo.create_server(self.service);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();port=server.server_address[1]
        def request(method,path,body=None,token=None,origin=None,host=None):
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=10)
            headers={'Host':host or '127.0.0.1:'+str(port),'Content-Type':'application/json'}
            if token:headers['X-Session-Token']=token
            if origin:headers['Origin']=origin
            conn.request(method,path,body=json.dumps(body) if body is not None else None,headers=headers);r=conn.getresponse();data=r.read();status=r.status;conn.close();return status,data
        try:
            status,html=request('GET','/');self.assertEqual(status,200);token=re.search(rb"const token='([^']+)'",html)[1].decode()
            self.assertEqual(request('GET','/api/review/config')[0],404)
            self.assertEqual(request('POST','/api/review/config',{'action':'get'})[0],403)
            self.assertEqual(request('POST','/api/review/config',{'action':'get'},token,'https://outside.invalid')[0],403)
            self.assertEqual(request('POST','/api/review/config',{'action':'get'},token,host='outside.invalid')[0],403)
            self.assertEqual(request('POST','/api/review/config',{'action':'get'},token)[0],200)
            status,data=request('POST','/api/review/manual',{'text':self.text},token);self.assertEqual(status,200);draft=json.loads(data)
            self.assertEqual(request('POST','/api/review/preview',self.payload(draft),token)[0],200)
            self.assertEqual(before,self.hashes())
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()

