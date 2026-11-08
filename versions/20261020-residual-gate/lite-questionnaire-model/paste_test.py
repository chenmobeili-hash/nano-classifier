"""Loopback-only paste page using the same classifier and scores as predict.py."""
import argparse
import hmac
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from classifier import canonical_answers, identify, inventory, load_references, DEFAULT_REFERENCES
from neural_core import load_network, DEFAULT_MODEL
from nlp_classifier import identify_nlp
import nlp_core
import fusion as fuse_mod
from fusion import identify_fused
from ability import score_ability
from ability_ranking import build_model_ranking
from result_scoring import score_results
from score_output import score_summary

HERE = Path(__file__).resolve().parent
MAX_BODY = 1_500_000
MAX_TEXT = 300_000

class AnalysisService:
    def __init__(self, engine='nlp', kind=None, min_margin=None, fuse_rule='geo'):
        self.lock = threading.Lock()
        self.stamp = None
        self.engine = engine
        self.kind = kind
        self.min_margin = min_margin
        self.fuse_rule = fuse_rule

    def _model_path(self):
        return DEFAULT_MODEL if self.engine in ('neural', 'distance') else nlp_core.DEFAULT_MODEL

    def _refresh(self):
        paths = sorted(DEFAULT_REFERENCES.glob('*.txt')) + [self._model_path()] + [HERE/n for n in ('sample_reviews.json','result_reviews.json','sample_annotations.json','known_external_references.json')]
        if self.engine == 'fused':
            paths = paths + [DEFAULT_MODEL]   # 融合引擎要同时盯两个权重
        stamp = tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in paths if p.exists())
        if stamp != self.stamp:
            refs, skipped = load_references()
            if skipped:
                raise ValueError('参考库存在无法读取的答卷，请先检查参考文件。')
            model = load_network() if self.engine in ('neural', 'distance') else nlp_core.load_model()
            ranking = build_model_ranking(refs)
            self.refs, self.model, self.ranking = refs, model, ranking
            self.mlp = load_network() if self.engine == 'fused' else None
            # 参考库文本向量与参考库同一生命周期，避免每次分析都重算（省约 1 秒）。
            self.nlp_vectors = (nlp_core.apply_idf(nlp_core.count_matrix(refs, model['meta'].get('include_a', False)), model['idf'])
                                if self.engine in ('nlp', 'fused') else None)
            self.stamp = stamp

    def status(self):
        with self.lock:
            self._refresh()
            meta = self.model['meta']
            if self.engine in ('neural', 'distance'):
                margin = self.min_margin if self.min_margin is not None else .1
            elif self.engine == 'fused':
                margin = self.min_margin if self.min_margin is not None else fuse_mod.CALIBRATED_MIN_MARGIN[self.fuse_rule]
            else:
                margin = self.min_margin if self.min_margin is not None else nlp_core.CALIBRATED_MIN_MARGIN
            return dict(version=('融合版' if self.engine=='fused' else 'NLP版'),engine=self.engine,model_kind=meta.get('kind') if self.engine!='neural' else 'mlp',
                        fuse_rule=self.fuse_rule if self.engine=='fused' else None,
                        min_margin=margin,reference_count=len(self.refs),families=len(meta['classes']))

    def analyze(self, text):
        if not isinstance(text,str) or not text.strip():
            raise ValueError('请先粘贴完整答卷。')
        if len(text)>MAX_TEXT:
            raise ValueError('答卷过长，请控制在30万字以内。')
        record = dict(raw_response=text)
        record['answers'] = canonical_answers(record)
        with self.lock:
            self._refresh()
            ability = score_ability(record,ranking=self.ranking)
            results = score_results(record)
            if self.engine=='fused':
                family = identify_fused(record,self.refs,rule=self.fuse_rule,min_margin=self.min_margin,
                                        mlp_model=self.mlp,nlp_model=self.model,reference_vectors=self.nlp_vectors)
            elif self.engine=='nlp':
                family = identify_nlp(record,self.refs,model=self.model,kind=self.kind,min_margin=self.min_margin,
                                      reference_vectors=self.nlp_vectors)
            else:
                family = identify(record,self.refs,model=self.model)
            return dict(scores=score_summary(ability,results),family=family,ability=ability,result_only=results,reference_inventory=inventory(self.refs,[]))

def create_server(port=0,service=None):
    service = service or AnalysisService()
    token = secrets.token_urlsafe(32)
    html = (HERE/'paste_test.html').read_text(encoding='utf-8').replace('__SESSION_TOKEN__',token).encode('utf-8')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass # Never log answer bodies.
        def _reply(self,status,body,kind='application/json; charset=utf-8'):
            data = json.dumps(body,ensure_ascii=False).encode('utf-8') if isinstance(body,dict) else body
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',f"default-src 'self'; script-src 'nonce-{token}'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass
        def _local(self):
            port=self.server.server_address[1]
            return self.headers.get('Host') in (f'127.0.0.1:{port}',f'localhost:{port}')
        def do_GET(self):
            if not self._local():return self._reply(403,dict(error='仅允许本机访问。'))
            if self.path=='/':return self._reply(200,html,'text/html; charset=utf-8')
            if self.path=='/api/status':
                try:return self._reply(200,service.status())
                except (ValueError,OSError,RuntimeError) as exc:return self._reply(500,dict(error=str(exc)))
            return self._reply(404,dict(error='页面不存在。'))
        def do_POST(self):
            origin=self.headers.get('Origin')
            if not self._local() or (origin is not None and origin!='http://'+self.headers.get('Host','')) or not hmac.compare_digest(self.headers.get('X-Session-Token',''),token):
                return self._reply(403,dict(error='请求来自其他页面，请重新打开本机测试页。'))
            if self.path!='/api/analyze':return self._reply(404,dict(error='页面不存在。'))
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self._reply(415,dict(error='请提交JSON格式的答卷。'))
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=MAX_BODY:return self._reply(413,dict(error='答卷为空或超过大小限制。'))
                data=json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(data,dict):raise ValueError('答卷格式不正确。')
                result=service.analyze(data.get('text'))
                self._reply(200,result)
            except (ValueError,KeyError,UnicodeError) as exc:self._reply(400,dict(error=str(exc)))
            except (OSError,RuntimeError) as exc:self._reply(500,dict(error=str(exc)))
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server

def main():
    parser=argparse.ArgumentParser(description='粘贴答卷，在本机识别并评分。')
    parser.add_argument('--port',type=int,default=0)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--engine',choices=('nlp','neural','fused'),default='nlp',help='nlp=文本n-gram线性模型（本版默认）；fused=与神经网络晚期融合；neural=原135维方案')
    parser.add_argument('--fuse',choices=fuse_mod.RULES,default='geo',help='fused 引擎的融合规则：geo 几何平均 / mean 算术平均')
    parser.add_argument('--nlp-kind',choices=nlp_core.KINDS,help='NLP 分类器；默认用权重 meta 里的 kind')
    parser.add_argument('--min-margin',type=float,default=None,help='候选间隔门槛；默认按引擎取标定值（nlp 0.0106 / fused geo 0.0746 / neural 0.1）')
    args=parser.parse_args()
    server=create_server(args.port,AnalysisService(engine=args.engine,kind=args.nlp_kind,min_margin=args.min_margin,fuse_rule=args.fuse))
    url=f'http://127.0.0.1:{server.server_address[1]}/'
    print('本机测试页：'+url+'  （引擎 '+args.engine+'）',flush=True)
    print('答卷仅在本机内存处理，不自动入库。使用结束后关闭此窗口即可停止服务。',flush=True)
    if not args.no_browser:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
