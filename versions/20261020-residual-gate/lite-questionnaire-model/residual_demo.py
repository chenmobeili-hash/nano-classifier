"""Loopback paste/file demo; optional user-confirmed AI review, never trains."""
import argparse
import hmac
import json
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import feature_profiles as F
import neural_core as M
import nlp_core as N
import residual_gate as R
from classifier import canonical_answers, fingerprint, load_references, read_answer
from ability import score_ability
from ability_ranking import build_model_ranking
from result_scoring import score_results
from score_output import score_summary
from ai_review import ReviewManager

HERE = Path(__file__).resolve().parent
MAX_BODY, MAX_TEXT = 1_500_000, 300_000


class ComparisonService:
    def __init__(self):
        self.lock = threading.RLock()
        self.reviewer = ReviewManager(self, HERE)
        self.meta, self.models = R.load_bundle(HERE/'residual-gate.npz')
        release_path = HERE/'CURRENT_RELEASE.json'
        self.release = json.loads(release_path.read_text(encoding='utf-8')) if release_path.exists() else {'label': '残差门控融合实验版'}
        self.refs, skipped = load_references()
        if skipped or N.reference_signature(self.refs) != self.meta['reference_signature']:
            raise ValueError('参考库与实验权重不一致，请先重新训练实验版。')
        evaluation = json.loads((HERE/'evaluation-residual-gate.json').read_text(encoding='utf-8'))
        if evaluation['reference_signature'] != self.meta['reference_signature']:
            raise ValueError('实验评估结果与权重不一致。')
        self.thresholds = {name: row['threshold'] for name, row in evaluation['runs']['sample']['summary'].items()}
        self.models_by_name = {}
        for index, ref in enumerate(self.refs):
            self.models_by_name.setdefault(ref['model'], []).append(index)
        self.ranking = build_model_ranking(self.refs)

    def status(self):
        return {'samples': len(self.refs), 'families': len(self.meta['classes']),
                'policy_enabled': bool(self.meta['selection']['enabled']), 'release': self.release}

    def verdict(self, scores, threshold):
        order = np.argsort(-scores)
        margin = float(scores[order[0]]-scores[order[1]])
        top = self.meta['classes'][int(order[0])]
        return {'top': top, 'candidate': top if margin >= threshold else None,
                'margin': margin, 'threshold': threshold,
                'scores': dict(zip(self.meta['classes'], scores.tolist()))}

    def analyze(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('请先粘贴完整答卷。')
        if len(text) > MAX_TEXT:
            raise ValueError('答卷过长，请控制在30万字以内。')
        record = {'raw_response': text}
        record['answers'] = canonical_answers(record)
        with self.lock:
            pm = M.predict_scores(record, self.models['mlp'])
            query_vector = N.apply_idf(N.count_vector(record)[None], self.models['nlp']['idf'])
            pn = N.predict_scores(query_vector, self.models['nlp'])[0]
            logits = R.fused_logits(pm[None], pn[None])
            raw, delta, _ = R.forward(R.local_inputs(R.buckets([record]), self.models['nlp']), logits, self.models['gate'])
            baseline = N.softmax(logits)[0]
            enabled = bool(self.meta['selection']['enabled'])
            distances = F.distances(record, self.refs, 'balanced')[0]
            nearest = np.argsort(distances, kind='stable')[:3]
            candidates = sorted(
                ({'model': model, 'family': self.refs[index[0]]['family'], 'samples': len(index),
                  'distance': float(distances[index].mean()),
                  'nearest_sample': self.refs[index[int(np.argmin(distances[index]))]]['source']}
                 for model, index in self.models_by_name.items()),
                key=lambda row: row['distance'])
            ability = score_ability(record, ranking=self.ranking)
            results = score_results(record)
            digest = fingerprint(record)
            return {'baseline': self.verdict(baseline, self.thresholds['baseline']),
                    'gated': self.verdict(raw[0], self.thresholds['gated']),
                    'policy': self.verdict(raw[0] if enabled else baseline, self.thresholds['selected']),
                    'policy_enabled': enabled,
                    'same_top': int(baseline.argmax()) == int(raw[0].argmax()),
                    'max_logit_correction': float(np.abs(delta).max()),
                    'protected': bool(R.support(logits)[0, 0] == 0),
                    'nearest_samples': [dict(source=self.refs[i]['source'], model=self.refs[i]['model'],
                                             family=self.refs[i]['family'], distance=float(distances[i])) for i in nearest],
                    'nearest_metric': '135维特征空间（与分类器同一空间）的标准化距离，越小越相近；不参与家族分数计算。',
                    'model_candidates': candidates[:5],
                    'model_candidates_metric': '按型号组取平均距离升序；同一型号多份答卷可降低单份噪声。只表示与库内已知型号的接近程度，不能识别库外新型号。',
                    'reference_matches': [r['source'] for r in self.refs if r['fingerprint'] == digest],
                    'review_provenance': self.reviewer.provenance(record),
                    'scores': score_summary(ability, results), 'ability': ability, 'result_only': results,
                    'note': '家族分数未经概率校准，不能当作身份正确概率；最近样本仅作诊断，答卷评分独立计算。'}


def read_input(path):
    record = read_answer(path)
    return record.get('raw_response') or '\n\n'.join('['+q+']\n'+a for q, a in record['answers'].items())


def create_server(service, initial_text='', port=0):
    token = secrets.token_urlsafe(32)
    encoded = json.dumps(initial_text, ensure_ascii=False).replace('<', '\\u003c').replace('&', '\\u0026')
    html = (HERE/'residual_demo.html').read_text(encoding='utf-8')
    html = html.replace('__SESSION_TOKEN__', token).replace('__INITIAL_TEXT_JSON__', encoded).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def local(self):
            port = self.server.server_address[1]
            return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

        def reply(self, status, body, kind='application/json; charset=utf-8'):
            data = json.dumps(body, ensure_ascii=False).encode('utf-8') if isinstance(body, dict) else body
            self.send_response(status)
            for key, value in {'Content-Type': kind, 'Content-Length': str(len(data)),
                               'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                               'Referrer-Policy': 'no-referrer',
                               'Content-Security-Policy': f"default-src 'self'; script-src 'nonce-{token}'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"}.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.local():
                return self.reply(403, {'error': '仅允许本机访问。'})
            if self.path == '/':
                return self.reply(200, html, 'text/html; charset=utf-8')
            if self.path == '/api/status':
                return self.reply(200, service.status())
            return self.reply(404, {'error': '页面不存在。'})

        def do_POST(self):
            origin = self.headers.get('Origin')
            if not self.local() or (origin is not None and origin != 'http://'+self.headers.get('Host', '')) or not hmac.compare_digest(self.headers.get('X-Session-Token', ''), token):
                return self.reply(403, {'error': '请从本机测试页面提交。'})
            routes = {'/api/analyze', '/api/review/config', '/api/review/propose', '/api/review/manual', '/api/review/preview', '/api/review/apply'}
            if self.path not in routes:
                return self.reply(404, {'error': '页面不存在。'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, {'error': '提交格式不正确。'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY:
                    return self.reply(413, {'error': '答卷为空或过长。'})
                payload = json.loads(self.rfile.read(size).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('答卷格式不正确。')
                if self.path == '/api/review/config':
                    result = service.reviewer.configure(payload)
                elif self.path == '/api/review/propose':
                    result = service.reviewer.propose(payload.get('text'))
                elif self.path == '/api/review/manual':
                    result = service.reviewer.manual(payload.get('text'))
                elif self.path == '/api/review/preview':
                    result = service.reviewer.preview(payload)
                elif self.path == '/api/review/apply':
                    result = service.reviewer.apply(payload)
                else:
                    result = service.analyze(payload.get('text'))
                return self.reply(200, result)
            except (ValueError, KeyError, UnicodeError) as exc:
                return self.reply(400, {'error': str(exc)})
            except (OSError, RuntimeError) as exc:
                return self.reply(500, {'error': '本机文件操作失败，未完成的复核会恢复备份；请检查目录权限并重试。'})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='原融合与局部残差门控的本机对照测试。')
    parser.add_argument('answer', nargs='?', help='可选：预先载入答卷文件')
    parser.add_argument('--json', action='store_true', help='只输出文件的对照JSON')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    if not args.json:
        print('正在载入实验模型和评分资料…', flush=True)
    service = ComparisonService()
    text = read_input(args.answer) if args.answer else ''
    if args.json:
        if not args.answer:
            parser.error('--json 需要指定答卷文件')
        print(json.dumps(service.analyze(text), ensure_ascii=False, indent=2))
        return
    server = create_server(service, text, args.port)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print('残差门控实验页：'+url, flush=True)
    print('粘贴答卷或选择文件后开始测试。关闭此窗口即可停止。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
