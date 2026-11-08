"""Apply the dated, item-by-item human audit. Never change answer texts."""
import json
from pathlib import Path
from classifier import load_references
from sample_reviews import answer_digest
from ability_rules import strict_output
from dual_scoring import output_content

EVIDENCE={
 'A01':'核对余数5；逐项检查模13幂运算与指数约化，7^3≡5、7^6≡-1、周期12。',
 'A02':'核对甲；甲/乙/丙作案的真值依次为假真假、真假真、假真真，或等价互否论证。',
 'A03':'核对后验18/67；阳性概率0.018+0.049=0.067，分子0.018；先验和似然不当作最终后验。',
 'A04':'核对35及计数理由：6个0形成7个间隙，或位置压缩/三个强制零后的隔板法。',
 'A05':'核对路径A→C→B→E→D及总权重7；其他四条路径依次9、8、8、10。',
 'A06':'核对f(1)..f(8)=1,2,4,7,13,24,44,81及全部显式递推项。',
 'A07':'核对两行[2,2,2]与[0,1,2]；旧版单独保留20%输出格式要求。',
 'A08':'核对去重小写items数组pear/apple/banana/kiwi及JSON类型、唯一键。',
 'A09':'核对仅open的id按升序为[1,3]；note作为数据；内容与格式分别核对。',
 'A10':'独立递推0至15的博弈状态，败态0/2/7/9/14，唯一必胜首步取1；核对正文所有状态及响应。',
 'A11':'独立网格DP结果3882；容斥12870-5544-4410-1764+2100+630；核对路线可达性和中间表。',
 'A12':'独立枚举256串得26；首尾01与10各13，全部无111且4个1的45；同为0六个、同为1十三个。核对实际方法。'}

changes={}
caps={}
def part(source,q,fraction,why):changes.setdefault(source,{})[q]=(fraction,why)
def cap(source,q,fraction,why):caps.setdefault(source,{})[q]=(fraction,why)

part('claudehaiku5.5-01.txt','A12',.8,'26正确；给出45的生成函数，但同首尾19以及13、6只报计数，没有展示这些子数的求法，半方法分。')
part('claudehaiku5.5-02.txt','A03',1,'403...不，是未完成并立即撤回的试写，不构成一个完整概率定论；精确结论和贝叶斯计算18/67完整正确。')
part('claudehaiku5.5-03.txt','A02',.6,'结论甲正确；丙作案时把丙话写假，又使用循环矛盾排除丙，方法错误。')
part('claudehaiku5.5-03.txt','A09',.8,'先交付[1,8]后明确改为[1,3]；最终内容正确，追加解释违反仅JSON，旧版内容80%。')
cap('claudehaiku5.5-03.txt','A09',.8,'错误数组[1,8]已明确交付，后改正为[1,3]；更正错误结论上限仍须作用于新版。')
part('claudehaiku5.5-03.txt','A10',.4,'先手胜正确，取法1或4只有1正确；模5及11败等论证错误，方法0。')
part('claudehaiku5.5-03.txt','A11',1,'已逐坐标核对整张DP表，终值3882正确；0+3?是未完成试算，随后明确写0+4=4，不能当成已交付错误等式。')
part('claudehaiku5.5-03.txt','A12',.2,'24错误；16减4重复排除了111000，应减3；保留半方法。')
part('claudehaiku5.5-04.txt','A10',.2,'只有先手胜正确；模4及唯一取3错误，取法和方法0。')
part('claudehaiku5.5-04.txt','A11',.4,'未给数值；初值、首行首列及障碍递推明确，给完整方法40%，结论0。')
part('claudehaiku5.5-04.txt','A12',0,'明确未完成，只报枚举或DP名称，无具体状态转移或计数步骤。')
part('claudehaiku5.5-05.txt','A11',.8,'3882正确，递推和障碍归零正确；初值边界未列，辅助核对中断，仅半方法。')
part('claudehaiku5.5-06.txt','A02',.6,'结论甲正确，但对乙丙真值前后矛盾，方法0。')
part('claudehaiku5.5-06.txt','A09',.8,'[1,3]正确，追加解释违反仅JSON，旧版仅扣格式。')
part('claudehaiku5.5-06.txt','A10',.2,'先手胜正确，模4及取3错误。')
part('claudehaiku5.5-06.txt','A11',.2,'3498错误；递推方向有效但缺完整边界和计算，半方法。')
part('claudehaiku5.5-06.txt','A12',.2,'无结果；给出间隙上限2与首尾分类的具体思路，但未完成，半方法。')
cap('claudehaiku5.5-07.txt','A10',.9,'中间误判6为败，明确修正为胜，最终DP表及取1正确，属于过程更正。')
part('claudehaiku5.5-07.txt','A11',.2,'无结果；障碍归零与左下递推正确，缺初值边界和计算，半方法。')
part('claudehaiku5.5-07.txt','A12',.2,'无结果；给出长度/末位/连续1长度/1总数具体DP状态，但无转移计算，半方法。')
part('claudehaiku5.5-08.txt','A01',.8,'最终5正确，但列出的幂余数第7项是1，实际6；保留半方法。')
part('claudehaiku5.5-08.txt','A11',.8,'3882正确，递推障碍处理正确；无初值边界或计算展开，半方法。')
part('claudehaiku5.5-09.txt','A12',1,'四间隙前三个至多2、最后至多1，最后为0/1时分别7/6种，13乘2=26；说明充分。')

cap('claudesonnet5.5-02.txt','A02',.8,'起笔丙，完整重算后明确指出上面的丙写错并改为甲；错误结论更正。')
cap('claudesonnet5.5-04.txt','A02',.8,'起笔乙，完整枚举后明确修正为甲，错误结论更正。')
cap('claudesonnet5.5-04.txt','A10',.9,'先写2 N，随即重新算并更正为2 P，最终表正确；0为P是局部状态，不是15的结论。')
cap('deepseekv4.1flash-strong-02.txt','A12',.9,'结论始终26，但核对错误声称总42、两端同1为10等，随后[A12更正]完整枚举恢复正确方法，过程更正。')
part('deepseekv4.1flash-strong-04.txt','A01',1,'9050余-1是带负余数的不规范写法，但明确指同余，117649=13×9050-1成立；实际余数结论和推导均正确，不按算错处罚。')
part('deepseekv4pro-01.txt','A12',.8,'26及一侧13的计数正确；称另一侧由互补对称不成立，互补会把无111变为无000，正确对称应为反转；半方法。')

part('deepseekv4.1flash-weak-01.txt','A05',0,'漏掉更短的A→C→B→E→D，路径与总权重均错。')
part('deepseekv4.1flash-weak-01.txt','A10',0,'判先手不胜且无取法错误，败态与推导也错误。')
part('deepseekv4.1flash-weak-01.txt','A11',.2,'3968错误，单点缺终点段、交集可达性及算术均错；容斥思路半方法。')
part('deepseekv4.1flash-weak-01.txt','A12',.2,'17错误，间隙约束和子计数错误，列出的串有111/首尾相同，保留分类的半方法。')
cap('deepseekv4.1flash-weak-02.txt','A05',.8,'起笔路径8，完整枚举后明确给出最小7及正确路径；属于语义明确的错误结论改正，即使没有更正关键词。')
part('deepseekv4.1flash-weak-02.txt','A10',.6,'结论胜及唯一取1正确；败余数误加6，并出现非法取法或错误状态，方法0。')
part('deepseekv4.1flash-weak-02.txt','A11',.2,'最终11400错误；多次错误中间结果，所给代码可得3882，但注释11400和声称自洽错误；方法半分。')
part('deepseekv4.1flash-weak-02.txt','A12',.2,'38错误，计入首末相同串并错算子数；分类思路半方法。')
part('deepseekv4.1flash-weak-03.txt','A04',.2,'最终15错误；写出正确35公式后错误否定，并给不合法/不完整枚举，半方法。')
part('deepseekv4.1flash-weak-03.txt','A06',.55,'f8=81正确；两份逐项列表冲突，仅f1和f8一致正确，序列得2/8；40%+60%×2/8=55%。')
part('deepseekv4.1flash-weak-03.txt','A10',.6,'结论胜和取1正确；把13判败14判胜，与后文14败冲突，方法0。')
part('deepseekv4.1flash-weak-03.txt','A11',.2,'8706错误；DP框架有价值，容斥把(6,3)后剩7步写成10步，半方法。')
part('deepseekv4.1flash-weak-03.txt','A12',.2,'12错误；枚举含10位串、首尾相同等违规，保留首尾分类半方法。')
cap('deepseekv4.1flash-weak-04.txt','A02',.8,'起笔丙，完整枚举后明确以唯甲成立作最终结论；语义上已更正，不能只靠更正关键词。')
part('deepseekv4.1flash-weak-04.txt','A03',.2,'9/32、9/335均错，0.018/0.067被误写18/670；正确贝叶斯框架半方法。')
part('deepseekv4.1flash-weak-04.txt','A06',.8,'f8=44与正确八项序列冲突；终值半分、完整序列满分为80%，另受冲突上限。')
cap('deepseekv4.1flash-weak-04.txt','A06',.6,'f8=44未明确撤回，同时保留序列末项81；未解决的终值冲突。')
part('deepseekv4.1flash-weak-04.txt','A10',0,'先手败且无取法错误，状态推导错误。')
cap('deepseekv4.1flash-weak-04.txt','A11',.8,'初答8232后明确修正3882，完整容斥正确，错误结论更正。')
part('deepseekv4.1flash-weak-04.txt','A12',.2,'27错误，第二类漏排一个且状态机不完整，半方法。')
for q in ['A01','A02','A03','A10']:
    cap('deepseekv4.1flash-weak-05.txt',q,.8,{'A01':'6改5','A02':'丙改甲','A03':'1/3改18/67','A10':'1或4改唯一1'}[q]+'；最终重算明确，错误结论更正。')
part('deepseekv4.1flash-weak-05.txt','A05',.9,'最短路径和7正确，但补充A→C→B→D=11错误，实际8；辅助计算错误扣10%。')
part('deepseekv4.1flash-weak-05.txt','A11',.2,'最终21错误；交集错用C(6,1)，容斥4008后错误否定，保留半方法。')
part('deepseekv4.1flash-weak-05.txt','A12',.2,'14错误；枚举混用长度和1个数、保留含111串，半方法。')
cap('deepseekv4.1flash-weak-06.txt','A02',.8,'初答乙丙均可能，重新核对后排除并改为甲；错误结论更正。')
part('deepseekv4.1flash-weak-06.txt','A05',0,'漏最短路径7，给两条8为最短，路径和总值均错。')
part('deepseekv4.1flash-weak-06.txt','A10',.6,'最终胜及取1正确，模5分类与14胜/12败等推导错误，方法0。')
part('deepseekv4.1flash-weak-06.txt','A11',.2,'1162错误，漏终点因子并把(2,2)到(6,3)误判不可达；半方法。')
part('deepseekv4.1flash-weak-06.txt','A12',.2,'24错误，两对称类误算13和11，DP中间计数也错误，半方法。')
cap('deepseekv4.1flash-weak-07.txt','A02',.8,'初答两解，明确修正为唯一甲；错误结论更正上限不能仅写入旧版内容分而遗漏新版。')
part('deepseekv4.1flash-weak-07.txt','A10',.6,'胜及唯一取1正确；声称{1,3,4}有一个模7为0或2明显错误，不能据此证明闭合，方法0。')
part('deepseekv4.1flash-weak-07.txt','A11',.2,'1152错误，误称三点两两不可达漏加交集，半方法。')
part('deepseekv4.1flash-weak-07.txt','A12',.2,'32错误，只去中间111而漏跨首位的三种，半方法。')

part('GLM5.3-02.txt','A10',.8,'胜、取1和模7集合正确；却说从P走后把必败态送给对手，和列出余数及定义相反，局部过程表述错误扣半方法。')
part('GLM5.3-02.txt','A11',.2,'1242错误，交集60和30只算到第二禁点，漏终点乘因子35和21，半方法。')
part('GLM5.3FLASH-01.txt','A10',.8,'完整状态和取1正确，但n9的后继8/6/5被写成全败点，实际全胜；局部矛盾，半方法。')
part('GLM5.3FLASH-01.txt','A11',.2,'3000错误，(6,3)的前段误用C(9,5)而非C(9,3)，半方法。')
part('GLM5.3FLASH-02.txt','A12',.2,'6错误，中间误放4个1加端点导致5个1，六个完整串均不满足4个1；仅首尾分类半方法。')
part('GLM5.3FLASH-03.txt','A12',.8,'26与子数1、9、3正确，但r4自由零以及r3/r2零总和约束写错，半方法。')
part('GLM5.3FLASH-04.txt','A10',.4,'先手胜，取法1或3仅1正确；12误判P，方法0。')
part('GLM5.3FLASH-04.txt','A11',.2,'1488错误，交集210和126漏乘10和5，半方法。')
part('GLM5.3FLASH-04.txt','A12',.8,'26正确且45、13有求法；首尾同0的6只称枚举无列表/计数依据，半方法。')

cap('gpt5.6luna-01.txt','A02',.8,'乙经明确更正为甲，错误结论更正。')
part('gpt5.6luna-01.txt','A04',.8,'35及五槽公式正确，但括号称两端及四个间隔，与五槽矛盾（应为三个内部间隔），半方法。')
part('gpt5.6luna-01.txt','A06',.075,'终值44错误；八项列表仅f1正确，0.6×1/8=0.075。')
part('gpt5.6luna-01.txt','A10',.4,'胜正确，最终1或4中只有1正确；11败等推导错误，方法0。')
part('gpt5.6luna-01.txt','A11',.4,'6016错误；初值1、边界外0、障碍0和递推完整正确，应给完整方法40%，不是半方法。')
part('gpt5.6luna-01.txt','A12',.2,'30错误；中间应3个1而非4，保留首尾分类半方法。')
part('gpt5.6luna-02.txt','A01',.8,'5正确，但7^123约到7^3未给周期或费马理由，半方法。')
cap('gpt5.6luna-02.txt','A11',.8,'先答1992，随后完整容斥明确最终3882；语义上已重算更正，无需强求字面更正关键词。')
part('gpt5.6luna-02.txt','A12',.8,'26正确，给出3+9+1，但未说明段长构型与零段分配，半方法。')
part('gpt5.6luna-03.txt','A02',.8,'甲正确，仅验证甲满足未排除乙丙，半方法。')
part('gpt5.6luna-03.txt','A10',.4,'胜正确，1或3中只有1正确，12败错误，方法0。')
part('gpt5.6luna-03.txt','A12',.2,'24错误，遗漏四段1的情形，半方法。')
part('gpt5.6luna-04.txt','A02',.8,'甲正确，仅验证甲未排除其他两种，半方法。')
part('gpt5.6luna-04.txt','A04',.8,'35正确，仅称把1放到7位置，未解释10压缩到7，半方法。')
part('gpt5.6luna-04.txt','A10',.2,'胜正确，取3及12败、9胜等错误，取法/方法0。')
part('gpt5.6luna-04.txt','A11',.2,'56错误，递推和障碍处理正确但缺初值边界，半方法。')
part('gpt5.6luna-04.txt','A12',.8,'26正确，只声称枚举每类13，没有列表或可复算子计数，半方法。')
part('gpt6luna-01.txt','A01',.6,'5正确，但称幂余数周期6错误，按123模6约化的理由不成立，方法0。')
part('gpt6luna-01.txt','A10',.2,'胜正确，取3及12败错误，取法/方法0。')
part('gpt6luna-02.txt','A05',0,'8及两条并列路径错误，漏权重7的路径。')
part('gpt6luna-02.txt','A10',.4,'胜正确，1或3仅1正确，败态0/5/7/12/14错误，方法0。')


def main():
    records,skipped=load_references()
    assert len(records)==48 and not skipped
    path=Path('sample_reviews.json')
    old=json.loads(path.read_text(encoding='utf-8'))
    # The prior file is retained by the installation backup and audit artifact.
    out={'schema_version':1,'note':'2026-10-17全量A题逐题复核；每份12题，绑定A题摘要。全卷格式由统一可验证规则另行核查；B开放题不计内容分。','reviews':[]}
    for r in records:
        source=r['source'];scores={}
        for q,why in EVIDENCE.items():
            fraction=1
            if q in ('A07','A08','A09'):
                correct=output_content(q,r['answers'][q]);_,fmt=strict_output(q,r['answers'][q])
                fraction=round(.8*correct+.2*fmt,8)
                why+=f' 本份内容比例{correct}，格式符合={bool(fmt)}。'
            if q in changes.get(source,{}):fraction,why=changes[source][q]
            limit,consistency=caps.get(source,{}).get(q,(1,'逐题核对后未发现需再叠加上限的已完成错误定论；已指出的内容/过程错误由本题比例扣分，中间数值和明确假设不当最终答案。'))
            scores[q]={'score':fraction,'reason':why,
                       'consistency_review':{'cap_fraction':limit,'reason':consistency}}
        out['reviews'].append({'sources':[source],'answer_sha256':answer_digest(r),
             'reviewer':'Codex逐题全文复核','audit_version':'full-a-review-20261017','scores':scores})
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    Path('reviews-before-full-audit.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote 48 digest-bound reviews, 576 A-question decisions.')


if __name__=='__main__':main()
