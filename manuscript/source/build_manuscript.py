from pathlib import Path
import os, json, re, shutil, sys
from copy import deepcopy
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
BASE=Path(__file__).resolve().parent
WORK=BASE
OUT=Path(os.environ.get('MANUSCRIPT_OUTPUT', str(BASE.parent)))
OUT.mkdir(parents=True,exist_ok=True)
E=json.loads((WORK/'evidence_legacy.json').read_text())
C=json.loads((WORK/'evidence_calibration.json').read_text())
REFS={v['key']:v for v in json.loads((WORK/'references_all.json').read_text())}
MAIN=(WORK/'main.md').read_text()
SUPP=(WORK/'supplement.md').read_text()
CITED=[]
for citation in re.findall(r'\[@([^\]]+)\]',MAIN):
    for key in citation.split(';'):
        key=key.strip().lstrip('@')
        if key not in CITED:
            assert key in REFS,key
            CITED.append(key)
NUM={key:i+1 for i,key in enumerate(CITED)}
def expand_citations(s):
    return re.sub(r'\[@([^\]]+)\]', lambda m: '['+', '.join(str(NUM[k.strip().lstrip('@')]) for k in m.group(1).split(';'))+']', s)

def set_cell_borders(cell):
    pr=cell._tc.get_or_add_tcPr()
    edges=OxmlElement('w:tcBorders')
    for edge in ('top','left','bottom','right','insideH','insideV'):
        x=OxmlElement('w:'+edge)
        # Journal-specific instructions override the generic document table style.
        for k,v in [('val','nil' if edge in ('left','right','insideV') else 'single'),('sz','4'),('color','D9D9D9')]:x.set(qn('w:'+k),v)
        edges.append(x)
    pr.append(edges)
    margins=OxmlElement('w:tcMar')
    for k,v in [('top',85),('bottom',85),('left',95),('right',95)]:
        x=OxmlElement('w:'+k);x.set(qn('w:w'),str(v));x.set(qn('w:type'),'dxa');margins.append(x)
    pr.append(margins)

def shade(cell, fill):
    x=OxmlElement('w:shd');x.set(qn('w:fill'),fill);cell._tc.get_or_add_tcPr().append(x)

def new_doc(supplement=False):
    d=Document()
    sec=d.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.78);sec.bottom_margin=Inches(.75)
    sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
    sec.header_distance=Inches(.3);sec.footer_distance=Inches(.3)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(11.5)
    normal.font.color.rgb=RGBColor(0,0,0)
    normal.paragraph_format.line_spacing=1.12
    normal.paragraph_format.space_after=Pt(7)
    normal.paragraph_format.widow_control=True
    for name,size in [('Title',17),('Heading 1',13),('Heading 2',11.5),('Heading 3',11.5)]:
        st=d.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
        st.font.bold=name!='Title';st.font.underline=False
        st.paragraph_format.space_before=Pt(12 if name!='Title' else 0)
        st.paragraph_format.space_after=Pt(6)
        st.paragraph_format.keep_with_next=True
    d.styles['Caption'].font.name='Times New Roman';d.styles['Caption'].font.size=Pt(10)
    d.styles['Caption'].font.color.rgb=RGBColor(0,0,0)
    d.styles['Caption'].font.bold=False
    d.styles['Caption'].font.italic=False
    d.styles['Caption'].paragraph_format.space_after=Pt(8)
    d.styles['Caption'].paragraph_format.line_spacing=1.02
    for name in ('Normal','Title','Heading 1','Heading 2','Heading 3','Caption'):
        st=d.styles[name]
        for border in list(st.element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
        for fonts in st.element.iter(qn('w:rFonts')):
            for attr in list(fonts.attrib):
                if 'theme' in attr.lower():del fonts.attrib[attr]
            for attr in ('ascii','hAnsi','eastAsia','cs'):
                fonts.set(qn('w:'+attr),'Times New Roman')
        for bold in st.element.iter(qn('w:bCs')):
            bold.set(qn('w:val'),'1' if st.font.bold else '0')
    header=sec.header.paragraphs[0]
    header.text='Supplementary material' if supplement else 'Perturbation structure and imperfect physics'
    header.runs[0].font.name='Times New Roman';header.runs[0].font.size=Pt(9)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');run._r.addnext(field)
    d.core_properties.author='Michael Doane'
    d.core_properties.title='Perturbation structure and imperfect physics in neural PDE forecasting'+(' Supplementary material' if supplement else '')
    d.core_properties.subject='Journal of Computational Science manuscript draft'
    return d

def inline(p, text):
    text=expand_citations(text)
    # Readable native text subscripts in prose; display mathematics uses OMML.
    tokens={'δlast':('δ','last'),'θx':('θ','x'),'fθ':('f','θ'),'sc':('s','c'),'Ft':('F','t'),'φ1':('φ','1'),'φ2':('φ','2')}
    pattern=r'(https?://[^\s]+|δlast|θx|fθ|\bsc\b|\bFt\b|φ1|φ2)'
    for part in re.split(pattern,text):
        if part.startswith(('https://','http://')):
            url=part.rstrip('.,;')
            hyperlink(p,url,url)
            if url!=part:p.add_run(part[len(url):])
        elif part in tokens:
            a,b=tokens[part];p.add_run(a);r=p.add_run(b);r.font.subscript=True
        else:p.add_run(part)

def para(d,text,style=None):
    p=d.add_paragraph(style=style);inline(p,text);return p

def table(d,caption,headers,rows,widths,numeric=None):
    cp=d.add_paragraph(caption,style='Caption');cp.paragraph_format.keep_with_next=True
    t=d.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for i,w in enumerate(widths):t.columns[i].width=Inches(w)
    for i,h in enumerate(headers):t.rows[0].cells[i].text=h
    rpt=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rpt)
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row):cells[i].text=str(v)
    for j,row in enumerate(t.rows):
        prevent=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
        for i,c in enumerate(row.cells):
            c.width=Inches(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;set_cell_borders(c)
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.04
                if caption.startswith(('Table 1.', 'Table 3.')):
                    p.paragraph_format.keep_with_next=j < len(t.rows)-1
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if numeric and i in numeric else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(9.5);r.bold=j==0
    d.add_paragraph().paragraph_format.space_after=Pt(1)
    return t

# Native, editable Office Math. Every object is built afresh to avoid XML reparenting.
def mr(s):
    x=OxmlElement('m:r');pr=OxmlElement('m:rPr');sty=OxmlElement('m:sty');sty.set(qn('m:val'),'p');pr.append(sty);x.append(pr)
    t=OxmlElement('m:t');t.set(qn('xml:space'),'preserve');t.text=s;x.append(t);return x

def part(tag,items):
    x=OxmlElement(tag)
    for item in items:x.append(mr(item) if isinstance(item,str) else item)
    return x

def sub(a,b):
    elements=list(a) if not isinstance(a,str) and a.tag==qn('m:e') else [a]
    return part('m:sSub',[part('m:e',elements),part('m:sub',[b])])

def sup(a,b):
    elements=list(a) if not isinstance(a,str) and a.tag==qn('m:e') else [a]
    return part('m:sSup',[part('m:e',elements),part('m:sup',[b])])

def frac(a,b):
    return part('m:f',[part('m:num',a if isinstance(a,list) else [a]),part('m:den',b if isinstance(b,list) else [b])])

def summation(lower,upper,operand):
    x=OxmlElement('m:nary');pr=OxmlElement('m:naryPr')
    ch=OxmlElement('m:chr');ch.set(qn('m:val'),'∑');pr.append(ch)
    loc=OxmlElement('m:limLoc');loc.set(qn('m:val'),'subSup');pr.append(loc);x.append(pr)
    x.append(part('m:sub',[lower]));x.append(part('m:sup',[upper]));x.append(part('m:e',[operand]));return x

def hat(x):
    a=OxmlElement('m:acc');pr=OxmlElement('m:accPr');c=OxmlElement('m:chr');c.set(qn('m:val'),'̂');pr.append(c);a.append(pr);a.append(part('m:e',[x]));return a

def omath(d,tokens,label=None):
    p=d.add_paragraph();p.paragraph_format.space_after=Pt(5);p.paragraph_format.space_before=Pt(2)
    p.paragraph_format.keep_together=True
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    m=part('m:oMath',tokens)
    for item in m.iter():
        if item.tag==qn('m:r'):
            wp=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Cambria Math');fonts.set(qn('w:hAnsi'),'Cambria Math');wp.append(fonts)
            sz=OxmlElement('w:sz');sz.set(qn('w:val'),'23');wp.append(sz);item.insert(0,wp)
    p._p.append(m)
    if label:p.add_run('   ('+label+')').font.size=Pt(10)
    return p

def equation(d,key):
    P=lambda:sub('P','q');f=lambda:sub('f','θ');dl=lambda:sub('δ','last');yd=lambda:sub('y','δ')
    if key=='target':omath(d,[yd(),' = ',P(),'(', 'u + ',dl(),', a)'],'1')
    elif key=='loss':
        omath(d,['L(θ) = ',sub('L','obs'),'(θ) + λ',sub('L','phys'),'(θ)'],'2')
        omath(d,[sub('L','obs'),' = E',sup(part('m:e',['‖',f(),'(H, A, q) − N(y)‖']),'2')])
        omath(d,[sub('L','phys'),' = E',sup(part('m:e',['‖',f(),'(H + δ, A, q) − N(',yd(),')‖']),'2')])
    elif key=='response':omath(d,['R(δ) = RMS[',frac([P(),'(u + ',dl(),', a) − ',P(),'(u, a)'],'s'),']'],'3')
    elif key=='matching':omath(d,['R(α',sub('δ','ind'),') = R(',sub('δ','smooth'),'),     ',sub('δ','matched'),' = α',sub('δ','ind')],'4')
    elif key=='reduction':omath(d,['MSE reduction (%) = 100(1 − ρ)'],'5')
    elif key=='wave':
        omath(d,[sub('u','t'),' = v,     ',sub('v','t'),' = ',sup('c','2'),sub('u','xx'),' − γv − κu + 0.5A(t) sin ',sub('θ','x')],'S1')
    elif key=='burgers':omath(d,[sub('u','t'),' = ν',sub('u','xx'),' − αu',sub('u','x'),' − δu + 0.4A(t) sin ',sub('θ','x')],'S2')
    elif key=='ks':omath(d,[sub('u','t'),' = −α',sub('u','xx'),' − β',sub('u','xxxx'),' − γu',sub('u','x'),' + A(t)[0.1 sin ',sub('θ','x'),' + 0.05 cos(2',sub('θ','x'),')]'],'S3')
    elif key=='transport':omath(d,[sub('u','t'),' = D',sub('u','xx'),' − v',sub('u','x'),' − γu + A(t)[0.4 sin ',sub('θ','x'),' + 0.15 cos(3',sub('θ','x'),')]'],'S4')
    elif key=='allen':omath(d,[sub('u','t'),' = D',sub('u','xx'),' + r(u − ',sup('u','3'),') + bA(t) sin ',sub('θ','x')],'S5')
    elif key=='gray':
        omath(d,[sub('U','t'),' = D',sub('U','xx'),' − U',sup('V','2'),' + ',sub('F','t'),'(x)(1 − U)'],'S6')
        omath(d,[sub('V','t'),' = ',frac('D','2'),sub('V','xx'),' + U',sup('V','2'),' − [',sub('F','t'),'(x) + k]V'])
        omath(d,[sub('F','t'),'(x) = F[1 + 0.35A(t) sin ',sub('θ','x'),']'])
    elif key=='cahn':omath(d,[sub('u','t'),' = −a',sub('u','xxxx'),' + b',sub(part('m:e',['(',sup('u','3'),' − u)']),'xx'),' + cA(t) sin ',sub('θ','x')],'S7')
    elif key=='fitz':
        omath(d,[sub('u','t'),' = D',sub('u','xx'),' + u − ',frac(sup('u','3'),'3'),' − v + cA(t) sin ',sub('θ','x')],'S8')
        omath(d,[sub('v','t'),' = b(u + 0.7 − 0.8v)'])
    elif key=='metric':
        error=sup(part('m:e',['[',frac([sub(hat('u'),'jhci'),' − ',sub('u','jhci')],sub('s','c')),']']),'2')
        omath(d,[sub('e','jh'),' = ',frac('1','CN'),summation('c = 1','C',summation('i = 1','N',error))],'S9')
    else:raise ValueError(key)

FAMILIES={'wave':'Wave','burgers':'Burgers','ks':'Kuramoto–Sivashinsky','advection_diffusion':'Advection–diffusion','allen_cahn':'Allen–Cahn','gray_scott':'Gray–Scott','cahn_hilliard':'Cahn–Hilliard','fitzhugh_nagumo':'FitzHugh–Nagumo'}
PRIORS={'correct':'Correct','coefficient':'Biased coefficients','structural':'Omitted term'}
ARMS={'none':'No augmentation','iid':'Independent','smooth':'Smooth','response_matched':'Response matched','probe_best':'Best short probe','best_fixed':'Fixed development choice','original_ridge':'Original ridge','guarded':'Guarded'}

def ci(x):return f"[{x[0]:.1f}, {x[1]:.1f}]"
def pv(x):return '<0.001' if x<.001 else f'{x:.3f}'

def add_table(d,key):
    if key=='design':
        table(d,'Table 1. Experimental stages and candidate counts. All candidates train for 4,000 updates. The first stage supplies development evidence for the later selector.',
            ['Stage','Equations and purpose','Grid cells','Seeds','Fits'],[
            ['Development','Six equations; four augmentation arms; correct and biased coefficients','32','6','864'],
            ['Resolution follow-up','Wave, advection–diffusion, Allen–Cahn; smooth and no augmentation; two priors','32 and 64','4','288'],
            ['Selector follow-up','Cahn–Hilliard, FitzHugh–Nagumo; four arms; correct, biased, and omitted-term priors','64','6','432']],
            [1.2,3.45,.85,.5,.6],{2,3,4})
    elif key=='resolution':
        rows=[[FAMILIES[v['family']],PRIORS[v['prior']],f"{v['mse_reduction_percent']:.1f}",ci(v['reduction_ci95_percent']),pv(v['p_holm'])] for v in E['resolution'] if v['grid']==64]
        table(d,'Table 3. Smooth augmentation relative to no augmentation at 64 cells. Reductions and intervals are percentages. Intervals are pointwise 95%; p-values use Holm correction across these six contrasts.',
            ['Equation','Physical prior','Reduction (%)','95% CI (%)','Holm p'],rows,[1.45,1.2,.95,2.2,.8],{2,3,4})
    elif key=='selector':
        rows=[[ARMS[v['baseline']],f"{v['mse_reduction_percent']:.1f}",ci(v['reduction_ci95_percent']),pv(v['p_holm'])] for v in E['selector_primary']]
        table(d,'Table 3. Primary guarded-selector comparisons across two new equations. The pointwise 95% intervals condition on the frozen selector and generated training datasets. Holm correction covers these two tests.',
            ['Baseline','Reduction (%)','95% CI (%)','Holm p'],rows,[2.3,1.1,2.3,.9],{1,2,3})
    elif key=='physical':
        rows=[['Wave','c, γ, κ','1, 0.1, 0.2','2π','0.05'],['Burgers','ν, α, δ','0.08, 1, 0.05','2π','0.1'],['Kuramoto–Sivashinsky','α, β, γ','1, 1, 1','32','0.25'],['Advection–diffusion','D, v, γ','0.06, 1, 0.04','2π','0.1'],['Allen–Cahn','D, r, b','0.035, 1, 0.2','2π','0.1'],['Gray–Scott','D, F, k','0.16, 0.03, 0.04','32','0.5'],['Cahn–Hilliard','a, b, c','0.1, 0.5, 0.05','2π','0.05'],['FitzHugh–Nagumo','D, b, c','0.03, 0.12, 0.2','2π','0.1']]
        table(d,'Table S1. Coefficient centers, domain lengths, and observation intervals for the dimensionless systems. Coefficients appear in the order used for multiplicative bias.',
              ['Equation','Coefficients','Centers','L','Δt'],rows,[1.7,1.0,2.0,.75,.75],{1,2,3,4})
    elif key=='training':
        rows=[['Training / validation trajectories','128 / 24'],['In-distribution / shifted-coefficient test trajectories','64 / 64'],['Retained transitions / burn-in transitions','160 / 24'],['Context length','4 states and 4 actions'],['Training updates / probe budget','4,000 / 300'],['Clean batch / auxiliary batch','32 / 32'],['Augmentation-bank windows','4,096'],['Physical-loss weight / perturbation scale','0.1 / 0.05'],['Width / nominal depth','64 / 3'],['Transformer heads / feedforward width','4 / 128'],['FNO retained real-FFT modes','8 including zero'],['Optimizer / learning rate','AdamW / 0.0003'],['Weight decay / gradient-norm limit','0.0001 / 1'],['Validation interval / rollout horizon','100 updates / 16 steps'],['Primary / secondary test horizon','64 / 96 steps']]
        table(d,'Table S2. Shared training settings in the full experiment profiles.', ['Setting','Value'],rows,[4.2,2.3])
    elif key=='development':
        rows=[[FAMILIES[v['family']],PRIORS[v['prior']],ARMS[v['baseline']],f"{v['reduction_percent']:.1f}",ci(v['adjusted_reduction_ci_percent'])] for v in E['development_mechanism']]
        table(d,'Table S3. Full development mechanism contrasts. The candidate is smooth augmentation throughout. The interval for each percentage reduction is Bonferroni-adjusted across all 24 contrasts (99.792% per comparison).',
            ['Equation','Prior','Baseline','Reduction (%)','Adjusted CI (%)'],rows,[1.45,1.1,1.15,.85,2.05],{3,4})
    elif key=='subsets':
        rows=[]
        for v in E['selector_family_descriptive']:
            rows.append([FAMILIES[v['family']],ARMS[v['baseline']],f"{v['mse_reduction_percent']:.1f}",ci(v['reduction_ci95_percent'])])
        for v in E['selector_prior_descriptive']:
            rows.append([PRIORS[v['prior']],ARMS[v['baseline']],f"{v['mse_reduction_percent']:.1f}",ci(v['reduction_ci95_percent'])])
        table(d,'Table S4. Descriptive selector breakdowns by equation and prior. Each row is a subset of the primary target experiment. Intervals are pointwise 95%.',
            ['Subset','Baseline','Reduction (%)','95% CI (%)'],rows,[1.8,1.8,1,2.0],{2,3})
    elif key=='secondary':
        rows=[['In-distribution' if v['split']=='test' else 'Coefficient shift',v['horizon'],ARMS[v['baseline']],f"{v['mse_ratio']:.3g}",f"{100*(1-v['mse_ratio']):.1f}"] for v in E['selector_secondary']]
        table(d,'Table S5. Descriptive secondary selector results. Ratios compare guarded-selector error with baseline error. These point estimates have no confirmatory test label.',
              ['Test distribution','Horizon','Baseline','MSE ratio','Reduction (%)'],rows,[1.3,.65,2.55,.85,1.05],{1,3,4})
    elif key=='costs':
        rows=[[ARMS[v['policy']],f"{v['mean_target_training_seconds']:.1f}",f"{v['median_target_training_seconds']:.1f}"] for v in E['costs']]
        table(d,'Table S6. Reconstructed target-route costs in seconds. Historical development, selector fitting, and common test evaluation are excluded.',
              ['Policy','Mean seconds','Median seconds'],rows,[3.6,1.5,1.5],{1,2})
    else:raise ValueError(key)

CAPTIONS={}
for i,block in enumerate((WORK/'figure_captions.md').read_text().split('\n\n')[:3]):CAPTIONS[i+1]=block
FIGS={'controls':(1,'figure_1_perturbation_controls'),'resolution':(2,'figure_2_grid_confirmation'),'architecture':(3,'figure_3_architecture_effects')}

def add_figure(d,key):
    n,name=FIGS[key];p=d.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(OUT/'figures'/f'{name}.png'),width=Inches(6.45))
    # Keep figure provenance accessible in the Office drawing.
    for pr in p._p.iter(qn('wp:docPr')):pr.set('descr',CAPTIONS[n])
    d.add_paragraph(CAPTIONS[n],style='Caption')

def hyperlink(p,text,url,size=None):
    h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(url,RT.HYPERLINK,is_external=True))
    r=OxmlElement('w:r');pr=OxmlElement('w:rPr')
    if size:
        sz=OxmlElement('w:sz');sz.set(qn('w:val'),str(round(size*2)));pr.append(sz)
    r.append(pr)
    t=OxmlElement('w:t');t.text=text;r.append(t);h.append(r);p._p.append(h)

def reference_text(v):
    authors=v.get('authors',[])
    if isinstance(authors,str):authors=[authors]
    text=', '.join(authors)+'. '+v['title']+'. '+str(v.get('venue',''))
    if v.get('volume'):text+=' '+str(v['volume'])
    text+=' ('+str(v['year'])+')'
    if v.get('pages') or v.get('article_number'):text+=' '+str(v.get('pages') or v['article_number'])
    if v.get('version'):text+='; version '+str(v['version'])
    if v.get('resource_type')=='software':text+=' [software]'
    if 'preprint' in str(v.get('publication_status',v.get('status',''))).lower() and 'preprint' not in text.lower():text+='; preprint'
    return text+'. '

def references(d):
    for key in CITED:
        v=REFS[key];p=d.add_paragraph()
        p.paragraph_format.left_indent=Inches(.25);p.paragraph_format.first_line_indent=Inches(-.25)
        p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.04
        r=p.add_run(f'[{NUM[key]}] '+reference_text(v));r.font.size=Pt(10.5)
        url='https://doi.org/'+v['doi'] if v.get('doi') else v.get('url')
        if url:hyperlink(p,url,url,size=10.5)

def build(text,supplement=False):
    d=new_doc(supplement)
    for block in text.split('\n\n'):
        block=block.strip()
        if not block:continue
        if block.startswith('# '):para(d,block[2:],'Title')
        elif block.startswith('### '):para(d,block[4:],'Heading 2')
        elif block.startswith('## '):para(d,block[3:],'Heading 1')
        elif block.startswith('[[TABLE '):add_table(d,block[8:-2])
        elif block.startswith('[[FIGURE '):add_figure(d,block[9:-2])
        elif block.startswith('[[EQUATION '):equation(d,block[11:-2])
        elif block=='[[REFERENCES]]':references(d)
        else:
            p=para(d,block)
            if block.startswith('Keywords:'):
                for r in p.runs:r.font.size=Pt(10.5)
    return d

# Revision-specific authoring functions retain native Word equations and tables.
_original_new_doc = new_doc
def new_doc(supplement=False):
    d=_original_new_doc(supplement)
    d.sections[0].header.paragraphs[0].text='Supplementary material' if supplement else 'Physical and neural forecast selection'
    d.sections[0].header.paragraphs[0].runs[0].font.size=Pt(9)
    d.core_properties.title='Choosing between physical and neural forecasters with imperfect equations'
    return d

_original_equation=equation
def equation(d,key):
    if key=='calibration':
        omath(d,[sub('q','cal'),' = q ⊙ exp(z)'],'1')
    elif key=='auxiliary_loss':
        omath(d,['L = ',sub('L','observed'),' + λ',sub('L','physical')],'2')
    elif key=='main_metric':
        error=sup(part('m:e',['[',frac([sub(hat('u'),'jhci'),' − ',sub('u','jhci')],sub('s','c')),']']),'2')
        omath(d,[sub('e','jh'),' = ',frac('1','CN'),summation('c = 1','C',summation('i = 1','N',error))],'3')
    else:_original_equation(d,key)

def sig(x):
    if x==0:return '0'
    text=f'{x:.3g}'
    if 'e' in text:
        base,exponent=text.split('e')
        return base+' × 10'+str(int(exponent)).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))
    return text

DISPLAY_STAGE={'development':'Dev 32','resolution_g32':'Follow-up 32','resolution_g64':'Follow-up 64','selector_g64':'New equations 64'}
CASES=[('resolution_g64','wave','coefficient'),('resolution_g64','advection_diffusion','coefficient'),
       ('resolution_g64','allen_cahn','coefficient'),('development','burgers','coefficient'),
       ('development','ks','coefficient'),('development','gray_scott','coefficient'),
       ('selector_g64','cahn_hilliard','coefficient'),('selector_g64','fitzhugh_nagumo','coefficient'),
       ('selector_g64','cahn_hilliard','structural'),('selector_g64','fitzhugh_nagumo','structural')]

def getrows(stage,family,prior,split='test',horizon=64):
    return {r['method']:r for r in C['absolute_accuracy'] if r['stage']==stage and r['family']==family and r['prior']==prior and r['split']==split and r['horizon']==horizon}

_original_add_table=add_table
def add_table(d,key):
    if key=='design':
        table(d,'Table 1. Experiment sequence and the available forecasting candidates. Neural counts are completed fits. Calibration settings reuse their observed trajectories and archived neural errors.',
              ['Stage','Purpose and systems','Grid','Neural fits','Calibration settings'],[
              ['Development','Six original equations; four augmentation arms and two physical priors','32','864','12'],
              ['Resolution follow-up','Wave, advection–diffusion, Allen–Cahn; smooth and ordinary training','32 and 64','288','12'],
              ['Selector follow-up','Cahn–Hilliard, FitzHugh–Nagumo; four arms and three physical priors','64','432','6']],
              [1.25,3.15,.6,.75,.85],{2,3,4})
    elif key=='absolute':
        rows=[]
        for stage,family,prior in CASES:
            rr=getrows(stage,family,prior)
            rows.append([FAMILIES[family], 'Biased' if prior=='coefficient' else 'Missing term',rr['calibrated']['grid'],
                         sig(rr['uncalibrated']['mean_nmse']),sig(rr['calibrated']['mean_nmse']),sig(rr['validation_best_neural']['mean_nmse'])])
        table(d,'Table 2. Primary 64-step ordinary-test normalized MSE. One representative setting is shown per equation for biased coefficients, followed by both omitted-term cases. Neural candidates are selected by validation for each seed. Values use up to three significant figures; complete intervals and all settings are in Supplementary Table S9 and the release.',
              ['Equation','Physical error','Grid','Uncalibrated physics','Calibrated physics','Selected neural'],rows,[1.55,.8,.5,1.1,1.15,1.5],{2,3,4,5})
    elif key=='computation':
        table(d,'Table 4. Current CPU computation. Runtime components refer to the additional calibration experiment after setup. Step times are medians across the 30 setting-specific median timings. Neural rows measure initialized copies of the archived architectures.',
              ['Measurement','Time','Scope'],[
              ['Complete new experiment','85 s','Includes regeneration, calibration, evaluation and timing'],
              ['Data regeneration','49 s','Original generator with eight CPU threads'],
              ['Calibration phase','9.2 s','Four concurrent fitting processes'],
              ['Calibration per setting','0.72 s','Median candidate fitting, validation and setup'],
              ['Calibrated physical step','0.45 ms','One trajectory; includes coefficient correction'],
              ['Fourier neural step','0.67 ms','Initialized architecture; full normalization'],
              ['Standard transformer step','1.2 ms','Initialized architecture; full normalization'],
              ['Looped transformer step','1.2 ms','Initialized architecture; full normalization']],
              [2.05,.8,3.75],{1})
    elif key=='calibration':
        rows=[]
        for j in C['jobs']:
            rows.append([FAMILIES[j['family']],DISPLAY_STAGE[j['stage']],PRIORS[j['prior']],
                         ', '.join(sig(v) for v in j['selected_multipliers']),sig(j['validation_nmse'])])
        table(d,'Table S7. Selected physical corrections in all 30 settings. Multipliers are rounded to three significant figures for reading; exact values and every fitted candidate are preserved in the result files.',
              ['Equation','Stage and grid','Prior','Selected multipliers','Validation NMSE'],rows,[1.5,1.2,1.05,1.7,1.15],{3,4})
    elif key=='choices':
        from collections import Counter
        rows=[]
        for j in C['jobs']:
            count=Counter(x['selected_method'] for x in j['frozen_decisions'])
            rows.append([FAMILIES[j['family']],DISPLAY_STAGE[j['stage']],PRIORS[j['prior']],count['uncalibrated'],count['calibrated'],count['neural']])
        table(d,'Table S8. Validation-selected forecasting classes by setting. Counts are seed-specific decisions sharing the same physical fit and data. Every selected neural model is an unaugmented Fourier operator in the omitted-term settings.',
              ['Equation','Stage and grid','Prior','Uncalibrated','Calibrated','Neural'],rows,[1.5,1.2,1.15,.95,.95,.85],{3,4,5})
    elif key=='absolute_full':
        rows=[]
        for j in C['jobs']:
            for split in ('test','ood'):
                rr=getrows(j['stage'],j['family'],j['prior'],split)
                vals=[]
                for name in ('calibrated','validation_best_neural'):
                    a=rr[name];vals.append(sig(a['mean_nmse'])+'\n['+sig(a['ci95_low'])+',\n'+sig(a['ci95_high'])+']')
                rows.append([FAMILIES[j['family']]+'\n'+DISPLAY_STAGE[j['stage']],PRIORS[j['prior']], 'Ordinary' if split=='test' else 'Shifted',*vals])
        table(d,'Table S9. Absolute 64-step errors and exploratory pointwise 95% intervals for every setting and both test distributions. Physical calibration and training data are held fixed in the bootstrap. The release contains the 96-step results and unchanged-physics, persistence, and individual neural candidates.',
              ['Equation stage and grid','Prior','Test set','Calibrated physics [95% CI]','Selected neural [95% CI]'],rows,[1.5,.95,.8,1.7,1.65],{3,4})
    else:_original_add_table(d,key)

newcaps=json.loads((WORK/'figures/captions.json').read_text())
oldcaps=(WORK/'figure_captions.md').read_text().split('\n\n')[:3]
CAPTIONS={1:'Figure 1. '+newcaps['figure1'],2:re.sub(r'^Figure 3\.', 'Figure 2.',oldcaps[2]),
          3:'Figure 3. '+newcaps['figure2'],'S1':re.sub(r'^Figure 1\.', 'Figure S1.',oldcaps[0]),
          'S2':re.sub(r'^Figure 2\.', 'Figure S2.',oldcaps[1])}
FIGS={'model_choice':(1,'figure1_absolute_accuracy'),'architecture':(2,'figure_3_architecture_effects'),
      'timing':(3,'figure2_cpu_execution'),'controls':('S1','figure_1_perturbation_controls'),
      'resolution':('S2','figure_2_grid_confirmation')}

def main():
    (OUT/'figures').mkdir(exist_ok=True)
    for p in (WORK/'figures').iterdir():
        if p.is_file() and p.suffix in ('.png','.pdf','.svg','.json','.csv','.py'):
            destination=OUT/'figures'/p.name
            if p.resolve()!=destination.resolve():shutil.copy2(p,destination)
    build(MAIN).save(OUT/'Physics_JOCS_Manuscript_Draft.docx')
    build(SUPP,True).save(OUT/'Physics_JOCS_Supplement_Draft.docx')
    highlights=[
        'Coefficient calibration reverses the primary neural advantage in eight PDE families.',
        'Neural forecasts remain more accurate in both tested missing-term systems.',
        'Validation selects physics for complete equations and neural models for omissions.',
        'Calibrated physics executes faster in all 30 matched CPU implementation comparisons.',
        'Physical supervision helps some neural forecasts and substantially harms others.'
    ]
    assert 3<=len(highlights)<=5 and all(len(v)<=85 for v in highlights)
    h=new_doc();h.add_paragraph('Highlights',style='Title')
    for value in highlights:h.add_paragraph(value,style='List Bullet')
    h.save(OUT/'Physics_JOCS_Highlights.docx')
    sections=[('01_Abstract_and_Introduction.docx','Abstract and introduction','## Abstract','## 2 Methods'),
              ('02_Methods.docx','Methods','## 2 Methods','## 3 Results'),
              ('03_Results.docx','Results','## 3 Results','## 4 Discussion'),
              ('04_Discussion_and_Conclusions.docx','Discussion and conclusions','## 4 Discussion','## Data and code availability')]
    for filename,title,start,end in sections:
        content=MAIN[MAIN.index(start):MAIN.index(end)]
        content=re.sub(r'\[\[(?:TABLE|FIGURE) [^\]]+\]\]\s*','',content)
        doc=build('# '+title+'\n\n'+content)
        doc.sections[0].header.paragraphs[0].text='Section review copy'
        if filename.startswith('04_'):
            doc.styles['Normal'].paragraph_format.space_after=Pt(5)
            doc.styles['Normal'].paragraph_format.line_spacing=1.06
        assert len(doc.tables)==0
        assert all(p.style.name!='Caption' for p in doc.paragraphs)
        doc.save(OUT/filename)
    ordered=[REFS[k] for k in CITED]
    (OUT/'references.json').write_text(json.dumps(ordered,indent=2,ensure_ascii=False)+'\n')
    bib=[]
    for r in ordered:
        fields={'title':r['title'],'author':' and '.join('{'+a+'}' if 'Contributors' in a else a for a in r['authors']),
                'year':str(r['year']),'journal':r.get('venue',''),'volume':str(r.get('volume','')),
                'pages':str(r.get('pages',r.get('article_number',''))).replace('–','--'),
                'doi':r.get('doi',''),'url':r.get('url','')}
        if r.get('bibtex_type')=='inproceedings':fields['booktitle']=fields.pop('journal')
        bib.append('@'+r.get('bibtex_type','article')+'{'+r['key']+',\n'+',\n'.join('  '+k+' = {'+v+'}' for k,v in fields.items() if v)+'\n}')
    (OUT/'references.bib').write_text('\n\n'.join(bib)+'\n')
    fullref='\n\n'.join(f'[{NUM[k]}] '+reference_text(REFS[k])+(('https://doi.org/'+REFS[k]['doi']) if REFS[k].get('doi') else REFS[k].get('url','')) for k in CITED)
    (OUT/'manuscript.md').write_text(expand_citations(MAIN).replace('[[REFERENCES]]',fullref))
    (OUT/'supplement.md').write_text(expand_citations(SUPP))
    (OUT/'figure_captions.txt').write_text('\n\n'.join(CAPTIONS.values())+'\n')
    (OUT/'highlights.txt').write_text('\n'.join(highlights)+'\n')
    abstract=MAIN.split('## Abstract\n\n')[1].split('\n\nKeywords:')[0]
    report={'title':MAIN.splitlines()[0][2:],'abstract_words':len(abstract.split()),'reference_count':len(CITED),
            'main_words_before_references':len(MAIN.split('## References')[0].split()),'highlight_characters':[len(v) for v in highlights],
            'docx_files':[p.name for p in OUT.glob('*.docx')],'review_sections_have_tables':False,'review_sections_have_captions':False,
            'native_equations':True}
    assert report['abstract_words']<=250
    (OUT/'content_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
