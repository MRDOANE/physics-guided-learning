import json, pathlib, statistics
p=pathlib.Path(__file__).resolve().parent
settings=[('ljung2010','ljung2010identification','Established fitting of dynamical-system models from observed input-output data. Cite when introducing physical coefficient fitting; this baseline is an application of system identification, not a new parameter-estimation method.',['https://doi.org/10.1016/j.arcontrol.2009.12.001','https://isy.gitlab-pages.liu.se/staff/lenlj48/seoul2dvinew/plenary2.pdf']),('brynjarsdottir2014','brynjarsdottir2014discrepancy','Distinguishes uncertainty in physical parameters from errors in the simulator form. Use to qualify parameter interpretation when equation terms are omitted; do not describe the present least-squares procedure as Bayesian calibration.',['https://doi.org/10.1088/0266-5611/30/11/114007','https://www.tonyohagan.co.uk/academic/pub.html']),('virtanen2020','virtanen2020scipy','Credits SciPy, whose bounded least_squares optimizer is used in the calibration implementation.',['https://doi.org/10.1038/s41592-019-0686-2','https://scipy.org/citing-scipy/'])]
refs=[]
for name,key,note,sources in settings:
 x=json.loads((p/(name+'_crossref.json')).read_text())['message']
 authors=[]
 for a in x['author']:
  authors.append(a.get('name') or (a.get('given','')+' '+a['family']).strip())
  if name=='virtanen2020' and a.get('name')=='SciPy 1.0 Contributors':break
 if name=='brynjarsdottir2014':authors=["Jenný Brynjarsdóttir","Anthony O'Hagan"]
 refs.append(dict(key=key,title=x['title'][0],authors=authors,year=2020 if name=='virtanen2020' else x['published']['date-parts'][0][0],venue=x['container-title'][0],volume=x.get('volume'),issue=x.get('issue'),pages=x.get('page'),doi=x['DOI'],url='https://doi.org/'+x['DOI'],publication_status='published',bibtex_type='article',relevance_note=note,verified_sources=sources,verification='Crossref publisher-deposited metadata; primary publisher/author/project page checked at the level described in JOURNAL_AUDIT.md',checked_on='2026-09-12'))
(p/'reference_additions.json').write_text(json.dumps(refs,indent=2,ensure_ascii=False)+'\n')
lines=[]
for r in refs:
 def author(a):
  if a=='SciPy 1.0 Contributors':return '{SciPy 1.0 Contributors}'
  return a
 lines += ['@article{'+r['key']+',','  title = {'+r['title']+'},','  author = {'+' and '.join(author(a) for a in r['authors'])+'},','  journal = {'+r['venue']+'},','  year = {'+str(r['year'])+'},','  volume = {'+r['volume']+'},','  number = {'+r['issue']+'},','  pages = {'+r['pages'].replace('-','--')+'},','  doi = {'+r['doi']+'},','  url = {'+r['url']+'}','}','']
(p/'reference_additions.bib').write_text('\n'.join(lines))
rows=json.loads((p/'jocs_sample_crossref.json').read_text())['message']['items']
sample=[dict(title=x['title'][0],doi=x['DOI'],year=x['published']['date-parts'][0][0],reference_count=x['reference-count'],url='https://doi.org/'+x['DOI']) for x in rows]
summary=dict(retrieval_date='2026-09-12',method='First 12 results returned in Crossref relevance order for journal ISSN 1877-7503, query neural physics, journal-article type, publication dates 2023-01-01 through 2026-09-12. All 12 concern physical neural modeling. This is a small relevance-selected comparison sample, not a random sample or a journal requirement.',source_url='https://api.crossref.org/journals/1877-7503/works?filter=from-pub-date:2023-01-01,until-pub-date:2026-09-12,type:journal-article&query=neural%20physics&rows=12',n=len(sample),median=statistics.median(x['reference_count'] for x in sample),minimum=min(x['reference_count'] for x in sample),maximum=max(x['reference_count'] for x in sample),items=sample)
(p/'reference_count_sample.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
lines=['# Journal of Computational Science — reference-count comparison sample','',f"The median publisher-deposited reference count in this 12-paper sample is **{summary['median']:g}**, with a range of {summary['minimum']}–{summary['maximum']}. The revised manuscript's 43 references would sit within this observed range. Coverage of the scientific questions should determine the final list; the median supplies context and is not a quota.",'',summary['method'],'','| Article | Year | References |','| --- | ---: | ---: |']
for x in sample:lines.append(f"| [{x['title']}]({x['url']}) | {x['year']} | {x['reference_count']} |")
lines+=['','Counts are Crossref `reference-count` values deposited by the publisher and were not manually recounted from each typeset PDF. The raw response and retrieval URL are retained in this directory. A broader 62-result retrieval is also retained; it contains unrelated neural-network applications and was not used as the physical-modeling comparison sample.']
(p/'REFERENCE_COUNT_SAMPLE.md').write_text('\n'.join(lines)+'\n')
print('Wrote',len(refs),'added references; sample median',summary['median'])
