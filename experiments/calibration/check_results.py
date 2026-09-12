#!/usr/bin/env python3
"""Run with Jupyter %run or Python to print the saved scientific results."""
from pathlib import Path
import argparse
from analysis import analyze

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=root/'outputs'/'full')
    args=parser.parse_args()
    report=analyze(args.output.resolve(),root/'assets')
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import HTML,display
            color={'GREEN':'#157347','YELLOW':'#997404','RED':'#b02a37'}.get(report['completion'],'#495057')
            display(HTML('<div style="padding:16px;border-left:8px solid '+color+';background:#f4f5f6;color:#212529">'
                         '<b>Completion: '+report['completion']+'</b><br>'+
                         str(report['verified_settings'])+'/'+str(report['requested_settings'])+' settings verified.<br>'+
                         'Scientific evidence: '+report['scientific_evidence']+'<br>'+
                         'Individual comparison labels and uncertainty intervals appear above.</div>'))
    except ImportError:
        pass
    return 0 if report['completion']=='GREEN' else 1
if __name__=='__main__': raise SystemExit(main())
