"""Self-contained reports over the same analysis returned by the Python API."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#18232b;--muted:#526574;--line:#d8e0e5;--accent:#096c75;--paper:#fff;--wash:#f4f7f8}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:auto;padding:44px 28px 70px}header{border-bottom:2px solid var(--ink);padding-bottom:25px;margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700}h1{font-size:clamp(27px,4vw,42px);line-height:1.15;letter-spacing:-.025em;margin:12px 0}
h2{font-size:20px;margin:30px 0 12px}.muted{color:var(--muted)}.meta{font-size:13px;overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}
.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:17px}.card strong{display:block;font-size:29px;font-weight:650;letter-spacing:-.025em}.label{font-size:12px;color:var(--muted)}
.panel{border:1px solid var(--line);border-radius:8px;background:var(--paper);padding:20px;margin:16px 0}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:15px 0}
input,select,button{font:inherit;border:1px solid #aebdc6;border-radius:5px;padding:8px 12px;background:white;color:var(--ink)}input{min-width:240px;flex:1}button{cursor:pointer}button:hover{background:#eaf3f4}button:disabled{opacity:.4;cursor:default}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;max-width:420px;overflow-wrap:anywhere}th{background:#eef3f5;font-weight:650;white-space:nowrap}th button{border:0;padding:0;background:none;font:inherit;text-align:left}tbody tr:last-child td{border-bottom:0}.num{font-variant-numeric:tabular-nums}.note{font-size:13px;color:var(--muted)}details{margin-top:16px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}footer{margin-top:30px;color:var(--muted);font-size:12px}
@media print{body{background:white}main{padding:16px}.toolbar,button{display:none}.panel{break-inside:avoid}.scroll{overflow:visible}td,th{padding:6px}details{display:none}}
</style></head><body><main>
<header><div class="eyebrow">EnvFoundry · AI Evaluation</div><h1>__TITLE__</h1><div id="identity" class="meta muted"></div></header>
<div id="cards" class="cards"></div><div id="method" class="panel"></div>
<section id="slice-section" hidden><h2>Slices</h2><div id="slices" class="scroll panel"></div></section>
<section><h2 id="sample-heading">Samples</h2><div class="toolbar"><label for="filter">Filter</label><input id="filter" type="search" placeholder="Sample, group or value"><label for="page-size">Rows</label><select id="page-size"><option>25</option><option>50</option><option>100</option></select><button id="download" type="button">Download analysis JSON</button></div>
<div id="sample-table" class="scroll panel"></div><div class="toolbar"><button id="previous" type="button">Previous</button><span id="page-label" class="note" aria-live="polite"></span><button id="next" type="button">Next</button></div></section>
<details><summary>Analysis and coverage</summary><pre id="details"></pre></details>
<footer>Generated from the recorded analysis. Score differences are right minus left; improvement accounts for the metric direction. Intervals use the reported resampling unit and method.</footer>
</main><script id="analysis" type="application/json">__DATA__</script><script>
"use strict";
const analysisText=document.getElementById("analysis").textContent;
const analysis=JSON.parse(analysisText);
const $=id=>document.getElementById(id);
const fmt=v=>v===null||v===undefined?"—":typeof v==="number"?(Number.isInteger(v)?String(v):Number(v.toPrecision(6)).toString()):typeof v==="object"?JSON.stringify(v):String(v);
const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=fmt(value);if(cls)e.className=cls;return e;};
const comparison=Object.hasOwn(analysis,"delta");
$("identity").textContent=[analysis.metric&&"Metric: "+analysis.metric,analysis.direction,analysis.benchmark_id&&"Benchmark: "+analysis.benchmark_id].filter(Boolean).join(" · ");
function card(label,value){const d=document.createElement("div");d.className="card";d.append(text("span",label,"label"),text("strong",value,"num"));$("cards").append(d);}
if(comparison){card("Baseline mean",analysis.left_mean);card("Candidate mean",analysis.right_mean);card("Score difference",analysis.delta);card("Direction-adjusted improvement",analysis.improvement);}else card("Weighted sample mean",analysis.mean??analysis.estimate);
if(analysis.interval)card(Math.round(analysis.interval.confidence*100)+"% interval",fmt(analysis.interval.lower)+" to "+fmt(analysis.interval.upper));
const coverage=analysis.coverage||{};
if(comparison){card("Complete paired samples",analysis.paired_samples??coverage.retained_samples);if(coverage.excluded_samples)card("Excluded samples",coverage.excluded_samples);}
for(const key of ["paired_samples","scored_samples","total_samples","scored_trials","planned_trials"]){if(Object.hasOwn(coverage,key))card(key.replaceAll("_"," "),coverage[key]);}
const interpretations={sample_weighted_mean_of_observed_trial_means:"Trial scores are averaged within each sample, then sample means are averaged using their declared weights. Incomplete samples contribute their observed scores; coverage describes what remains unscored.",sample_weighted_mean_of_complete_paired_trial_mean_differences:"Candidate minus baseline on complete paired samples, using declared sample weights. The comparison keeps the same cases and independent groups together. Excluded samples change the population represented by the estimate.",sample_weighted_probability_of_success_in_k_attempts:"Estimated probability of at least one success within k attempts, averaged with declared sample weights. This interpretation assumes independent attempts from the same sampling policy."};
$("method").append(text("strong","Interpretation"),text("p",interpretations[analysis.estimand]||analysis.estimand||"The recorded metric is analysed on its stated sample population."));
if(analysis.interval)$("method").append(text("p","Interval: "+[analysis.interval.method,analysis.interval.independent_groups!==undefined?analysis.interval.independent_groups+" independent groups":null,analysis.interval.resamples!==undefined?analysis.interval.resamples+" resamples":null].filter(Boolean).join("; "),"note"));
else $("method").append(text("p","No interval is available for this result; inspect the recorded coverage and independent-group count.","note"));
const preferred=["sample_id","group","weight","left_mean","right_mean","left","right","mean","delta","improvement","planned_trials","observed_trials","scored_trials","complete","n","successes","estimate"];
function columns(rows){const keys=new Set(rows.flatMap(Object.keys));return [...preferred.filter(k=>keys.has(k)),...Array.from(keys).filter(k=>!preferred.includes(k)&&rows.some(r=>r[k]===null||typeof r[k]!=="object"))];}
function table(rows,cols,onSort){const t=document.createElement("table");t.setAttribute("aria-label",onSort?"Per-sample results":"Slice results");const head=t.createTHead().insertRow();for(const col of cols){const th=document.createElement("th");th.scope="col";if(onSort){th.setAttribute("aria-sort",col===sortKey?(ascending?"ascending":"descending"):"none");const b=text("button",col.replaceAll("_"," ")+(col===sortKey?(ascending?" ↑":" ↓"):""));b.type="button";b.addEventListener("click",()=>onSort(col));th.append(b);}else th.textContent=col.replaceAll("_"," ");head.append(th);}const body=t.createTBody();for(const row of rows){const tr=body.insertRow();for(const col of cols){const td=tr.insertCell();td.textContent=fmt(row[col]);if(typeof row[col]==="number")td.className="num";}}return t;}
if(Array.isArray(analysis.slices)&&analysis.slices.length){$("slice-section").hidden=false;const slices=analysis.slices.map(s=>({slice:s.missing?"(missing label)":s.value,...(comparison?{left_mean:s.left_mean,right_mean:s.right_mean,delta:s.delta,improvement:s.improvement,paired_samples:s.paired_samples,excluded_samples:s.coverage?.excluded_samples}:{mean:s.mean,scored_samples:s.coverage?.scored_samples,total_samples:s.coverage?.total_samples}),interval:s.interval?fmt(s.interval.lower)+" to "+fmt(s.interval.upper):null}));$("slices").append(table(slices,["slice",...columns(slices).filter(c=>c!=="slice")]));}
const rows=Array.isArray(analysis.samples)?analysis.samples:[];const cols=columns(rows);let page=0,sortKey=null,ascending=true;
function render(){const q=$("filter").value.toLowerCase();let filtered=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(q));if(sortKey)filtered.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];const v=x==null?(y==null?0:1):y==null?-1:typeof x==="number"&&typeof y==="number"?x-y:String(x).localeCompare(String(y));return ascending?v:-v;});const size=Number($("page-size").value);const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);$("sample-table").replaceChildren(filtered.length?table(filtered.slice(page*size,(page+1)*size),cols,key=>{ascending=key===sortKey?!ascending:true;sortKey=key;render();}):text("p","No matching samples.","muted"));$("page-label").textContent=filtered.length+" matching samples · page "+(page+1)+" of "+pages;$("previous").disabled=page===0;$("next").disabled=page===pages-1;}
$("filter").addEventListener("input",()=>{page=0;render();});$("page-size").addEventListener("change",()=>{page=0;render();});$("previous").addEventListener("click",()=>{page--;render();});$("next").addEventListener("click",()=>{page++;render();});
$("download").addEventListener("click",()=>{const u=URL.createObjectURL(new Blob([analysisText+"\n"],{type:"application/json"}));const a=document.createElement("a");a.href=u;a.download="evaluation-analysis.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});
const details={...analysis};delete details.samples;$("details").textContent=JSON.stringify(details,null,2);render();
</script></body></html>'''


def write_report(analysis: dict, output: str | Path, *, title: str | None = None) -> dict:
    """Write a new portable HTML report containing the complete analysis JSON."""
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object")
    title = title or ("Paired system comparison" if "delta" in analysis else "Evaluation results")
    data = json.dumps(analysis, ensure_ascii=False, sort_keys=True, allow_nan=False)
    data = (data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    replacements = {"TITLE": html.escape(title, quote=True), "DATA": data}
    page = re.sub(r"__(TITLE|DATA)__", lambda match: replacements[match[1]], _PAGE)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(page)
    return {"path": str(output.resolve()), "bytes": output.stat().st_size}
