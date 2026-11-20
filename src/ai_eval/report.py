"""Portable, offline views of the analysis returned by the Python API."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re

from .contracts import _parse


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title><style>
:root{color-scheme:light;--ink:#18232b;--muted:#526574;--line:#d8e0e5;--accent:#096c75;--second:#a35324;--paper:#fff;--wash:#f4f7f8}
*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:auto;padding:42px 28px 65px}header{border-bottom:2px solid var(--ink);padding-bottom:22px;margin-bottom:24px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700}h1{font-size:clamp(27px,4vw,42px);line-height:1.15;letter-spacing:-.025em;margin:12px 0}
h2{font-size:20px;margin:28px 0 12px}p{margin:8px 0}.muted{color:var(--muted)}.meta{font-size:13px;overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.card{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:16px}.card strong{display:block;font-size:27px;font-weight:650;letter-spacing:-.025em;overflow-wrap:anywhere}.label{font-size:12px;color:var(--muted)}
.panel{border:1px solid var(--line);border-radius:8px;background:var(--paper);padding:18px;margin:14px 0}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin:14px 0}
input,select,button{font:inherit;border:1px solid #aebdc6;border-radius:5px;padding:8px 12px;background:white;color:var(--ink)}input{min-width:150px;flex:1}button{cursor:pointer}button:hover{background:#eaf3f4}button:disabled{opacity:.4;cursor:default}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;max-width:380px;overflow-wrap:anywhere}th{background:#eef3f5;font-weight:650;white-space:nowrap}th button{border:0;padding:0;background:none;font:inherit;text-align:left}tbody tr:last-child td{border-bottom:0}.num{font-variant-numeric:tabular-nums}.note{font-size:13px;color:var(--muted)}
.chart svg{display:block;width:100%;height:auto;min-width:460px;max-height:510px}.chart{overflow:auto}.legend{display:flex;flex-wrap:wrap;gap:20px;font-size:13px;margin:4px 0 8px}.legend span{display:inline-flex;align-items:center;gap:7px}.swatch{width:11px;height:11px;border-radius:50%;display:inline-block}details{margin-top:18px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:420px;overflow:auto}footer{margin-top:30px;color:var(--muted);font-size:12px}
@media(max-width:600px){main{padding:26px 14px 45px}.panel{padding:12px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.card strong{font-size:23px}.toolbar input{width:100%}}
@media print{body{background:white}main{padding:16px}.toolbar,button{display:none}.panel{break-inside:avoid}.scroll{overflow:visible}td,th{padding:6px}details{display:none}}
</style></head><body><main>
<header><div class="eyebrow">AI Evaluation</div><h1>__TITLE__</h1><div id="identity" class="meta muted"></div></header>
<div id="cards" class="cards"></div><div id="method" class="panel"></div><div id="results"></div>
<section id="slice-section" hidden><h2 id="slice-heading">Slices</h2><div id="slices" class="scroll panel"></div></section>
<section><h2 id="sample-heading">Samples</h2><div class="toolbar"><label for="filter">Filter</label><input id="filter" type="search" placeholder="Sample, group or value"><label for="page-size">Rows</label><select id="page-size"><option>25</option><option>50</option><option>100</option></select><button id="download" type="button">Download analysis JSON</button></div>
<div id="sample-table" class="scroll panel"></div><div class="toolbar"><button id="previous" type="button">Previous</button><span id="page-label" class="note" aria-live="polite"></span><button id="next" type="button">Next</button></div></section>
<details><summary>Analysis and coverage</summary><pre id="details"></pre></details>
<footer>Tables and plots present the recorded analysis. The downloadable JSON contains all results, sample details and exclusions.</footer>
</main><script id="analysis" type="application/json">__DATA__</script><script>
"use strict";
const analysisText = document.getElementById("analysis").textContent;
const analysis = JSON.parse(analysisText);
const $ = id => document.getElementById(id);
const has = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const fmt = value => value == null ? "—" : typeof value === "number"
    ? (Number.isInteger(value) ? String(value) : Number(value.toPrecision(6)).toString())
    : typeof value === "object" ? JSON.stringify(value) : String(value);
const words = value => String(value).replaceAll("_", " ");
const text = (tag, value, className) => {
    const element = document.createElement(tag);
    element.textContent = fmt(value);
    if (className) element.className = className;
    return element;
};
const intervalText = (interval, reason) => interval
    ? `${fmt(interval.lower)} to ${fmt(interval.upper)} (${fmt(100 * interval.confidence)}%)`
    : reason ? `— (${words(reason)})` : "—";
const systemName = system => system.label || [system.name, system.revision].filter(value => value != null && value !== "").join(" · ");
const comparison = has(analysis, "delta");
const kind = analysis.kind || (comparison ? "comparison" : "summary");
const coverage = analysis.coverage || {};
const factorA = analysis.factors?.a || "A";
const factorB = analysis.factors?.b || "B";
const effectNames = {
    main_a: `${factorA} main effect`, main_b: `${factorB} main effect`,
    interaction: `${factorA} × ${factorB} interaction`,
    a_at_b0: `${factorA} at ${factorB} = 0`, a_at_b1: `${factorA} at ${factorB} = 1`,
    b_at_a0: `${factorB} at ${factorA} = 0`, b_at_a1: `${factorB} at ${factorA} = 1`
};
const directionLabel = analysis.direction === "minimize" ? "lower is better" : analysis.direction === "maximize" ? "higher is better" : "";
const benchmarkName = analysis.benchmark && [analysis.benchmark.name, analysis.benchmark.revision].filter(Boolean).join(" · ");
$("identity").textContent = [benchmarkName && `Benchmark: ${benchmarkName}`,
    analysis.metric && `Metric: ${analysis.metric}`, directionLabel,
    analysis.system && `System: ${systemName(analysis.system)}`].filter(Boolean).join("  /  ");

function card(label, value) {
    const element = text("div", "", "card");
    element.append(text("span", label, "label"), text("strong", value, "num"));
    $("cards").append(element);
}
function table(rows, columns, options = {}) {
    const result = document.createElement("table");
    result.setAttribute("aria-label", options.label || "Analysis results");
    const header = result.createTHead().insertRow();
    for (const column of columns) {
        const cell = document.createElement("th");
        cell.scope = "col";
        if (options.onSort) {
            const active = column.key === options.sortKey;
            cell.setAttribute("aria-sort", active ? (options.ascending ? "ascending" : "descending") : "none");
            const button = text("button", column.label + (active ? (options.ascending ? " ↑" : " ↓") : ""));
            button.type = "button";
            button.addEventListener("click", () => options.onSort(column.key));
            cell.append(button);
        } else cell.textContent = column.label;
        header.append(cell);
    }
    const body = result.createTBody();
    for (const row of rows) {
        const tr = body.insertRow();
        for (const column of columns) {
            const cell = tr.insertCell();
            cell.textContent = fmt(row[column.key]);
            if (typeof row[column.key] === "number") cell.className = "num";
        }
    }
    return result;
}
const cols = entries => entries.map(entry => Array.isArray(entry) ? {key: entry[0], label: entry[1]} : {key: entry, label: words(entry)});
function resultPanel(heading, note) {
    const section = document.createElement("section");
    section.append(text("h2", heading));
    const panel = text("div", "", "panel scroll");
    section.append(panel);
    if (note) section.append(text("p", note, "note"));
    $("results").append(section);
    return panel;
}

// SVG uses the same recorded estimates as the tables; no statistics are recomputed.
const svgNS = "http://www.w3.org/2000/svg";
function svgElement(tag, attributes = {}, value) {
    const element = document.createElementNS(svgNS, tag);
    for (const [key, item] of Object.entries(attributes)) element.setAttribute(key, String(item));
    if (value !== undefined) element.textContent = String(value);
    return element;
}
function scale(values, start, end, floor = null) {
    const unit = values.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0) || 1;
    const normalized = values.map(value => value / unit);
    let low = Math.min(...normalized), high = Math.max(...normalized);
    const padding = (high - low || 1) * 0.08;
    const limit = Number.MAX_VALUE / unit;
    low = Math.max(-limit, low - padding);
    high = Math.min(limit, high + padding);
    if (floor !== null) low = Math.max(low, floor / unit);
    return {
        point: value => start + (value / unit - low) / (high - low) * (end - start),
        ticks: Array.from({length: 5}, (_, index) => (low + (high - low) * index / 4) * unit)
    };
}
let chartNumber = 0;
function plot(title, description, xValues, yValues, xLabel, yLabel, xTicks, xFloor = null) {
    const svg = svgElement("svg", {viewBox: "0 0 900 430", role: "img"});
    const titleId = `plot-title-${++chartNumber}`, descriptionId = `plot-description-${chartNumber}`;
    svg.setAttribute("aria-labelledby", `${titleId} ${descriptionId}`);
    svg.append(svgElement("title", {id: titleId}, title), svgElement("desc", {id: descriptionId}, description));
    const x = scale(xValues, 90, 840, xFloor), y = scale(yValues, 340, 35);
    const axisNumber = value => Number(value.toPrecision(3)).toString();
    for (const value of y.ticks) {
        const position = y.point(value);
        svg.append(svgElement("line", {x1: 90, y1: position, x2: 840, y2: position, stroke: "#e2e8ec"}));
        svg.append(svgElement("text", {x: 79, y: position + 4, "text-anchor": "end", fill: "#526574", "font-size": 12}, axisNumber(value)));
    }
    for (const value of xTicks || x.ticks) {
        const position = x.point(value);
        svg.append(svgElement("line", {x1: position, y1: 340, x2: position, y2: 346, stroke: "#526574"}));
        svg.append(svgElement("text", {x: position, y: 365, "text-anchor": "middle", fill: "#526574", "font-size": 12}, axisNumber(value)));
    }
    svg.append(svgElement("line", {x1: 90, y1: 340, x2: 840, y2: 340, stroke: "#526574"}));
    svg.append(svgElement("text", {x: 465, y: 406, "text-anchor": "middle", fill: "#18232b", "font-size": 13}, xLabel));
    svg.append(svgElement("text", {transform: "translate(21 188) rotate(-90)", "text-anchor": "middle", fill: "#18232b", "font-size": 13}, yLabel));
    return {svg, x: x.point, y: y.point};
}
function legend(panel, entries) {
    const element = text("div", "", "legend");
    for (const [label, color] of entries) {
        const item = text("span", "");
        const swatch = text("i", "", "swatch");
        swatch.style.backgroundColor = color;
        item.append(swatch, text("span", label));
        element.append(item);
    }
    panel.prepend(element);
}
function resourcePlot() {
    const systems = (analysis.systems || []).filter(system => Number.isFinite(system.mean) && Number.isFinite(system.resource_mean));
    if (!analysis.resource || !systems.length) return;
    const frontier = new Set(analysis.pareto_frontier || []);
    const xLabel = analysis.resource === "elapsed_seconds" ? "Mean seconds per attempt · lower is better" : `${analysis.resource} per attempt · lower is better`;
    const figure = plot("Quality and resource use", "Weighted system means on the same complete cohort. Colored points are the observed Pareto frontier.",
        systems.map(system => system.resource_mean), systems.map(system => system.mean),
        xLabel, `${analysis.metric} · ${directionLabel}`, undefined, 0);
    systems.forEach((system, index) => {
        const x = figure.x(system.resource_mean), y = figure.y(system.mean);
        const point = svgElement("circle", {cx: x, cy: y, r: frontier.has(system.index) ? 6 : 5,
            fill: frontier.has(system.index) ? "#096c75" : "#83929c", stroke: "white", "stroke-width": 1.5});
        point.append(svgElement("title", {}, `${systemName(system)}: ${analysis.metric} ${fmt(system.mean)}; ${analysis.resource} ${fmt(system.resource_mean)}`));
        figure.svg.append(point);
        // The full names remain in the table and point titles when a large panel is dense.
        if (systems.length <= 12) figure.svg.append(svgElement("text", {x: x + (x > 700 ? -10 : 10), y: y + (index % 2 ? 17 : -10),
            "text-anchor": x > 700 ? "end" : "start", fill: "#18232b", "font-size": 12}, systemName(system)));
    });
    const panel = resultPanel("Quality and resource use", "Frontier membership describes observed means: no other recorded system has both at least as good a score and no greater resource use, with one strict improvement. It does not establish statistical superiority. Intervals in the table are pointwise, not simultaneous.");
    panel.classList.add("chart");
    panel.append(figure.svg);
    legend(panel, [["Observed frontier", "#096c75"], ["Other systems", "#83929c"]]);
}
function interactionPlot() {
    const conditions = (analysis.conditions || []).filter(condition => Number.isFinite(condition.mean));
    if (conditions.length !== 4) return;
    const values = conditions.flatMap(condition => condition.interval ? [condition.mean, condition.interval.lower, condition.interval.upper] : [condition.mean]);
    const figure = plot("Factor interaction", "Two lines show the common-cohort mean at each level of the first factor, separately by the second factor.",
        [0, 1], values, `${factorA} level`, `${analysis.metric} · ${directionLabel}`, [0, 1]);
    const colors = ["#096c75", "#a35324"];
    for (const level of [0, 1]) {
        const line = conditions.filter(condition => condition.factor_b_level === level).sort((a, b) => a.factor_a_level - b.factor_a_level);
        figure.svg.append(svgElement("polyline", {points: line.map(condition => `${figure.x(condition.factor_a_level)},${figure.y(condition.mean)}`).join(" "),
            fill: "none", stroke: colors[level], "stroke-width": 2.5}));
        for (const condition of line) {
            const x = figure.x(condition.factor_a_level), y = figure.y(condition.mean);
            if (condition.interval) {
                const lower = figure.y(condition.interval.lower), upper = figure.y(condition.interval.upper);
                figure.svg.append(svgElement("path", {d: `M ${x} ${lower} V ${upper} M ${x - 5} ${lower} H ${x + 5} M ${x - 5} ${upper} H ${x + 5}`,
                    fill: "none", stroke: colors[level], "stroke-width": 1.5}));
            }
            const point = svgElement("circle", {cx: x, cy: y, r: 5, fill: colors[level], stroke: "white", "stroke-width": 1.5});
            point.append(svgElement("title", {}, `${factorA} = ${condition.factor_a_level}, ${factorB} = ${level}: ${fmt(condition.mean)}`));
            figure.svg.append(point);
        }
    }
    const note = "Lines connect weighted condition means on one complete cohort. " +
        (conditions.some(condition => condition.interval) ? "Error bars are marginal condition intervals. " : "") +
        `Interaction is the slope at ${factorB} = 1 minus the slope at ${factorB} = 0. Its uncertainty comes from the paired interaction contrast, not from overlap of condition intervals.`;
    const panel = resultPanel("Factor interaction", note);
    panel.classList.add("chart");
    panel.append(figure.svg);
    legend(panel, [[`${factorB} = 0`, colors[0]], [`${factorB} = 1`, colors[1]]]);
}

function studyRows(study) {
    const frontier = new Set(study.pareto_frontier || []);
    return (study.systems || []).map(system => ({
        system: systemName(system), baseline: system.index === analysis.baseline?.index ? "yes" : "",
        mean: system.mean, interval: intervalText(system.interval, system.interval_reason),
        difference: system.delta, difference_interval: intervalText(system.delta_interval, system.paired_interval_reason),
        improvement: system.improvement, improvement_interval: intervalText(system.improvement_interval, system.paired_interval_reason),
        resource: system.resource_mean, resource_interval: intervalText(system.resource_interval, system.resource_interval_reason),
        frontier: system.mean == null || system.resource_mean == null ? "—" : frontier.has(system.index) ? "yes" : "no"
    }));
}
const studyColumns = cols([["system", "System · revision"], ["baseline", "Baseline"], ["mean", "Mean score"],
    ["interval", "Score interval"], ["difference", "Difference vs baseline"], ["difference_interval", "Difference interval"],
    ["improvement", "Direction-adjusted improvement"], ["improvement_interval", "Improvement interval"]]);
if (analysis.resource) studyColumns.push(...cols([["resource", analysis.resource === "elapsed_seconds" ? "Seconds / attempt" : `${analysis.resource} / attempt`],
    ["resource_interval", "Resource interval"], ["frontier", "Observed frontier"]]));
function effectRows(result) {
    return Object.entries(result.effects || {}).map(([name, effect]) => ({
        contrast: effectNames[name] || words(name), estimate: effect.estimate,
        interval: intervalText(effect.interval, effect.interval_reason), improvement: effect.improvement,
        improvement_interval: intervalText(effect.improvement_interval, effect.interval_reason)
    }));
}
const effectColumns = cols([["contrast", "Contrast"], ["estimate", "Raw score contrast"], ["interval", "Contrast interval"],
    ["improvement", "Direction-adjusted contrast"], ["improvement_interval", "Direction-adjusted interval"]]);
const interpretations = {
    sample_weighted_mean_of_observed_trial_means: "Trial scores are averaged within each sample, then sample means are averaged using their declared weights. Incomplete samples contribute their observed scores; coverage describes what remains unscored.",
    sample_weighted_mean_of_complete_paired_trial_mean_differences: "Candidate minus baseline on complete paired samples, using declared sample weights. The comparison keeps the same cases and independent groups together. Excluded samples change the population represented by the estimate.",
    sample_weighted_probability_of_success_in_k_attempts: "Estimated probability of at least one success within k attempts, averaged with declared sample weights. This interpretation assumes independent attempts from the same sampling policy.",
    sample_weighted_trial_means_on_common_complete_cases: "Every system and baseline contrast uses the same complete samples, with attempts averaged within sample before sample weighting. When resource use is selected, this cohort also requires complete resource observations. Intervals resample paired independent groups and are pointwise, not simultaneous across systems or slices.",
    sample_weighted_contrasts_of_complete_four_condition_trial_means: "Four condition means use one complete sample cohort. Main effects average a factor's contrast equally over the other factor's two levels. Interaction is the full difference of differences: 11 − 10 − 01 + 00. Intervals are marginal, not simultaneous across contrasts or slices. A contrast identifies a causal effect only when the experimental design supports that interpretation.",
    item_behavior_conditional_on_the_reference_panel_and_common_complete_cohort: "Item behavior is conditional on this reference panel and the common complete cohort. Each distinct system has equal panel weight; overall summaries use the declared item weights. Facility is the panel success fraction for binary higher-is-better scores with a supported [0, 1] scale. Discrimination is a Pearson correlation across systems between an item's score and their weighted rest scores, excluding the item's entire independent group. These are descriptive properties of this panel.",
    equal_case_exact_agreement_conditional_on_paired_labels: "Exact categorical agreement conditions on cases with both judge and reference labels. Each paired case has equal weight; intervals resample independent groups. Kappa adjusts for the observed label marginals. Agreement with a reference is not a measurement of objective correctness."
};
$("method").append(text("strong", "Interpretation"), text("p", interpretations[analysis.estimand] || analysis.estimand || "The recorded metric is analysed on its stated sample population."));
if (analysis.interval) $("method").append(text("p", "Interval: " + [analysis.interval.method,
    analysis.interval.independent_groups !== undefined && `${analysis.interval.independent_groups} independent groups`,
    analysis.interval.resamples !== undefined && `${analysis.interval.resamples} resamples`].filter(Boolean).join("; "), "note"));
else if (analysis.interval_reason) $("method").append(text("p", `Interval unavailable: ${words(analysis.interval_reason)}.`, "note"));
if (coverage.retained_weight_fraction != null) $("method").append(text("p", `${fmt(100 * coverage.retained_weight_fraction)}% of planned sample weight retained. Missing-data policy: ${analysis.missing || "recorded population"}.`, "note"));

if (kind === "study") {
    card("Systems", analysis.systems?.length);
    card("Complete common samples", coverage.retained_samples);
    card("Independent retained groups", coverage.retained_groups);
    card("Excluded samples", coverage.excluded_samples);
    resultPanel("Systems", `All contrasts use ${analysis.baseline?.label || "the declared baseline"} as the baseline. Positive direction-adjusted improvement favors the candidate.`)
        .append(table(studyRows(analysis), studyColumns, {label: "System estimates and baseline contrasts"}));
    resourcePlot();
} else if (kind === "factorial") {
    for (const name of ["main_a", "main_b", "interaction"]) card(effectNames[name] + " · raw", analysis.effects?.[name]?.estimate);
    card("Complete common samples", coverage.retained_samples);
    card("Independent retained groups", coverage.retained_groups);
    card("Excluded samples", coverage.excluded_samples);
    const conditions = (analysis.conditions || []).map(condition => ({
        condition: condition.cell, factor_a: condition.factor_a_level, factor_b: condition.factor_b_level,
        system: systemName(condition), mean: condition.mean, interval: intervalText(condition.interval, condition.interval_reason)
    }));
    resultPanel("Conditions").append(table(conditions, cols([["condition", "Cell"], ["factor_a", factorA], ["factor_b", factorB],
        ["system", "System · revision"], ["mean", "Mean score"], ["interval", "Score interval"]]), {label: "Four factorial conditions"}));
    interactionPlot();
    resultPanel("Factor contrasts", "Interaction uses the full difference of differences. Raw contrasts retain the score scale; direction-adjusted contrasts reverse sign for a lower-is-better metric.")
        .append(table(effectRows(analysis), effectColumns, {label: "Factorial contrasts"}));
} else if (kind === "benchmark_analysis") {
    const summary = analysis.summary || {};
    card("Reference systems", analysis.panel?.length);
    card("Complete panel items", coverage.retained_samples);
    card("Constant across panel", summary.constant_samples);
    card("Discrimination defined", summary.discrimination_defined);
    const panelRows = (analysis.panel || []).map(system => ({system: systemName(system), mean: system.mean}));
    resultPanel("Reference panel", "Means use the common item cohort with declared item weights. Panel members are treated as the chosen reference systems, not independent draws from all possible systems.")
        .append(table(panelRows, cols([["system", "System · revision"], ["mean", "Mean score"]]), {label: "Benchmark reference panel"}));
    resultPanel("Item profile").append(table([summary], cols([["panel_mean", "Weighted panel mean"], ["mean_item_range", "Mean item range"],
        ["mean_item_stddev", "Mean item SD"], ["floor_samples", "At performance floor"], ["ceiling_samples", "At performance ceiling"],
        ["negative_discrimination_samples", "Negative discrimination"]]), {label: "Benchmark item profile"}));
    $("sample-heading").textContent = "Items: facility and discrimination";
} else if (kind === "judge_calibration") {
    card("Exact agreement", analysis.agreement);
    card("Cohen's kappa", analysis.kappa);
    card("Paired cases", coverage.paired_samples);
    card("Total cases", coverage.total_samples);
    if (analysis.confusion) {
        const labels = analysis.confusion.labels || [];
        const confusionRows = (analysis.confusion.rows || []).map((row, index) => Object.fromEntries([
            ["reference", labels[index]], ...row.map((value, column) => [`label_${column}`, value])
        ]));
        resultPanel("Label agreement", "Rows are reference labels; columns are judge labels.").append(table(confusionRows,
            [{key: "reference", label: "Reference / judge"}, ...labels.map((label, index) => ({key: `label_${index}`, label}))], {label: "Judge confusion matrix"}));
    }
    if (analysis.per_class?.length) resultPanel("Per-label agreement").append(table(analysis.per_class,
        cols(["label", "reference_count", "judge_count", "matches", "precision", "recall", "f1"]), {label: "Per-label judge agreement"}));
} else {
    if (comparison) {
        card("Baseline mean", analysis.left_mean); card("Candidate mean", analysis.right_mean);
        card("Score difference", analysis.delta); card("Direction-adjusted improvement", analysis.improvement);
        card("Complete paired samples", analysis.paired_samples ?? coverage.retained_samples);
        if (coverage.excluded_samples) card("Excluded samples", coverage.excluded_samples);
    } else card(analysis.k != null ? `Success within ${analysis.k} attempts` : "Weighted sample mean", analysis.mean ?? analysis.estimate);
    for (const key of ["scored_samples", "total_samples", "scored_trials", "planned_trials"]) if (has(coverage, key)) card(words(key), coverage[key]);
}
if (analysis.interval) card(`${fmt(100 * analysis.interval.confidence)}% interval`, `${fmt(analysis.interval.lower)} to ${fmt(analysis.interval.upper)}`);

if (Array.isArray(analysis.slices) && analysis.slices.length) {
    $("slice-section").hidden = false;
    $("slice-heading").textContent = analysis.slice_by ? `Slices · ${analysis.slice_by}` : "Slices";
    let sliceRows = [], sliceColumns;
    const sliceLabel = slice => slice.missing ? "(missing label)" : slice.value === null ? "(null)" : slice.value;
    const retained = slice => ({retained: slice.coverage?.retained_samples, excluded: slice.coverage?.excluded_samples,
        groups: slice.coverage?.retained_groups});
    const coverageColumns = cols([["retained", "Retained samples"], ["excluded", "Excluded samples"], ["groups", "Retained groups"]]);
    if (kind === "study") {
        sliceRows = analysis.slices.flatMap(slice => studyRows(slice).map(row => ({slice: sliceLabel(slice), ...row, ...retained(slice)})));
        sliceColumns = [...studyColumns, ...coverageColumns];
    } else if (kind === "factorial") {
        sliceRows = analysis.slices.flatMap(slice => effectRows(slice).map(row => ({slice: sliceLabel(slice), ...row, ...retained(slice)})));
        sliceColumns = [...effectColumns, ...coverageColumns];
    } else if (kind === "benchmark_analysis") {
        sliceRows = analysis.slices.map(slice => ({...slice, slice: sliceLabel(slice), ...retained(slice)}));
        sliceColumns = cols([["panel_mean", "Panel mean"], ["mean_item_range", "Mean item range"], ["constant_samples", "Constant items"],
            ["floor_samples", "Floor items"], ["ceiling_samples", "Ceiling items"], ["discrimination_defined", "Discrimination defined"],
            ["negative_discrimination_samples", "Negative discrimination"]]).concat(coverageColumns);
    } else {
        sliceRows = analysis.slices.map(slice => ({slice: sliceLabel(slice), ...(comparison ? {
            left_mean: slice.left_mean, right_mean: slice.right_mean, delta: slice.delta, improvement: slice.improvement,
            paired_samples: slice.paired_samples, excluded_samples: slice.coverage?.excluded_samples
        } : {mean: slice.mean, scored_samples: slice.coverage?.scored_samples, total_samples: slice.coverage?.total_samples}),
            interval: intervalText(slice.interval, slice.interval_reason)}));
        sliceColumns = cols(comparison ? ["left_mean", "right_mean", "delta", "improvement", "paired_samples", "excluded_samples", "interval"]
            : ["mean", "scored_samples", "total_samples", "interval"]);
    }
    $("slices").append(table(sliceRows, [{key: "slice", label: "Slice"}, ...sliceColumns], {label: "Slice estimates and coverage"}));
}

// Path keys preserve nested fields and arrays without conflating literal dots in names.
function flatten(row) {
    const flat = Object.create(null);
    function visit(value, path) {
        if (value !== null && typeof value === "object") {
            for (const [key, child] of Object.entries(value)) visit(child, [...path, key]);
        } else flat[JSON.stringify(path)] = value;
    }
    for (const [key, value] of Object.entries(row)) {
        if (kind === "study" && key === "resources" && !analysis.resource) continue;
        visit(value, [key]);
    }
    return flat;
}
const panelSystems = analysis.systems || analysis.panel || [];
function sampleColumnLabel(key) {
    const path = JSON.parse(key);
    if (["means", "system_means", "rest_system_means", "resources"].includes(path[0]) && path.length === 2) {
        const system = panelSystems[Number(path[1])];
        const label = path[0] === "resources" ? "Resource" : path[0] === "rest_system_means" ? "Rest score" : "Score";
        return `${label} · ${system ? systemName(system) : `system ${Number(path[1]) + 1}`}`;
    }
    if (path[0] === "contrasts" && path.length === 2) return effectNames[path[1]] || words(path[1]);
    if (path[0] === "scores" && path.length === 2) return `Score · ${path[1]}`;
    return path.map(words).join(" · ");
}
const preferred = ["sample_id", "group", "weight", "facility", "discrimination", "discrimination_reason", "panel_mean",
    "panel_min", "panel_max", "panel_range", "panel_stddev", "floor", "ceiling", "left_mean", "right_mean", "left", "right", "mean",
    "delta", "improvement", "estimate", "reference", "judge", "agreement", "status", "planned_trials", "observed_trials", "scored_trials",
    "complete", "trials", "n", "successes"].map(key => JSON.stringify([key]));
const rows = (Array.isArray(analysis.samples) ? analysis.samples : []).map(flatten);
const keys = new Set(rows.flatMap(Object.keys));
const sampleColumns = [...preferred.filter(key => keys.has(key)), ...Array.from(keys).filter(key => !preferred.includes(key))]
    .map(key => ({key, label: sampleColumnLabel(key)}));
let page = 0, sortKey = null, ascending = true;
function renderSamples() {
    const query = $("filter").value.toLowerCase();
    const filtered = rows.filter(row => Object.values(row).some(value => fmt(value).toLowerCase().includes(query)));
    if (sortKey) filtered.sort((a, b) => {
        const left = a[sortKey], right = b[sortKey];
        if (left == null || right == null) return left == null ? (right == null ? 0 : 1) : -1;
        const order = typeof left === "number" && typeof right === "number" ? left - right : String(left).localeCompare(String(right));
        return ascending ? order : -order;
    });
    const size = Number($("page-size").value), pages = Math.max(1, Math.ceil(filtered.length / size));
    page = Math.max(0, Math.min(page, pages - 1));
    $("sample-table").replaceChildren(filtered.length ? table(filtered.slice(page * size, (page + 1) * size), sampleColumns, {
        label: kind === "benchmark_analysis" ? "Item scores, facility and discrimination" : "Per-sample results",
        sortKey, ascending, onSort: key => {ascending = key === sortKey ? !ascending : true; sortKey = key; renderSamples();}
    }) : text("p", "No matching samples.", "muted"));
    $("page-label").textContent = `${filtered.length} matching samples · page ${page + 1} of ${pages}`;
    $("previous").disabled = page === 0; $("next").disabled = page === pages - 1;
}
$("filter").addEventListener("input", () => {page = 0; renderSamples();});
$("page-size").addEventListener("change", () => {page = 0; renderSamples();});
$("previous").addEventListener("click", () => {page--; renderSamples();});
$("next").addEventListener("click", () => {page++; renderSamples();});
$("download").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([analysisText + "\n"], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url; link.download = "evaluation-analysis.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const details = {...analysis}; delete details.samples;
$("details").textContent = JSON.stringify(details, null, 2);
renderSamples();
</script></body></html>'''


def load_analysis(path: str | Path) -> dict:
    """Read a JSON analysis object, preserving the result contract's JSON rules."""
    path = Path(path)
    analysis = _parse(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object")
    return analysis


def write_report(analysis: dict, output: str | Path, *, title: str | None = None) -> dict:
    """Write a new portable HTML report containing the complete analysis JSON."""
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be an object")
    titles = {"study": "Multi-system study", "factorial": "2 × 2 factorial comparison",
              "benchmark_analysis": "Benchmark item analysis", "judge_calibration": "Judge calibration"}
    fallback = "Paired system comparison" if "delta" in analysis else "Evaluation results"
    if "k" in analysis:
        fallback = f"Pass@{analysis['k']} evaluation"
    title = title or titles.get(analysis.get("kind"), fallback)
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
