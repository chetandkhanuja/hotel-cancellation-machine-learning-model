"""Self-contained interactive SVG components used by the Streamlit labs."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True).replace("</", "<\\/")


def render_tree(
    nodes: list[dict[str, Any]],
    visible_depth: int,
    path: list[int] | None = None,
    height: int = 790,
    title: str = "Fitted decision tree",
) -> None:
    """Render a progressively revealed tree from actual fitted sklearn nodes."""
    data = _json_for_script(nodes)
    path_data = _json_for_script(path or [])
    page_title = json.dumps(title)
    html = r"""
<!doctype html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#071821;color:#e8f2f1;font-family:Inter,"Segoe UI",sans-serif}
.frame{border:1px solid #24414a;border-radius:18px;background:linear-gradient(145deg,#0b2029,#08161e);padding:16px;box-shadow:0 18px 48px #02090d66}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.title{font-size:15px;font-weight:700;letter-spacing:.02em}.sub{color:#8eaaad;font-size:12px;margin-top:4px}
.controls{display:flex;gap:7px;align-items:center;flex-wrap:wrap}
button{border:1px solid #31525a;background:#102832;color:#d9ece9;padding:7px 11px;border-radius:9px;font-weight:600;cursor:pointer}
button:hover{background:#153b40;border-color:#32d39a}button.primary{background:#08775c;border-color:#16a878;color:#fff}
.speed{display:flex;align-items:center;gap:6px;color:#9eb8b8;font-size:11px}.speed input{width:80px;accent-color:#21c88a}
.viewport{overflow:auto;margin-top:12px;border:1px solid #18333b;border-radius:13px;background:radial-gradient(ellipse at 50% 0%,#14313a 0%,#0a1c24 72%);min-height:290px}
svg{display:block}.edge{fill:none;stroke:#31515a;stroke-width:2;opacity:0;stroke-dasharray:500;stroke-dashoffset:500}
.edge.shown{opacity:.9;stroke-dashoffset:0;transition:stroke-dashoffset .8s ease,opacity .45s}
.edge.path{stroke:#3ce2a5;stroke-width:3;filter:drop-shadow(0 0 4px #21c88a)}
.edge-label{fill:#78989b;font-size:10px;text-anchor:middle;opacity:0}.edge-label.shown{opacity:1;transition:opacity .4s}
.node{opacity:0;cursor:pointer}
.node.shown{opacity:1;transition:opacity .38s ease}
.node rect{fill:#102932;stroke:#41717a;stroke-width:1.5;filter:drop-shadow(0 7px 9px #02090d88)}
.node.shown rect{transform-box:fill-box;transform-origin:center;animation:node-pop .38s cubic-bezier(.2,.8,.2,1)}
@keyframes node-pop{from{transform:scale(.88);opacity:.6}to{transform:scale(1);opacity:1}}
.node.high rect{fill:#10352e;stroke:#27bd86}.node.leaf rect{fill:#12332c;stroke:#43d6a0}
.node.selected rect{stroke:#7df2c5;stroke-width:3}.node.in-path rect{stroke:#5cf0b7;stroke-width:2.5;filter:drop-shadow(0 0 9px #20c88a)}
.node-title{fill:#e9f4f1;font-size:11px;font-weight:700;text-anchor:middle}
.node-stat{fill:#8fadaf;font-size:10px;text-anchor:middle}.node-badge{fill:#6fe5b2;font-size:9px;font-weight:700;text-anchor:middle}
.info{display:grid;grid-template-columns:1.15fr .85fr;gap:12px;margin-top:12px}
.panel{border:1px solid #1f3b43;border-radius:12px;background:#0a1b23;padding:13px 15px;min-height:108px}
.panel h3{margin:0 0 8px;font-size:13px;color:#dff4ee}.muted{color:#92acad;font-size:12px;line-height:1.5}
.facts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.fact{background:#0e252d;border-radius:8px;padding:8px}.fact small{display:block;color:#86a3a5;font-size:10px}.fact b{font-size:13px;color:#e6f4f0}
.legend{font-size:10px;color:#88a5a7;display:flex;gap:14px;margin-top:9px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#20bd85;margin-right:5px}
@media(max-width:650px){.info{grid-template-columns:1fr}.controls{justify-content:flex-start}.frame{padding:10px}}
</style></head>
<body><main class="frame">
  <div class="top">
    <div><div class="title" id="heading"></div><div class="sub">Actual fitted splits · click any visible node to inspect its statistics</div></div>
    <div class="controls">
      <button class="primary" id="run">▶ Run Tree</button><button id="pause">Ⅱ Pause</button>
      <button id="continue">▶ Continue</button><button id="restart">↺ Restart</button><button id="all">Reveal All</button>
      <label class="speed">Speed <input id="speed" type="range" min="180" max="1100" value="580"></label>
      <button id="trace">Trace booking</button>
    </div>
  </div>
  <div class="viewport"><svg id="tree"></svg></div>
  <div class="info">
    <section class="panel" id="inspector"><h3>Node inspector</h3><div class="muted">Select a node after it appears to inspect its real split, impurity, samples and cancellation distribution.</div></section>
    <section class="panel"><h3>Booking journey</h3><div class="muted" id="journey">The highlighted route follows the booking through the fitted tree.</div></section>
  </div>
  <div class="legend"><span><i class="dot"></i>Cancellation probability at the node</span><span>Visible depth: __DEPTH__</span></div>
</main>
<script>
const nodes=__DATA__, maxDepth=__DEPTH__, route=__PATH__, heading=__TITLE__;
document.getElementById("heading").textContent=heading;
const map=new Map(nodes.map(n=>[n.id,n])), svg=document.getElementById("tree");
const ns="http://www.w3.org/2000/svg", nodeW=168,nodeH=78, gapY=116;
function el(tag,attrs={},text){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e}
function terminalCount(id){const n=map.get(id);if(!n||n.leaf||n.depth>=maxDepth)return 1;return terminalCount(n.left)+terminalCount(n.right)}
const leaves=terminalCount(0), width=Math.max(850,leaves*190+40), height=(maxDepth+1)*gapY+50;
svg.setAttribute("width",width);svg.setAttribute("height",height);svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
const coords=new Map(),edges=[];
function place(id,x0,span){const n=map.get(id);if(!n)return;const x=x0+span/2,y=55+n.depth*gapY;coords.set(id,{x,y});
 if(!n.leaf&&n.depth<maxDepth){const l=terminalCount(n.left),r=terminalCount(n.right),unit=span/(l+r);place(n.left,x0,unit*l);place(n.right,x0+unit*l,unit*r);edges.push({from:id,to:n.left,side:"left"});edges.push({from:id,to:n.right,side:"right"})}}
place(0,20,width-40);
const edgeLayer=el("g"),labelLayer=el("g"),nodeLayer=el("g");svg.append(edgeLayer,labelLayer,nodeLayer);
for(const edge of edges){const a=coords.get(edge.from),b=coords.get(edge.to),n=map.get(edge.from),sx=a.x,sy=a.y+nodeH/2-2,ex=b.x,ey=b.y-nodeH/2+2,mid=(sy+ey)/2;
 const p=el("path",{d:`M ${sx} ${sy} C ${sx} ${mid}, ${ex} ${mid}, ${ex} ${ey}`,class:"edge","data-from":edge.from,"data-to":edge.to});edgeLayer.appendChild(p);
 const branch=n.feature_kind==="categorical"?(edge.side==="left"?"no":"yes"):(edge.side==="left"?"≤":" >");
 labelLayer.appendChild(el("text",{x:(sx+ex)/2,y:mid-3,class:"edge-label","data-from":edge.from,"data-to":edge.to},branch))}
function wrap(text,max=24){const words=String(text).split(/\s+/),lines=[];let line="";for(const w of words){if((line+" "+w).trim().length>max&&line){lines.push(line);line=w}else line=(line+" "+w).trim()}if(line)lines.push(line);return lines.slice(0,2)}
for(const n of nodes){const p=coords.get(n.id);if(!p)continue;const left=p.x-nodeW/2,top=p.y-nodeH/2,group=el("g",{class:`node${n.leaf?" leaf":""}${n.cancel_probability>=.5?" high":""}`,"data-id":n.id});
 group.appendChild(el("rect",{x:left,y:top,width:nodeW,height:nodeH,rx:13}));
 const title=n.leaf?(n.prediction===1?"Predicts cancellation":"Predicts no cancellation"):n.condition;
 const lines=wrap(title,25);lines.forEach((line,i)=>group.appendChild(el("text",{x:p.x,y:top+22+i*14,class:"node-title"},line)));
 group.appendChild(el("text",{x:p.x,y:top+57,class:"node-stat"},`${n.samples.toLocaleString()} bookings · ${Math.round(n.cancel_probability*100)}% cancel`));
 if(n.leaf)group.appendChild(el("text",{x:p.x,y:top+71,class:"node-badge"},`LEAF · ${n.prediction===1?"CANCEL":"KEEP"}`));
 else if(n.depth===maxDepth)group.appendChild(el("text",{x:p.x,y:top+71,class:"node-badge"},"SPLIT · MORE BELOW"));
 group.addEventListener("click",()=>inspect(n,group));nodeLayer.appendChild(group)}
const nodeEls=[...document.querySelectorAll(".node")],edgeEls=[...document.querySelectorAll(".edge")],labelEls=[...document.querySelectorAll(".edge-label")];
let revealed=-1,timer=null,chosen=null;
function reveal(level){for(const node of nodeEls){const d=map.get(Number(node.dataset.id)).depth;if(d<=level)node.classList.add("shown")}
 for(const edge of edgeEls){const d=map.get(Number(edge.dataset.from)).depth;if(d<level)edge.classList.add("shown")}
 for(const label of labelEls){const d=map.get(Number(label.dataset.from)).depth;if(d<level)label.classList.add("shown")}
 revealed=Math.max(revealed,level)}
function clearTimer(){if(timer){clearTimeout(timer);timer=null}}
function play(){clearTimer();if(revealed>=maxDepth)return;timer=setTimeout(()=>{reveal(revealed+1);play()},Number(document.getElementById("speed").value))}
function restart(){clearTimer();revealed=-1;chosen=null;nodeEls.forEach(x=>x.classList.remove("shown","selected","in-path"));edgeEls.forEach(x=>x.classList.remove("shown","path"));labelEls.forEach(x=>x.classList.remove("shown"));document.getElementById("journey").textContent="The highlighted route follows the booking through the fitted tree."}
function inspect(n,group){nodeEls.forEach(x=>x.classList.remove("selected"));group.classList.add("selected");chosen=n.id;
 const classes=n.class_labels.map((label,i)=>`${label==="1"?"Cancelled":"Not cancelled"}: ${n.counts[i].toLocaleString()} (${(n.counts[i]/Math.max(1,n.samples)*100).toFixed(1)}%)`).join(" · ");
 const childStats=id=>{const child=map.get(id);return child?`${child.samples.toLocaleString()} samples · ${(child.cancel_probability*100).toFixed(1)}% cancel`:"below visible depth"};
 const rows=n.leaf?`<div class="muted">${classes}</div><div class="muted" style="margin-top:7px">Final decision at ${Math.round(n.classification_threshold*100)}% cutoff: <b>${n.prediction===1?"cancelled":"not cancelled"}</b> · cancellation probability ${(n.cancel_probability*100).toFixed(1)}%</div>`:
 `<div class="muted">Split test: <b>${n.condition}</b></div><div class="muted" style="margin:5px 0">Left child · node ${n.left}: ${childStats(n.left)}<br>Right child · node ${n.right}: ${childStats(n.right)}</div><div class="muted">${classes}</div>`;
 document.getElementById("inspector").innerHTML=`<h3>${n.leaf?"Leaf node":"Decision node"} ${n.id}</h3><div class="facts"><div class="fact"><small>Gini impurity</small><b>${n.gini.toFixed(4)}</b></div><div class="fact"><small>Samples here</small><b>${n.samples.toLocaleString()}</b></div><div class="fact"><small>Cancellation rate</small><b>${(n.cancel_probability*100).toFixed(1)}%</b></div></div><div style="margin-top:9px">${rows}</div>`}
function trace(){if(!route.length){return}clearTimer();route.forEach((id,i)=>{const n=map.get(id);if(n)reveal(n.depth)});nodeEls.forEach(x=>x.classList.remove("in-path"));edgeEls.forEach(x=>x.classList.remove("path"));
 let i=0;const visible=route.filter(id=>map.has(id));document.getElementById("journey").textContent=`Following ${route.length-1} actual split(s) to leaf ${route[route.length-1]}. The complete condition-by-condition path is listed below the visualization.`;
 function step(){if(i>=visible.length)return;const id=visible[i];const node=document.querySelector(`.node[data-id="${id}"]`);if(node)node.classList.add("in-path");if(i>0){const prev=visible[i-1],edge=document.querySelector(`.edge[data-from="${prev}"][data-to="${id}"]`);if(edge)edge.classList.add("path")}i++;if(i<visible.length)timer=setTimeout(step,Number(document.getElementById("speed").value))}step()}
document.getElementById("run").onclick=()=>{restart();play()};document.getElementById("pause").onclick=clearTimer;document.getElementById("continue").onclick=play;
document.getElementById("restart").onclick=restart;document.getElementById("all").onclick=()=>{clearTimer();reveal(maxDepth)};
document.getElementById("trace").onclick=trace;
setTimeout(()=>{document.querySelector(".viewport").scrollLeft=Math.max(0,(width-document.querySelector(".viewport").clientWidth)/2);play()},350);
</script></body></html>
"""
    html = (
        html.replace("__DATA__", data)
        .replace("__PATH__", path_data)
        .replace("__DEPTH__", str(visible_depth))
        .replace("__TITLE__", page_title)
    )
    st.iframe(
        html,
        height=height,
        alt="Interactive fitted decision tree",
    )


def render_forest(cards: list[dict[str, Any]], height: int = 335) -> None:
    """Animate cards representing actual trees and their individual outputs."""
    data = _json_for_script(cards)
    html = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;background:#071821;color:#eaf3f0;font-family:Inter,"Segoe UI",sans-serif}
.wrap{border:1px solid #23424a;border-radius:17px;padding:15px;background:linear-gradient(145deg,#0b2029,#08161e);box-shadow:0 18px 46px #02090d66}
.head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}.title{font-weight:700;font-size:14px}.caption{font-size:11px;color:#8da9aa;margin-top:4px}
.controls{display:flex;gap:6px}.controls button{border:1px solid #31535a;background:#102932;color:#d9ece9;padding:7px 10px;border-radius:8px;cursor:pointer;font-weight:600}.controls button:hover{border-color:#39d99d}.controls .primary{background:#08775c;color:#fff}
.forest{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:9px;margin-top:14px}
.card{opacity:0;transform:translateY(8px) scale(.97);transition:opacity .45s,transform .45s;background:#0e252d;border:1px solid #24444c;border-radius:12px;padding:11px;min-height:135px}
.card.shown{opacity:1;transform:translateY(0) scale(1)}.card.cancel{border-color:#b17756}.treeid{color:#6edeb0;font-size:10px;font-weight:700;letter-spacing:.08em}.split{font-size:12px;line-height:1.35;min-height:33px;margin:8px 0;color:#e5efec}
.meter{height:5px;background:#1a3b40;border-radius:5px;overflow:hidden}.meter i{height:100%;display:block;background:linear-gradient(90deg,#18a979,#64e3b2);transition:width .5s}
.prob{display:flex;justify-content:space-between;color:#91acac;font-size:10px;margin-top:7px}.pred{font-size:11px;color:#e5efec;margin-top:4px}
.aggregate{border-top:1px solid #1f3b43;margin-top:13px;padding-top:10px;display:flex;align-items:center;justify-content:space-between;gap:12px}.agglabel{font-size:11px;color:#96acad}.aggvalue{font-size:20px;font-weight:800;color:#6ce4b1}.note{font-size:10px;color:#7f9a9d}
@media(max-width:650px){.forest{grid-template-columns:repeat(2,minmax(120px,1fr))}}
</style></head><body><main class="wrap"><div class="head"><div><div class="title">An ensemble of fitted trees</div><div class="caption">Eight actual members shown · every tree in the trained forest contributes equally to the final probability.</div></div><div class="controls"><button class="primary" id="run">▶ Run Forest</button><button id="pause">Ⅱ Pause</button><button id="continue">▶ Continue</button><button id="restart">↺ Restart</button><button id="all">Reveal All</button></div></div><div class="forest" id="forest"></div><div class="aggregate"><div><div class="agglabel">Mean cancellation probability · all __TOTAL__ trees</div><div class="note">Displayed member probabilities are a sample; the value at right uses the complete forest.</div></div><div class="aggvalue">__MEAN__%</div></div></main>
<script>
const cards=__DATA__,container=document.getElementById("forest");let shown=0,timer=null;
for(const c of cards){const card=document.createElement("section");card.className="card"+(c.probability>=.5?" cancel":"");card.innerHTML=`<div class="treeid">TREE ${String(c.index+1).padStart(3,"0")}</div><div class="split">${c.root}</div><div class="meter"><i style="width:${Math.round(c.probability*100)}%"></i></div><div class="prob"><span>${(c.probability*100).toFixed(1)}% cancel probability</span><span>${c.samples.toLocaleString()} root samples</span></div><div class="pred">At the current cutoff: <b>${c.probability>=c.threshold?"CANCEL":"KEEP"}</b></div>`;container.appendChild(card)}
const els=[...document.querySelectorAll(".card")];function clear(){if(timer){clearTimeout(timer);timer=null}}
function reveal(){if(shown>=els.length){clear();return}els[shown].classList.add("shown");shown++;timer=setTimeout(reveal,500)}
function run(){clear();els.forEach(e=>e.classList.remove("shown"));shown=0;reveal()}
document.getElementById("run").onclick=run;document.getElementById("pause").onclick=clear;document.getElementById("continue").onclick=()=>{clear();reveal()};
document.getElementById("restart").onclick=()=>{clear();shown=0;els.forEach(e=>e.classList.remove("shown"))};
document.getElementById("all").onclick=()=>{clear();shown=els.length;els.forEach(e=>e.classList.add("shown"))};
setTimeout(run,250);
</script></body></html>
"""
    mean_probability = cards[0].get("forest_probability", 0.0) if cards else 0.0
    html = (
        html.replace("__DATA__", data)
        .replace("__MEAN__", f"{mean_probability * 100:.1f}")
        .replace("__TOTAL__", str(cards[0].get("forest_size", 0) if cards else 0))
    )
    st.iframe(
        html,
        height=height,
        alt="Animated random forest tree predictions",
    )
