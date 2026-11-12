/* Execute page scripts against a small DOM harness and the real local API.
 * This checks rendering/control logic; it is not a browser visual-layout test.
 * Usage: node tests/ui_smoke.cjs http://127.0.0.1:8080
 */
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const base = process.argv[2] || "http://127.0.0.1:8080";
class Element {
  constructor(tag="div") { this.tag=tag; this.children=[]; this.dataset={}; this.attrs={}; this.events={}; this.value=""; this._text=""; }
  set textContent(value) { this._text=String(value); this.children=[]; }
  get textContent() { return this._text+this.children.map(c=>typeof c==="string"?c:c.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text=""; this.children=[...children]; }
  setAttribute(key,value) { this.attrs[key]=String(value); }
  addEventListener(name,handler) { this.events[name]=handler; }
  click() { return this.events.click?.(); }
}
async function page(name) {
  const html=fs.readFileSync(`connect4/static/${name}.html`,"utf8");
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  const document={getElementById:id=>{assert(elements.has(id),`Unknown element ${id}`);return elements.get(id);},createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),hidden:false};
  const outstanding=[];
  const errors=[];
  let playState={mode:"simulation",status:"starting",updated_at:1};
  const calls=[];
  const sandbox={document,console,Date,JSON,Number,Math,Array,Object,String,Boolean,URL,Blob,
    fetch:(url,options)=>{
      if(name==="play" && !url.startsWith("/debug/")) {
        calls.push({url,options});
        if(url==="/mode"||url==="/human/new") playState={mode:"human",status:"human_turn",updated_at:playState.updated_at+1,game_id:"ui-test",model_two:"shared",board:Array.from({length:7},()=>Array(5).fill(0)),move_count:0,next_player:1,done:false,last_column:null};
        if(url==="/human/move") {
          const payload=JSON.parse(options.body);playState.board[payload.column][4]=1;playState.board[6][4]=-1;
          Object.assign(playState,{move_count:2,updated_at:playState.updated_at+1,last_column:6,last_player:-1});
        }
        return Promise.resolve({ok:true,json:async()=>JSON.parse(JSON.stringify(playState))});
      }
      const result=fetch(new URL(url,base),options);outstanding.push(result);return result;
    },setTimeout:()=>0};
  const context=vm.createContext(sandbox);
  const script=name==="explain"?"explain-simple":name==="explain-technical"?"explain":name;
  vm.runInContext(fs.readFileSync(`connect4/static/${script}.js`,"utf8"),context,{filename:`${script}.js`});
  for(let i=0;i<100;i++) {
    await new Promise(resolve=>setTimeout(resolve,30));
    if(name==="debug" && elements.get("freshness").textContent.startsWith("Snapshot")) break;
    if(name==="explain" && elements.get("live-model").textContent.startsWith("Live:")) break;
    if(name==="explain-technical" && elements.get("live-model").textContent.startsWith("Live:")) break;
    if(name==="play" && elements.get("connection").textContent==="Game connected") break;
  }
  return {elements,context,calls};
}
(async()=>{
  const debug=await page("debug"), d=debug.elements;
  assert(d.get("freshness").textContent.startsWith("Snapshot"),d.get("alert").textContent);
  assert(d.get("generation").textContent.startsWith("Version"));
  assert(d.get("overview-status").textContent.length>0);
  assert(d.get("overview-score").textContent.length>0);
  assert.equal(d.get("sample-board").children.length,35);
  assert(d.get("mixture").children.length===3);
  assert(d.get("losses").textContent.includes("Policy loss"));
  d.get("inspect-budget").value="32";
  d.get("inspect-player").value="1";
  await d.get("analyze").events.click();
  assert(d.get("analysis-status").textContent.includes("No learning or game action performed"),d.get("analysis-status").textContent);
  assert.equal(d.get("analysis").children.length,7);
  const explanation=await page("explain"), e=explanation.elements;
  const technical=await page("explain-technical"),t=technical.elements;
  t.get("perspective").value="1";t.get("perspective").events.change();
  assert.equal(t.get("own-plane").children.filter(c=>c.textContent==="1").length,2);
  const ownAsFirst=t.get("own-plane").children.map(c=>c.textContent).join("");
  t.get("perspective").value="-1";t.get("perspective").events.change();
  assert.notEqual(t.get("own-plane").children.map(c=>c.textContent).join(""),ownAsFirst);
  e.get("reward-distance").value="10";e.get("reward-distance").events.input();
  assert(e.get("reward-win-label").textContent.includes("0.970"));
  e.get("reward-distance").value="0";e.get("reward-distance").events.input();
  assert(e.get("reward-win-label").textContent.includes("1.000"));
  assert(e.get("live-model").textContent.startsWith("Live:"),e.get("live-model").textContent);
  const play=await page("play"),p=play.elements;
  assert.equal(p.get("play-board").children.length,35);
  assert.equal(p.get("column-buttons").children.length,7);
  assert(!play.calls.some(call=>call.url==="/mode"),"Opening the play page must not start or reset a shared game");
  await p.get("start-game").events.click();
  assert.equal(p.get("game-status").textContent,"Your turn. Choose a column.");
  await p.get("column-buttons").children[0].events.click();
  assert.equal(p.get("move-count").textContent,"2 / 35 discs");
  assert.equal(play.calls.filter(call=>call.url==="/human/move").length,1);
  assert.equal(p.get("play-board").children.filter(cell=>cell.className.includes("yellow")).length,1);
  console.log("PASS: debug, explanation controls, and play startup/move rendering without changing the live game");
})().catch(error=>{console.error(error);process.exitCode=1;});
