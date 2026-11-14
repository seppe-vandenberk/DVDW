/* DOM logic check; layout must also be reviewed in a real browser. */
const fs=require("node:fs"),vm=require("node:vm"),assert=require("node:assert/strict");
class Element {
  constructor(){this.children=[];this.hidden=false;this.value="";this.style={setProperty(k,v){this[k]=v;}};this.dataset={};this.attributes={};this.events={};this.textContent="";this.classes=new Set();this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x)};}
  animate(frames,options){(this.animations??=[]).push({frames,options});}
  replaceChildren(){this.children=[];}
  append(x){this.children.push(x);}setAttribute(k,v){this.attributes[k]=v;}getAttribute(k){return this.attributes[k];}
  addEventListener(k,v){this.events[k]=v;}querySelector(){return this.button??=new Element();}
}
async function page(path){
  const html=fs.readFileSync("connect4/static/event.html","utf8");
  assert(!html.includes("<header>"));assert(!html.includes("<nav>"));
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
  elements.get("manual-player").value="human";
  const document={body:new Element(),getElementById:id=>{assert(elements.has(id),id);return elements.get(id);},createElement:()=>new Element()};
  let state={setup:{robot_color:"B",first:"human",robot_character:"orbit",human_character:"astronaut"},board:Array.from({length:7},()=>Array(5).fill("X")),game_id:"stem-test",revision:1,status:"waiting",suggestion:null,thinking_until:0,next:"human",winner:null,move_count:0,camera:{ok:true,last_success:new Date().toISOString(),latency_ms:1},model:{ok:null},manual:false,camera_source:"mock"};
  const calls=[],timers=[];let paused=false,delay=500;
  const sandbox={document,location:{pathname:path},Date,JSON,Number,Math,Array,Object,String,Boolean,AbortSignal,
    setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length;},clearTimeout:()=>{},
    fetch:async(url,options={})=>{calls.push({url,options});
      if(url==="/event/models")return {ok:true,json:async()=>({models:[{id:"shared",label:"Live",playable:true},{id:"saved-"+"a".repeat(24),label:"Generation 1",playable:true,generation:1}],training:{pause_requested:paused,move_delay_ms:delay,games:20,phase:paused?"paused":"collecting experience"}})};
      if(url==="/event/training"){let p=JSON.parse(options.body);if(p.action==="pause")paused=true;if(p.action==="resume")paused=false;if(p.action==="delay")delay=p.delay_ms;return {ok:true,json:async()=>({accepted:true})};}
      return {ok:true,json:async()=>JSON.parse(JSON.stringify(state))};}};
  const context=vm.createContext(sandbox);vm.runInContext(fs.readFileSync("connect4/static/event.js","utf8"),context);
  await new Promise(r=>setImmediate(r));
  const render=s=>{state=s;context.input=s;vm.runInContext("render(input)",context);};
  assert.equal(elements.get("board").children.length,35);assert.equal(elements.get("slots").children.length,7);
  assert.equal(elements.get("admin").hidden,path!=="/instructor");assert.equal(elements.get("headline").textContent,"Your move, human!");
  assert.equal(elements.get("current-turn").textContent,"Jouw beurt");
  assert(elements.get("human-label").textContent.includes("Rood"));
  assert.equal(elements.get("robot-label").textContent,"Robot - Blauw");
  const placedBoard=state.board.map(col=>[...col]);placedBoard[2][4]="R";
  render({...state,board:placedBoard,revision:2});
  const placedCell=elements.get("board").children[4*7+2];
  assert.equal(placedCell.animations.length,1);
  assert(placedCell.animations[0].frames[0].transform.includes("-5"));
  render(state);assert.equal(placedCell.animations.length,1);
  render({...state,status:"thinking",next:"robot"});assert.equal(elements.get("current-turn").textContent,"Robots beurt");assert.equal(elements.get("robot-image").getAttribute("src"),"/static/robot/thinking.gif");
  render({...state,status:"suggestion",suggestion:5});assert.equal(elements.get("headline").textContent,"My pick: column 6!");
  assert(elements.get("robot").className.includes("carrying"));
  timers.findLast(t=>t.ms===1000).fn();assert(elements.get("slots").children[5].classes.has("active"));
  const before=timers.filter(t=>t.ms===1000).length;render(state);assert.equal(timers.filter(t=>t.ms===1000).length,before);
  render({...state,revision:2,status:"waiting",suggestion:null,next:"human",manual:true});
  assert(!elements.get("slots").children[5].classes.has("active"));assert(elements.get("manual-status").textContent.includes("Manual takeover"));
  if(path==="/instructor"){
    assert.equal(elements.get("opponent-model").children.length,2);
    elements.get("camera-delay").value="1250";
    await elements.get("apply-camera-delay").events.click();
    assert(calls.some(c=>c.url==="/event/camera/settings"&&JSON.parse(c.options.body).delay_ms===1250));
    render({...state,camera_poll_delay_ms:1250,manual:false});
    assert(elements.get("source").textContent.includes("1250 ms"));
    elements.get("training-delay").value="750";
    await elements.get("apply-delay").events.click();
    assert(calls.some(c=>c.url==="/event/training"&&JSON.parse(c.options.body).delay_ms===750));
    await elements.get("training-toggle").events.click();
    assert.equal(elements.get("training-toggle").textContent,"Resume training");
    await elements.get("training-toggle").events.click();
    assert.equal(elements.get("training-toggle").textContent,"Pause training");

    elements.get("size-editor").hidden=true;
    elements.get("size-toggle").events.click();
    assert.equal(elements.get("size-editor").hidden,false);
    await elements.get("size-larger").events.click();
    assert(calls.some(c=>c.url==="/event/size"&&JSON.parse(c.options.body).scale===105));
    elements.get("size-close").events.click();
    assert.equal(elements.get("size-editor").hidden,true);
    render({...state,board_scale:125});
    assert.equal(document.body.style["--board-width-scale"],1.25);
    await elements.get("manual-columns").children[2].events.click();
    const call=calls.find(c=>c.url==="/event/drop");assert.deepEqual(JSON.parse(call.options.body),{column:2,player:"human",revision:2});
    render({...state,status:"move_error",next:"robot",model:{ok:false,error:"Offline"}});
    assert.equal(elements.get("model-detail").textContent,"Offline");assert.equal(elements.get("retry").disabled,false);
    await elements.get("retry").events.click();assert(calls.some(c=>c.url==="/event/retry"));
  }
  render({...state,score_history:[{game:0,score:0,points:0},{game:1,score:1,points:1},{game:2,score:0,points:-1}]});
  assert.equal(elements.get("score-total").textContent,"0");
  assert(elements.get("score-chart").innerHTML.includes("<path"));
  assert(elements.get("score-chart").getAttribute("aria-label").includes("na 2 spellen"));
  render({...state,preview_winner:"human"});
  assert.equal(elements.get("result-title").textContent,"Jij wint!");
  assert(elements.get("robot").className.includes("sad"));
  render({...state,preview_winner:"robot"});
  assert.equal(elements.get("result-title").textContent,"Volgende keer beter!");
  assert(elements.get("robot").className.includes("happy"));
  render({...state,preview_winner:null,status:"finished",winner:"human"});assert.equal(elements.get("headline").textContent,"You connected four!");
  assert.equal(elements.get("current-turn").textContent,"Spel afgelopen");
  render({...state,setup:{...state.setup,robot_color:"R"}});
  assert(elements.get("human-label").textContent.includes("Blauw"));
  assert.equal(elements.get("robot-label").textContent,"Robot - Rood");
}
(async()=>{await page("/participant");await page("/instructor");console.log("Participant and instructor UI logic checks passed.");})().catch(error=>{console.error(error);process.exitCode=1;});
