"use strict";
const $ = id => document.getElementById(id);
const instructor = location.pathname === "/instructor";
document.body.classList.add(instructor?"instructor-view":"participant-view");
$("admin").hidden = !instructor;
$("participant-info").hidden = instructor;
let state = null, pending = false, initialized = false, revealTimer = null, revealKey = null;
const cells = [], slots = [], buttons = [];
const characters = {astronaut:"\u{1F9D1}\u200D\u{1F680}", scientist:"\u{1F9D1}\u200D\u{1F52C}", explorer:"\u{1F9ED}"};
for (let c=0;c<7;c++) {
  const slot=document.createElement("div");slot.className="slot";slot.textContent=c+1;$("slots").append(slot);slots.push(slot);
  const button=document.createElement("button");button.textContent=c+1;button.type="button";
  button.setAttribute("aria-label",`Place a disc in column ${c+1}`);
  button.addEventListener("click",()=>command("/event/drop",{column:c,player:$("manual-player").value,revision:state.revision}));
  $("manual-columns").append(button);buttons.push(button);
}
for(let r=0;r<5;r++) for(let c=0;c<7;c++) {const cell=document.createElement("div");$("board").append(cell);cells.push({cell,r,c});}
async function api(path,body) {
  const response=await fetch(path,{cache:"no-store",signal:AbortSignal.timeout(25000),...(body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})});
  const data=await response.json();if(!response.ok)throw Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail));return data;
}
function health(name,data) {
  $(name+"-health").textContent=data.ok===null?"Not called":data.ok?"Responding":"Needs attention";
  $(name+"-detail").textContent=data.error || (data.last_success?`Last success ${new Date(data.last_success).toLocaleTimeString()} - ${data.latency_ms??"n/a"} ms`:"Waiting for first response");
}
function renderScore(s) {
  const history=s.score_history??[{game:0,score:0,points:0}];
  const recent=history.slice(-31), values=recent.map(p=>p.score);
  const low=Math.min(-1,...values),high=Math.max(1,...values);
  const x=i=>34+i*550/Math.max(1,recent.length-1);
  const y=value=>8+(high-value)*50/(high-low);
  const path=recent.map((p,i)=>`${i?"L":"M"}${x(i).toFixed(2)},${y(p.score).toFixed(2)}`).join(" ");
  const score=history[history.length-1].score;
  $("score-total").textContent=score>0?`+${score}`:String(score);
  $("score-chart").innerHTML=`<line x1="34" y1="${y(0)}" x2="584" y2="${y(0)}" stroke="#cfc6dd" stroke-dasharray="4 4"/>
    <text x="2" y="13" fill="#82738f" font-size="10">${high}</text>
    <text x="2" y="60" fill="#82738f" font-size="10">${low}</text>
    <path d="${path}" fill="none" stroke="#7752c9" stroke-width="2.5" stroke-linejoin="round"/>
    ${recent.map((p,i)=>`<circle cx="${x(i)}" cy="${y(p.score)}" r="3" fill="${p.points<0?"#ef665d":p.points>0?"#658a21":"#7752c9"}"><title>Spel ${p.game}: score ${p.score}</title></circle>`).join("")}
    <text x="34" y="76" fill="#82738f" font-size="10">Spel ${recent[0].game}</text>
    <text x="584" y="76" text-anchor="end" fill="#82738f" font-size="10">${recent.length>1?`Spel ${recent[recent.length-1].game}`:"Nog geen spellen"}</text>`;
  $("score-chart").setAttribute("aria-label",`Robotscore ${score} na ${history[history.length-1].game} spellen. Winst plus 1, verlies min 1, gelijkspel 0.`);
}
function render(s) {
  const previous=state;
  const additions=previous?cells.filter(({r,c})=>previous.board[c][r]==="X"&&s.board[c][r]!=="X"):[];
  const unchangedDiscs=previous&&cells.every(({r,c})=>previous.board[c][r]==="X"||previous.board[c][r]===s.board[c][r]);
  const animatePlacement=unchangedDiscs&&additions.length>0&&additions.length<=2&&
    !(typeof matchMedia==="function"&&matchMedia("(prefers-reduced-motion: reduce)").matches);
  state=s;
  document.body.style.setProperty("--board-width-scale",(s.board_scale??100)/100);
  if(document.activeElement!==$("board-scale")) {
    $("board-scale").value=s.board_scale??100;
    $("size-value").textContent=`${s.board_scale??100}%`;
  }
  $("size-smaller").disabled=pending||(s.board_scale??100)<=50;
  $("size-larger").disabled=pending||(s.board_scale??100)>=175;
  renderScore(s);
  $("score-storage-error").hidden=!s.score_error;
  $("score-storage-error").textContent=s.score_error??"";
  const resultWinner=s.preview_winner??s.winner;
  const showResult=(s.preview_winner!=null||s.status==="finished")&&resultWinner!=null;
  $("result-screen").hidden=!showResult;
  $("board").style.visibility=showResult?"hidden":"visible";
  $("slots").style.visibility=showResult?"hidden":"visible";
  $("result-title").textContent=resultWinner==="human"?"Jij wint!":resultWinner==="robot"?"Volgende keer beter!":"Gelijkspel!";
  $("result-detail").textContent=s.preview_winner!=null?"Testweergave - telt niet mee voor de score":resultWinner==="human"?"Goed gespeeld! Nog een potje?":"Goed gespeeld! Probeer het nog eens.";
  const robotColor=s.setup.robot_color, humanColor=robotColor==="B"?"R":"B";
  const colorName=c=>c==="B"?"Blue":"Red";
  $("human-label").textContent=`${characters[s.setup.human_character]} Jij - ${humanColor==="B"?"Blauw":"Rood"}`;
  $("human-label").style.setProperty("--player-color",humanColor==="B"?"var(--blue)":"var(--red)");
  $("robot-label").textContent=`Robot - ${robotColor==="B"?"Blauw":"Rood"}`;
  $("robot-label").style.setProperty("--player-color",robotColor==="B"?"var(--blue)":"var(--red)");
  const turnText=showResult?"Spel afgelopen":s.next==="human"?"Jouw beurt":"Robots beurt";
  $("current-turn").textContent=turnText;
  $("current-turn").dataset.player=showResult?"finished":s.next;
  if(!initialized) {
    $("robot-color").value=robotColor;$("first").value=s.setup.first;
    $("robot-character").value=s.setup.robot_character;$("human-character").value=s.setup.human_character;
    initialized=true;colorHint();
  }
  for(const {cell,r,c} of cells){const value=s.board[c][r];cell.className=`cell ${value==="B"?"blue":value==="R"?"red":""}`;}
  if(animatePlacement) {
    additions.forEach(({cell,r},i)=>{
      cell.animate?.([
        {transform:`translateY(calc(-${r+1} * (100% + 6px)))`,opacity:.8},
        {transform:"translateY(0)",opacity:1}
      ],{duration:500+r*55,delay:i*180,easing:"cubic-bezier(.35,0,.75,.75)",fill:"backwards"});
    });
  }
  $("board").setAttribute("aria-label",s.board.map((col,c)=>`Column ${c+1}: ${col.map(v=>v==="X"?"empty":colorName(v)).join(", ")}`).join(". "));
  const thinking=!showResult&&s.status==="thinking", suggestion=!showResult&&s.status==="suggestion";
  const key=suggestion?`${s.game_id}:${s.revision}:${s.suggestion}`:null;
  $("robot").className=`robot ${s.setup.robot_character}${suggestion?" carrying":""}${showResult?(resultWinner==="human"?" sad":resultWinner==="robot"?" happy":""):""}`;
  $("robot").style.left=`${suggestion?s.suggestion*100/7:300/7}%`;
  $("carried-disc").style.background=robotColor==="B"?"var(--blue)":"var(--red)";
  const src=thinking?"/static/robot/thinking.gif":"/static/robot/waiting.gif";
  if($("robot-image").getAttribute("src")!==src)$("robot-image").setAttribute("src",src);
  $("robot-image").alt=thinking?"Robot thinking about its next move":suggestion?`Robot carrying a disc to column ${s.suggestion+1}`:"Robot waiting for your move";
  $("thought").hidden=!thinking;
  if(key!==revealKey){
    clearTimeout(revealTimer);revealKey=key;slots.forEach(slot=>slot.classList.remove("active"));
    if(suggestion)revealTimer=setTimeout(()=>{if(revealKey===key)slots[s.suggestion].classList.add("active");},1000);
  }
  let headline="Your move, human!",subline="Drop your disc into the physical board. I'll watch.";
  if(!s.camera.last_success&&!s.manual){headline="Let's find the board";subline="Waiting for the first camera observation.";}
  else if(thinking){headline="Hmm... let me think.";subline="Seven possibilities. Looking for a good one...";}
  else if(suggestion){headline=`My pick: column ${s.suggestion+1}!`;subline=`Place my ${colorName(robotColor).toLowerCase()} disc in the glowing slot.`;}
  else if(s.status==="move_error"){headline="A little help, please?";subline="The instructor can help me choose my next move.";}
  else if(s.status==="finished"){headline=s.winner==="human"?"You connected four!":s.winner==="robot"?"Four in a row. Nice game!":"It's a draw. What a match!";subline="Ask the instructor to start your next challenge.";}
  else if(s.next==="robot"){headline="Ready for my turn";subline="Waiting for the next confirmed board.";}
  $("headline").textContent=headline;$("subline").textContent=subline;
  $("count").textContent=`${s.move_count} / 35 discs`;
  $("turn").textContent=s.status==="finished"?"Spel afgelopen":s.next==="human"?"Jouw beurt":"Robots beurt";
  health("camera",s.camera);health("model",s.model);
  $("source").textContent=`${s.camera_source==="mock"?"Mock camera":"External camera"} - ${s.manual?"Manual takeover active":"Observations every 2 seconds"}`;
  $("mock-control").hidden=s.camera_source!=="mock";
  $("manual-status").textContent=s.manual?"Manual takeover active. Resume camera when the physical board is ready.":"Camera is in control.";
  buttons.forEach((b,c)=>b.disabled=pending||s.status==="finished"||s.next!==$("manual-player").value||s.board[c][0]!=="X");
  $("retry").disabled=pending||thinking||(s.next!=="robot"&&s.status!=="finished");
  $("resume").disabled=pending;$("setup").querySelector("button").disabled=pending;
  $("connection-warning").hidden=s.camera.ok||s.manual;
  $("connection-warning").textContent="Camera is unavailable or its board is invalid. Ask the instructor for help.";
}
async function command(path,body={}) {
  if(pending)return;pending=true;$("error").hidden=true;if(state)render(state);
  try {render(await api(path,body));}catch(error){$("error").textContent=error.message;$("error").hidden=false;}
  finally {pending=false;if(state)render(state);}
}
function colorHint(){$("human-color").textContent=$("robot-color").value==="B"?"Human: Red (R)":"Human: Blue (B)";}
$("robot-color").addEventListener("change",colorHint);
$("manual-player").addEventListener("change",()=>state&&render(state));
$("setup").addEventListener("submit",event=>{event.preventDefault();command("/event/new",{robot_color:$("robot-color").value,first:$("first").value,robot_character:$("robot-character").value,human_character:$("human-character").value});});
$("retry").addEventListener("click",()=>command("/event/retry"));
$("preview-human").addEventListener("click",()=>command("/event/preview",{winner:"human"}));
$("preview-robot").addEventListener("click",()=>command("/event/preview",{winner:"robot"}));
$("preview-clear").addEventListener("click",()=>command("/event/preview",{winner:null}));
function showSizeEditor(show) {
  if(!instructor)return;
  $("size-editor").hidden=!show;
  $("size-toggle").setAttribute("aria-expanded",String(show));
}
$("size-toggle").addEventListener("click",()=>showSizeEditor($("size-editor").hidden));
$("size-close").addEventListener("click",()=>showSizeEditor(false));
$("size-editor").addEventListener("keydown",event=>{if(event.key==="Escape")showSizeEditor(false);});
$("board-scale").addEventListener("input",()=>{$("size-value").textContent=`${$("board-scale").value}%`;});
$("board-scale").addEventListener("change",()=>command("/event/size",{scale:Number($("board-scale").value)}));
$("size-smaller").addEventListener("click",()=>command("/event/size",{scale:Math.max(50,(state?.board_scale??100)-5)}));
$("size-larger").addEventListener("click",()=>command("/event/size",{scale:Math.min(175,(state?.board_scale??100)+5)}));
$("size-reset").addEventListener("click",()=>command("/event/size",{scale:100}));
$("resume").addEventListener("click",()=>command("/event/resume"));
async function loadMock(){try{$("mock-json").value=JSON.stringify(await api("/event/camera/mock"),null,2);}catch(error){$("error").textContent=error.message;$("error").hidden=false;}}
$("mock-load").addEventListener("click",loadMock);
$("mock-send").addEventListener("click",async()=>{
  try{const data=JSON.parse($("mock-json").value);data.timestamp=new Date().toISOString();await api("/event/camera/mock",data);$("mock-json").value=JSON.stringify(data,null,2);$("error").hidden=true;}
  catch(error){$("error").textContent=error.message;$("error").hidden=false;}
});
async function poll(){
  try{const s=await api("/event/state");if(!pending)render(s);$("connection").textContent="Connected";$("connection").className="pill";}
  catch{$("connection").textContent="Reconnecting...";$("connection").className="pill offline";$("connection-warning").hidden=false;$("connection-warning").textContent="Connection lost. The displayed board may be out of date.";buttons.forEach(b=>b.disabled=true);}
  setTimeout(poll,500);
}
poll();if(instructor)loadMock();
