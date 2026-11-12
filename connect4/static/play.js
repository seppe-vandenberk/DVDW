"use strict";
const el = id => document.getElementById(id);
let current = null, busy = false, connected = false;
const blank = () => Array.from({length:7}, () => Array(5).fill(0));
const columnButtons = [], cells = [];
for (let col=0; col<7; col++) {
  const button=document.createElement("button");button.type="button";button.textContent="↓";
  button.setAttribute("aria-label",`Drop a yellow disc in column ${col+1}`);
  button.addEventListener("click",()=>move(col));el("column-buttons").append(button);columnButtons.push(button);
}
for (let row=0;row<5;row++) for (let col=0;col<7;col++) {
  const cell=document.createElement("span");cell.addEventListener("click",()=>move(col));
  el("play-board").append(cell);cells.push({cell,row,col});
}
function winningCells(board) {
  const result=new Set();
  for (let col=0;col<7;col++) for (let row=0;row<5;row++) for (const [dc,dr] of [[1,0],[0,1],[1,1],[1,-1]]) {
    const value=board[col][row];if (!value) continue;
    const line=Array.from({length:4},(_,i)=>[col+i*dc,row+i*dr]);
    if (line.every(([c,r])=>c>=0&&c<7&&r>=0&&r<5&&board[c][r]===value)) line.forEach(([c,r])=>result.add(`${c}:${r}`));
  }
  return result;
}
function render(snapshot) {
  if (snapshot && current && snapshot.updated_at<current.updated_at) return;
  if (snapshot) current=snapshot;
  const human=current?.mode==="human", board=human?current.board:blank();
  const playable=connected&&!busy&&human&&!current.done&&current.status==="human_turn"&&current.next_player===1;
  const finished=human&&current.done, failure=human&&current.status==="human_error";
  const winners=finished?winningCells(board):new Set();
  const lastRow=human&&current.last_column!=null?board[current.last_column].findIndex(v=>v!==0):-1;
  for (const {cell,row,col} of cells) {
    const value=board[col][row];
    cell.className=`cell${value===1?" yellow":value===-1?" red":""}${human&&col===current.last_column&&row===lastRow?" latest":""}${winners.has(`${col}:${row}`)?" winner":""}`;
    cell.title=`Column ${col+1}, row ${row+1}: ${value===1?"yellow":value===-1?"red":"empty"}`;
  }
  el("play-board").className=`play-board${playable?" playable":""}`;
  el("play-board").setAttribute("aria-label",`${human?current.move_count:0} discs on the board. ${board.map((column,i)=>`Column ${i+1}: ${column.join(",")}`).join(". ")}`);
  columnButtons.forEach((button,c)=>button.disabled=!playable||board[c][0]!==0);
  el("human-turn").hidden=!playable;el("ai-turn").hidden=!(human&&(busy||current.next_player===-1)&&!finished);
  el("start-game").disabled=busy||!connected;
  el("start-game").textContent=!human?"Start game":finished?"Play again":"Start over";
  el("retry-bot").hidden=!failure;el("retry-bot").disabled=busy||!connected;
  el("move-count").textContent=`${human?current.move_count:0} / 35 discs`;
  el("opponent").textContent="Learns through practice";
  let status="Ready for a challenge?",detail="Start a game, then choose a column to drop your yellow disc.";
  if (!connected) {status="Connecting to the game…";detail="Waiting for the game service. Your existing board will be restored.";}
  else if (busy) {status="The AI is thinking…";detail="Your move is being confirmed and the AI is looking ahead.";}
  else if (failure) {status="The AI could not finish its move.";detail="Retry the AI action. Your yellow disc will not be placed a second time.";}
  else if (finished) {status=current.winner===1?"You connected four!":current.winner===-1?"The AI connected four.":"A full board. It's a draw.";detail="The game outcome was confirmed. Choose Play again for another challenge.";}
  else if (human) {status=playable?"Your turn. Choose a column.":"Waiting for the AI…";detail="Connect four yellow discs, or block the AI's next line.";}
  el("game-status").textContent=status;el("game-detail").textContent=detail;
  el("last-move").textContent=human&&current.last_column!=null?`${current.last_player===1?"You":"AI"} played column ${current.last_column+1}`:"You are yellow. The AI is red.";
  el("connection").textContent=connected?"Game connected":"Reconnecting";el("connection").className=`badge ${connected?"live":"offline"}`;
  el("game-reference").textContent="Play the AI · your finished game gives it more experience";
}
async function request(path,payload) {
  const response=await fetch(path,payload===undefined?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await response.json();if(!response.ok) throw new Error(typeof data.detail==="string"?data.detail:JSON.stringify(data.detail));return data;
}
async function command(path,payload) {
  if(busy)return;busy=true;el("game-error").hidden=true;render();
  try {const snapshot=await request(path,payload);connected=true;render(snapshot);}
  catch(error) {el("game-error").textContent=error.message;el("game-error").hidden=false;try{render(await request("/state"));}catch{connected=false;}}
  finally {busy=false;render();}
}
async function move(column) {
  if(busy||!connected||current?.mode!=="human"||current.done||current.status!=="human_turn"||current.next_player!==1||current.board[column][0]!==0)return;
  return command("/human/move",{game_id:current.game_id,move_count:current.move_count,column});
}
el("start-game").addEventListener("click",()=>command(current?.mode==="human"?"/human/new":"/mode",current?.mode==="human"?{}:{mode:"human"}));
el("retry-bot").addEventListener("click",()=>current&&command("/human/retry",{game_id:current.game_id,move_count:current.move_count}));
async function poll() {
  if(!busy)try {const snapshot=await request("/state");connected=true;render(snapshot);}catch{connected=false;render();}
  setTimeout(poll,document.hidden?3000:1000);
}
async function modelInfo() {
  try {
    const model=current?.mode==="human"?current.model_two:"shared";
    const data=await request(`/debug/data?model=${encodeURIComponent(model)}`);
    el("generation").textContent=`Version ${data.published.generation}`;
    el("model-detail").textContent="It uses patterns learned through practice, then looks ahead before choosing a move.";
    const active=data.background.running&&!data.background.pause_requested;
    el("learning-status").textContent=active?"Background practice continues":data.background.pause_requested?"Practice paused by the operator":"Practice is currently stopped";
    el("practice-note").textContent=active?"The AI practices while you play.":"Play the AI's published experience.";
  }catch {el("model-detail").textContent="Model details are temporarily unavailable. Game controls reconnect independently.";}
  setTimeout(modelInfo,5000);
}
render();poll();modelInfo();
