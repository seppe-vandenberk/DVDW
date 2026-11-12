"use strict";
const exampleBoard = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,-1],[0,0,0,-1,1],[0,0,0,0,1],[0,0,0,0,0],[0,0,0,0,0]];
function encodeExample() {
  const player=Number(document.getElementById("perspective").value);
  const containers=["encoding-board","own-plane","other-plane","empty-plane"].map(id=>document.getElementById(id));
  containers.forEach(container=>container.replaceChildren());
  for(let row=0;row<5;row++) for(let col=0;col<7;col++) {
    const value=exampleBoard[col][row], normalized=value*player;
    const disc=document.createElement("span"); disc.className=value===1?"yellow":value===-1?"red":"empty"; containers[0].append(disc);
    [normalized===1,normalized===-1,normalized===0].forEach((on,index)=>{
      const cell=document.createElement("span"); cell.textContent=on?"1":"0"; cell.className=on?"on":"";containers[index+1].append(cell);
    });
  }
}
function updateReward() {
  const distance=Number(document.getElementById("reward-distance").value), magnitude=0.997**distance;
  document.getElementById("reward-plies").textContent=distance;
  document.getElementById("reward-win-dot").setAttribute("cx",325+260*magnitude);
  document.getElementById("reward-loss-dot").setAttribute("cx",325-260*magnitude);
  document.getElementById("reward-win-label").textContent=`+${magnitude.toFixed(3)} win target`;
  document.getElementById("reward-loss-label").textContent=`−${magnitude.toFixed(3)} loss target`;
}
document.getElementById("perspective").addEventListener("change",encodeExample);
document.getElementById("reward-distance").addEventListener("input",updateReward);
encodeExample();updateReward();
async function liveModel() {
  try {
    const response=await fetch("/debug/data?model=shared",{cache:"no-store"});
    if(!response.ok) throw new Error("No live telemetry");
    const model=await response.json(), config=model.background.config;
    document.getElementById("live-model").textContent=`Live: generation ${model.published.generation} · ${config.training_simulations} training / ${config.serving_simulations} visitor search simulations · ${model.device.toUpperCase()}. Diagram architecture is policy-value-7x5-v1.`;
  } catch {
    document.getElementById("live-model").textContent="Live telemetry unavailable. The explanation and diagrams remain available offline from the local server.";
  }
}
liveModel();
