import {infer} from './inference.mjs';
const $=id=>document.getElementById(id);
const pad=$('pad'), ctx=pad.getContext('2d',{willReadFrequently:true});
const preview=$('preview'), previewCtx=preview.getContext('2d');
const scratch=document.createElement('canvas');scratch.width=scratch.height=28;
const small=scratch.getContext('2d',{willReadFrequently:true});
let model,weights,activePointer=null,hasInk=false,timer,lastPoint,lastResult=null;
const rows=Array.from({length:10},(_,digit)=>{
  const row=document.createElement('div');row.className='bar-row';
  row.innerHTML=`<span class="bar-label">${digit}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value">—</span>`;
  $('bars').append(row);return row;
});
function resetResult(){
  lastResult=null;$('digit').textContent='?';$('result-label').textContent='숫자를 기다리고 있어요';
  $('confidence').textContent='숫자를 쓰면 자동으로 인식해요';
  rows.forEach(row=>{row.classList.remove('winner');row.children[1].firstChild.style.width='0%';row.children[2].textContent='—';});
}
function clearPad(){
  clearTimeout(timer);
  if(activePointer!==null && pad.hasPointerCapture(activePointer))pad.releasePointerCapture(activePointer);
  activePointer=null;ctx.clearRect(0,0,560,560);hasInk=false;
  $('pad-hint').classList.remove('hidden');$('predict').disabled=true;
  previewCtx.fillStyle='#000';previewCtx.fillRect(0,0,28,28);resetResult();
  return {cleared:true};
}
function getPoint(event){const r=pad.getBoundingClientRect();return {x:(event.clientX-r.left)*560/r.width,y:(event.clientY-r.top)*560/r.height};}
function start(event){
  if(activePointer!==null || (event.pointerType==='mouse' && event.button!==0))return;
  event.preventDefault();clearTimeout(timer);activePointer=event.pointerId;pad.setPointerCapture(activePointer);
  lastPoint=getPoint(event);ctx.fillStyle='#172033';ctx.strokeStyle='#172033';ctx.lineWidth=25;ctx.lineCap='round';ctx.lineJoin='round';
  ctx.beginPath();ctx.arc(lastPoint.x,lastPoint.y,12.5,0,Math.PI*2);ctx.fill();
  hasInk=true;$('pad-hint').classList.add('hidden');$('predict').disabled=!model;
}
function move(event){
  if(event.pointerId!==activePointer)return;event.preventDefault();
  const coalesced=event.getCoalescedEvents?.();
  const samples=coalesced?.length ? coalesced : [event];
  for(const sample of samples){const p=getPoint(sample);ctx.beginPath();ctx.moveTo(lastPoint.x,lastPoint.y);ctx.lineTo(p.x,p.y);ctx.stroke();lastPoint=p;}
}
function end(event){
  if(event.pointerId!==activePointer)return;
  if(pad.hasPointerCapture(activePointer))pad.releasePointerCapture(activePointer);
  activePointer=null;clearTimeout(timer);timer=setTimeout(predict,350);
}
// MNIST convention: light ink on black, longest side 20px, center of mass at 13.5.
function preprocess(){
  const image=ctx.getImageData(0,0,560,560);let x0=560,y0=560,x1=-1,y1=-1;
  for(let y=0;y<560;y++)for(let x=0;x<560;x++)if(image.data[(y*560+x)*4+3]>20){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}
  if(x1<0)return null;
  const w=x1-x0+1,h=y1-y0+1,scale=20/Math.max(w,h);
  small.clearRect(0,0,28,28);small.imageSmoothingEnabled=true;small.imageSmoothingQuality='high';
  small.drawImage(pad,x0,y0,w,h,(28-w*scale)/2,(28-h*scale)/2,w*scale,h*scale);
  const raw=small.getImageData(0,0,28,28).data;
  let mass=0,mx=0,my=0;for(let i=0;i<784;i++){let v=raw[4*i+3]/255;mass+=v;mx+=(i%28)*v;my+=Math.floor(i/28)*v;}
  if(mass<.1)return null;
  const dx=Math.round(13.5-mx/mass),dy=Math.round(13.5-my/mass),input=new Float32Array(784);
  const output=previewCtx.createImageData(28,28);
  for(let y=0;y<28;y++)for(let x=0;x<28;x++){
    const sx=x-dx,sy=y-dy,i=y*28+x;
    const v=sx>=0&&sx<28&&sy>=0&&sy<28?raw[(sy*28+sx)*4+3]:0;
    input[i]=v/255;output.data[4*i]=output.data[4*i+1]=output.data[4*i+2]=v;output.data[4*i+3]=255;
  }
  previewCtx.putImageData(output,0,0);return input;
}
function predict(){
  clearTimeout(timer);if(!hasInk || activePointer!==null)return null;
  const input=preprocess();if(!input){resetResult();return null;}if(!model)return null;
  const probabilities=infer(input,weights,model.layers);
  const digit=probabilities.indexOf(Math.max(...probabilities)),confidence=probabilities[digit];
  $('digit').textContent=digit;$('result-label').textContent=confidence<.65?'조금 헷갈리지만, 이 숫자 같아요':'이 숫자로 읽었어요';
  $('confidence').textContent=`예측 확률 ${(confidence*100).toFixed(1)}%`;
  rows.forEach((row,i)=>{row.classList.toggle('winner',i===digit);row.children[1].firstChild.style.width=`${probabilities[i]*100}%`;row.children[2].textContent=(probabilities[i]*100).toFixed(1);});
  lastResult={digit,probabilities};return lastResult;
}
async function loadModel(){
  try{
    const [metaResponse,weightResponse]=await Promise.all([fetch('./model.json'),fetch('./model.bin')]);
    if(!metaResponse.ok||!weightResponse.ok)throw new Error('Model request failed');
    const meta=await metaResponse.json();const buffer=await weightResponse.arrayBuffer();
    const count=meta.layers.reduce((total,l)=>total+l.input*l.output+l.output,0);
    if(buffer.byteLength!==count*4)throw new Error('Invalid model weights');
    weights=new Float32Array(buffer);model=meta;
    $('model-status').textContent='MNIST 모델 준비 완료';$('model-status').classList.remove('error');
    $('predict').disabled=!hasInk;
    $('accuracy').textContent=`MNIST 테스트 정확도 ${(meta.testAccuracy*100).toFixed(2)}% · ${meta.testSamples.toLocaleString('ko-KR')}장`;
    if(hasInk)predict();
  }catch(error){
    $('model-status').textContent='모델을 불러오지 못했어요. 새로고침해 주세요.';$('model-status').classList.add('error');
    $('confidence').textContent='연결을 확인한 뒤 새로고침해 주세요.';console.error(error);
  }
}
pad.addEventListener('pointerdown',start);pad.addEventListener('pointermove',move);
pad.addEventListener('pointerup',end);pad.addEventListener('pointercancel',end);pad.addEventListener('lostpointercapture',end);
$('clear').addEventListener('click',clearPad);$('predict').addEventListener('click',predict);
clearPad();loadModel();
if(document.modelContext?.registerTool){
  const lifecycle=new AbortController();
  const register=(tool)=>{try{Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(console.error);}catch(e){console.error(e);}};
  const schema={type:'object',properties:{},additionalProperties:false};
  const validate=(input)=>{if(input===null || typeof input!=='object' || Array.isArray(input) || Object.keys(input).length)throw new Error('Expected an empty object');};
  register({name:'recognize_written_digit',title:'작성한 숫자 인식',description:'Recognize the digit currently drawn on the pad and display probabilities.',inputSchema:schema,annotations:{readOnlyHint:false},execute(input){validate(input);if(!model)throw new Error('Model not ready');if(!hasInk)throw new Error('Draw a digit first');if(activePointer!==null)throw new Error('Finish drawing first');return predict();}});
  register({name:'clear_digit_pad',title:'숫자 패드 지우기',description:'Clear the current drawing and prediction.',inputSchema:schema,annotations:{readOnlyHint:false},execute(input){validate(input);return clearPad();}});
  window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true});
}
