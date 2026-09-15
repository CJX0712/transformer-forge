// _uicheck.js — 无头 DOM 桩，驱动 UI 脚本端到端自检（不依赖浏览器）
// 验证：engine + ui 两段脚本可在 vm 中加载；点击「训练并评估」后 selftest 文本被填充且含 ALL GREEN。
const fs=require('fs'),vm=require('vm'),path=require('path');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const eng=html.match(/<script id="engine">([\s\S]*?)<\/script>/)[1];
const ui=html.match(/<script id="ui">([\s\S]*?)<\/script>/)[1];

const defaults={inpTask:'shift',inpVocab:'4',inpT:'8',inpD:'20',inpL:'1',inpSteps:'1000'};
const els={};
function makeEl(id){ const handlers={};
  return { id, _text:'', _html:'', value:defaults[id]||'', disabled:false,
    set textContent(v){ this._text=String(v); }, get textContent(){ return this._text; },
    set innerHTML(v){ this._html=String(v); }, get innerHTML(){ return this._html; },
    addEventListener(ev,fn){ handlers[ev]=fn; }, getContext(){ return new Proxy({},{get(){return ()=>{};}}); },
    _fire(ev){ if(handlers[ev]) handlers[ev](); } };
}
function getEl(id){ if(!els[id]) els[id]=makeEl(id); return els[id]; }
const document={ getElementById:getEl };
const sandbox={ console, Math, Object, Array, Float64Array, Infinity, NaN, isFinite, document };
sandbox.globalThis=sandbox; vm.createContext(sandbox);
vm.runInContext(eng, sandbox, {filename:'engine.js'});
vm.runInContext(ui, sandbox, {filename:'ui.js'});

// 模拟点击「训练并评估」
getEl('btnTrain')._fire('click');
const self=els['selftest']._text || '';
const pass=self.includes('ALL GREEN');
const accFilled=els['acc']._text.length>0, lossFilled=els['loss']._text.length>0, predFilled=els['predTable']._html.length>0;
console.log("selftest:\n"+self);
console.log("UI 元素已填充: acc="+accFilled+" loss="+lossFilled+" predTable="+predFilled);
console.log(pass && accFilled && lossFilled && predFilled ? "UICHECK PASS" : "UICHECK FAIL");
process.exit(pass && accFilled && lossFilled && predFilled ? 0 : 1);
