const {chromium}=require('/hpc/home/ns458/.cache/ms-playwright-go/1.50.1/package');
const fs=require('fs');
(async()=>{
 const b=await chromium.launch({headless:true,args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const p=await b.newPage({viewport:{width:1600,height:1100}});p.on('pageerror',e=>console.log('PAGE ERROR',e.message));
 await p.goto('file:///cwork/ns458/insula-paper/results/supplementary/operculum/viewer/operculum-inspector.html',{timeout:180000});
 await p.waitForFunction(()=>window.operculumInspector?.snapshot().ready&&!document.getElementById('main').classList.contains('loading'),null,{timeout:180000});
 console.log('STATE',await p.evaluate(()=>({snapshot:window.operculumInspector.snapshot(),decision:document.getElementById('decision').outerHTML,decisionValues:[...document.getElementById('decision').options].map(x=>x.value),ids:document.querySelectorAll('#decision').length})));
 console.log('SCENE',await p.evaluate(()=>{const s=document.getElementById('brain')._fullLayout.scene._scene;return {keys:Object.keys(s),glplot:Object.keys(s.glplot),cameraKeys:Object.keys(s.glplot.camera||{}),cameraParams:s.glplot.cameraParams,view:s.glplot.camera?.view,projection:s.glplot.camera?.projection,dataScale:s.dataScale,axes:s.fullSceneLayout?.aspectratio}}));
 fs.mkdirSync('/cwork/ns458/insula-paper/results/supplementary/operculum/viewer/browser_test',{recursive:true});
 await p.screenshot({path:'/cwork/ns458/insula-paper/results/supplementary/operculum/viewer/browser_test/debug.png',fullPage:true});
 await b.close();
})().catch(e=>{console.error(e);process.exit(1)});
