/* Browser verification uses the existing Playwright runtime, via Slurm. */
const fs=require('fs'),path=require('path'),assert=require('assert');
const http=require('http');
const {pathToFileURL}=require('url');
const {chromium}=require('/hpc/home/ns458/.cache/ms-playwright-go/1.50.1/package');
const root='/cwork/ns458/insula-paper/results/supplementary/operculum/viewer';
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const page=await browser.newPage({viewport:{width:1600,height:1100},acceptDownloads:true});
 const errors=[],network=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
 await page.goto(pathToFileURL(path.join(root,'operculum-inspector.html')).href,{waitUntil:'load',timeout:180000});
 async function ready(subject){await page.waitForFunction(s=>window.operculumInspector?.snapshot().ready&&window.operculumInspector.snapshot().subject===s&&!document.getElementById('main').classList.contains('loading')&&document.getElementById('brain')._fullLayout,subject,{timeout:180000})}
 await ready('D0064');
 assert.equal(await page.isChecked('#insula'),true);
 const subjects=await page.evaluate(()=>window.operculumInspector.subjects());
 const pixels=JSON.parse(fs.readFileSync(path.join(root,'native_pixel_checks.json'),'utf8'));
 const checks=[];
 for(const s of subjects){
  await page.selectOption('#subject',s.subject);await ready(s.subject);
  assert.equal(await page.isChecked('#insula'),true);
  await page.uncheck('#insula');
  await page.waitForFunction(()=>!document.getElementById('brain').data.some(t=>t.name==='Insula'));
  for(const p of pixels.filter(x=>x.subject===s.subject)){
   const color=await page.evaluate(p=>window.operculumInspector.rawPixel(p.plane,p.x,p.y),p);
   assert(color.every(x=>Math.abs(x-p.expected_gray)<=1),JSON.stringify({pixel:p,actual:color}));
  }
  assert.equal(await page.isChecked('#atlas'),false);
  const beforeHighlight=await page.evaluate(()=>window.operculumInspector.snapshot());
  await page.check('#insula');
  await page.waitForFunction(()=>document.getElementById('brain').data.some(t=>t.name==='Insula'&&t.i.length>0&&t.color==='#6A5ACD'&&t.opacity===.55));
  const afterHighlight=await page.evaluate(()=>window.operculumInspector.snapshot());
  assert.equal(afterHighlight.selected,beforeHighlight.selected);assert.deepEqual(afterHighlight.cursor,beforeHighlight.cursor);assert.equal(afterHighlight.HGA_channel,beforeHighlight.HGA_channel);
  const overlayChecks=await page.evaluate(()=>{
   const codes=new Set(record.insula_codes),index=atlas.findIndex(c=>codes.has(c));
   if(index<0)throw Error('No native insula mask');
   const shape=record.shape,ijk=[Math.floor(index/(shape[1]*shape[2])),Math.floor(index/shape[2])%shape[1],index%shape[2]];
   const xyz=record.voxel_to_tkRAS.slice(0,3).map(row=>row[0]*ijk[0]+row[1]*ijk[1]+row[2]*ijk[2]+row[3]);
   for(const [name,fixed] of [['sagittal',0],['coronal',1],['axial',2]]){
    const axis=record.voxel_to_tkRAS[fixed].slice(0,3).findIndex(x=>Math.abs(x)>1e-8),slider=document.getElementById(name+'-range');slider.value=ijk[axis];slider.dispatchEvent(new Event('input'));
   }
   const gray=Math.max(0,Math.min(255,(volume[index]-record.window[0])*255/Math.max(1,record.window[1]-record.window[0])));
   return ['sagittal','coronal','axial'].map(plane=>{const [x,y]=window.operculumInspector.pixelLocation(xyz,plane);return {plane,actual:window.operculumInspector.rawPixel(plane,x,y),expected:[106,90,205].map(c=>.35*c+.65*gray)}});
  });
  for(const c of overlayChecks)assert(c.actual.every((v,i)=>Math.abs(v-c.expected[i])<=1),JSON.stringify(c));
  await page.click('#home');
  const targets=[s.points.find(p=>p.kind==='contact'&&p.atlas_region==='Frontal operculum'),s.points.find(p=>p.kind==='contact'&&p.atlas_label==='Unknown'),s.points.find(p=>p.kind==='midpoint')].filter(Boolean);
  for(const target of targets){
   await page.locator('#contact-list button').filter({hasText:target.kind==='midpoint'?'◇ '+target.name:target.name}).filter({hasNotText:'__nonexistent__'}).evaluateAll((els,id)=>{const b=els.find(e=>e.dataset.id===id);if(!b)throw Error('Missing contact '+id);b.click()},target.id);
   await page.waitForFunction(id=>window.operculumInspector.snapshot().selected===id,target.id);
   const snapshot=await page.evaluate(()=>window.operculumInspector.snapshot());assert.deepEqual(snapshot.cursor,target.xyz);assert.deepEqual(snapshot.selected_xyz,target.xyz);
   for(const name of ['sagittal','coronal','axial']){
    const range=await page.locator('#'+name+'-range').inputValue();assert(Number.isFinite(Number(range)));
    const png=await page.locator('#'+name).evaluate(c=>c.toDataURL());assert(png.length>1000);
   }
   if(target.kind==='midpoint'){
    await page.waitForFunction(id=>document.getElementById('hga-details').dataset.channel===id.split(':midpoint:')[1],target.id);
    const task=await page.locator('#hga-task').inputValue();
    if(task){
     const result=await page.evaluate(({channel,task})=>({expected:window.operculumInspector.HGA(channel,task),traces:document.getElementById('hga').data.map(t=>({x:Array.from(t.x),y:Array.from(t.y)}))}),{channel:target.id.split(':midpoint:')[1],task});
     const expected=['Stimulus','Delay','Response'].map(p=>result.expected[p]).filter(p=>p?.available);
     assert.equal(result.traces.length,expected.length);
     for(let i=0;i<expected.length;i++){assert.deepEqual(result.traces[i].x,expected[i].times);assert.deepEqual(result.traces[i].y,expected[i].mean)}
    }
   }
  }
  const old=await page.evaluate(()=>window.operculumInspector.snapshot().cursor);
  await page.locator('[data-step="sagittal:1"]').click();const moved=await page.evaluate(()=>window.operculumInspector.snapshot().cursor);assert.notDeepEqual(old,moved);
  await page.click('#home');assert.deepEqual(await page.evaluate(()=>window.operculumInspector.snapshot().cursor),old);
  await page.check('#atlas');await page.waitForFunction(()=>document.getElementById('atlas-label').textContent.startsWith('Atlas 标签：'));await page.uncheck('#atlas');
  checks.push({subject:s.subject,real_contacts:s.points.filter(p=>p.kind==='contact').length,virtual_midpoints:s.points.filter(p=>p.kind==='midpoint').length,insula_surface_and_three_plane_highlight_verified:true,selection_and_three_slice_navigation_passed:true,HGA_pair_identity_and_waveforms_verified:true});
  console.log('Verified native MRI/HGA:',s.subject);
 }
 await page.click('#example');await ready('D0079');
 await page.waitForFunction(()=>document.getElementById('hga-details').dataset.case==='D0079_LAI7-8_LFMI2-3');
 assert.equal((await page.evaluate(()=>window.operculumInspector.snapshot())).HGA_channel,'D0079_LAI7-8');
 async function checkManualCase(){
  const result=await page.evaluate(()=>({traces:document.getElementById('hga').data.map(t=>({x:Array.from(t.x),y:Array.from(t.y),color:t.line.color})),expected:['D0079_LAI7-8','D0079_LFMI2-3'].map(ch=>window.operculumInspector.HGA(ch,'LexicalDelay')),contactTrace:document.getElementById('brain').data.find(t=>t.name==='contact'),raw:record.contacts.find(p=>p.channel==='D0079_LFMI2-3')}));
  assert.equal(result.traces.length,6);assert.equal(result.raw.atlas_label,'Left-Cerebral-White-Matter');assert.equal(result.raw.atlas_region,'Other');
  for(let phase=0;phase<3;phase++)for(let site=0;site<2;site++){
   const expected=result.expected[site][['Stimulus','Delay','Response'][phase]],trace=result.traces[2*phase+site];
   assert.deepEqual(trace.x,expected.times);assert.deepEqual(trace.y,expected.mean);assert.equal(trace.color,['#6A5ACD','#2369BD'][site]);
  }
  for(const contact of ['LFMI2','LFMI3']){const index=result.contactTrace.customdata.indexOf('D0079:contact:'+contact);assert(index>=0);assert.equal(result.contactTrace.marker.color[index],'#2369BD')}
 }
 await checkManualCase();
 await page.locator('#case-links button[data-channel="D0079_LFMI2-3"]').click();
 await page.waitForFunction(()=>window.operculumInspector.snapshot().HGA_channel==='D0079_LFMI2-3');await checkManualCase();
 const manualSelected=subjects.find(s=>s.subject==='D0079').points.find(p=>p.channel==='D0079_LFMI2-3');
 assert.deepEqual((await page.evaluate(()=>window.operculumInspector.snapshot())).cursor,manualSelected.xyz);
 await page.screenshot({path:path.join(root,'browser_test','D0079-manual-case.png'),fullPage:true});
 await page.click('#case-close');
 await page.waitForFunction(()=>document.getElementById('hga-details').dataset.case===''&&document.getElementById('hga').data.length===3);
 await page.selectOption('#subject','D0064');await ready('D0064');
 const fo=subjects.find(s=>s.subject==='D0064').points.find(p=>p.name==='LAIF7'&&p.kind==='contact');
 await page.evaluate(id=>window.operculumInspector.selectPoint(id),fo.id);
 const target3D=subjects.find(s=>s.subject==='D0064').points.find(p=>p.name==='LAI13'&&p.kind==='contact');
 const screen=await page.evaluate(id=>window.operculumInspector.screenLocation(id),target3D.id);
 assert(screen&&Number.isFinite(screen.x)&&Number.isFinite(screen.y));
 await page.mouse.click(screen.x,screen.y);
 await page.waitForFunction(id=>window.operculumInspector.snapshot().selected===id,target3D.id,{timeout:15000});
 await page.evaluate(id=>window.operculumInspector.selectPoint(id),fo.id);
 await page.selectOption('#decision','frontal_operculum');await page.fill('#note','Browser verification only');
 const downloadPromise=page.waitForEvent('download');await page.click('#export');const download=await downloadPromise;
 const csvPath=path.join(root,'browser_test','test-review.csv');fs.mkdirSync(path.dirname(csvPath),{recursive:true});await download.saveAs(csvPath);
 const csv=fs.readFileSync(csvPath,'utf8');assert(csv.includes('LAIF7')&&csv.includes('frontal_operculum')&&csv.includes('MRI_sha256'));
 fs.unlinkSync(csvPath);
 await page.selectOption('#decision','unreviewed');await page.fill('#note','');
 const bipolar=subjects.find(s=>s.subject==='D0064').points.find(p=>p.name==='LAI3-4'&&p.kind==='midpoint');
 await page.selectOption('#pair',bipolar.id);await page.waitForFunction(id=>window.operculumInspector.snapshot().HGA_channel===id.split(':midpoint:')[1],bipolar.id);
 await page.screenshot({path:path.join(root,'browser_test','D0064-viewer.png'),fullPage:true});
 await page.selectOption('#subject','D0057');await ready('D0057');assert((await page.evaluate(()=>window.operculumInspector.snapshot())).review_flags.length>0);
 await page.screenshot({path:path.join(root,'browser_test','D0057-review-flag.png'),fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 const server=http.createServer((req,res)=>{const name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);const file=path.join(root,'web',name==='/'?'index.html':path.basename(name));if(!fs.existsSync(file)){res.writeHead(404);return res.end()}res.setHeader('Content-Type',file.endsWith('.json')?'application/json':file.endsWith('.js')?'text/javascript':'text/html');fs.createReadStream(file).pipe(res)});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const url='http://127.0.0.1:'+server.address().port+'/';
 await page.goto(url,{waitUntil:'load',timeout:180000});await ready('D0064');
 assert.equal((await page.evaluate(()=>window.operculumInspector.snapshot())).HGA_channel,'D0064_LAI3-4');
 assert((await page.locator('#hga').evaluate(d=>d.data.length))>0);
 await page.screenshot({path:path.join(root,'browser_test','D0064-viewer-HGA.png'),fullPage:true});
 await page.selectOption('#subject','D0057');await ready('D0057');
 assert((await page.evaluate(()=>window.operculumInspector.snapshot())).review_flags.length>0);
 await page.click('#example');await ready('D0079');
 await page.waitForFunction(()=>document.getElementById('hga-details').dataset.case==='D0079_LAI7-8_LFMI2-3');await checkManualCase();
 assert.deepEqual(errors,[]);server.close();
 const receipt={passed:true,subjects:checks,manual_D0079_case_both_channels_MRI_HGA_and_original_atlas_labels_verified:true,offline_no_network_requests:true,web_lazy_loading_MRI_and_HGA_verified:true,native_voxel_pixel_checks:pixels.length,actual_3D_pointer_selection_verified:true,manual_review_CSV_export_verified:true,browser_errors:errors,main_analysis_modified:false};
 fs.writeFileSync(path.join(root,'browser_validation.json'),JSON.stringify(receipt,null,2)+'\n');console.log(JSON.stringify(receipt,null,2));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
