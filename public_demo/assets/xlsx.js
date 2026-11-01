/* Minimal dependency-free OOXML writer: inline strings, no formulas or macros. */
((root) => {
  const encoder = new TextEncoder();
  const xml = v => String(v).replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g,'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const crcTable = Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=(n&1)?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
  const crc = bytes => {let n=0xffffffff;for(const b of bytes)n=crcTable[(n^b)&255]^(n>>>8);return (n^0xffffffff)>>>0;};
  function zip(files) {
    const chunks=[], directory=[]; let offset=0;
    for (const [path,content] of Object.entries(files)) {
      const name=encoder.encode(path), data=encoder.encode(content), checksum=crc(data);
      const head=new Uint8Array(30+name.length), h=new DataView(head.buffer);
      h.setUint32(0,0x04034b50,true);h.setUint16(4,20,true);h.setUint16(10,0,true);h.setUint16(12,33,true);
      h.setUint32(14,checksum,true);h.setUint32(18,data.length,true);h.setUint32(22,data.length,true);h.setUint16(26,name.length,true);head.set(name,30);
      const central=new Uint8Array(46+name.length), c=new DataView(central.buffer);
      c.setUint32(0,0x02014b50,true);c.setUint16(4,20,true);c.setUint16(6,20,true);c.setUint16(14,33,true);
      c.setUint32(16,checksum,true);c.setUint32(20,data.length,true);c.setUint32(24,data.length,true);c.setUint16(28,name.length,true);c.setUint32(42,offset,true);central.set(name,46);
      chunks.push(head,data);directory.push(central);offset+=head.length+data.length;
    }
    const size=directory.reduce((sum,p)=>sum+p.length,0), end=new Uint8Array(22), e=new DataView(end.buffer);
    e.setUint32(0,0x06054b50,true);e.setUint16(8,directory.length,true);e.setUint16(10,directory.length,true);e.setUint32(12,size,true);e.setUint32(16,offset,true);
    const out=new Uint8Array(offset+size+end.length);let cursor=0;
    for (const part of [...chunks,...directory,end]) {out.set(part,cursor);cursor+=part.length;}
    return out;
  }
  function column(index) {let s='';for(index++;index;index=Math.floor((index-1)/26))s=String.fromCharCode(65+(index-1)%26)+s;return s;}
  function workbook(sheets) {
    const ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main';
    const files={'_rels/.rels':'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'};
    files['[Content_Types].xml']='<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'+sheets.map((_,i)=>'<Override PartName="/xl/worksheets/sheet'+(i+1)+'.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>').join('')+'</Types>';
    files['xl/workbook.xml']='<?xml version="1.0"?><workbook xmlns="'+ns+'" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'+sheets.map((s,i)=>'<sheet name="'+xml(s.name)+'" sheetId="'+(i+1)+'" r:id="rId'+(i+1)+'"/>').join('')+'</sheets></workbook>';
    files['xl/_rels/workbook.xml.rels']='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+sheets.map((_,i)=>'<Relationship Id="rId'+(i+1)+'" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet'+(i+1)+'.xml"/>').join('')+'</Relationships>';
    sheets.forEach((sheet,i)=>{files['xl/worksheets/sheet'+(i+1)+'.xml']='<?xml version="1.0"?><worksheet xmlns="'+ns+'"><sheetData>'+sheet.rows.map((row,r)=>'<row r="'+(r+1)+'">'+row.map((v,c)=>v==null?'':typeof v==='number'&&Number.isFinite(v)?'<c r="'+column(c)+(r+1)+'"><v>'+v+'</v></c>':'<c r="'+column(c)+(r+1)+'" t="inlineStr"><is><t xml:space="preserve">'+xml(v)+'</t></is></c>').join('')+'</row>').join('')+'</sheetData></worksheet>';});
    return zip(files);
  }
  root.DemoXlsx={workbook};
  if (typeof module !== 'undefined') module.exports = root.DemoXlsx;
})(globalThis);
