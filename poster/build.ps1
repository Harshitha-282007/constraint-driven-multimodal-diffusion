# Build the A1 poster PDF from poster.html with headless Edge, then render a preview PNG.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$url = "file:///" + ($here -replace '\\','/') + "/poster.html"
$pdf = Join-Path $here "OneSheepArmy_IE643_CourseProject_Poster.pdf"
& $edge --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$pdf" $url | Out-Null
$dom = & $edge --headless --disable-gpu --virtual-time-budget=3000 --dump-dom $url
[regex]::Match(($dom -join "`n"), '<pre id="m"[^>]*>(.*?)</pre>', 'Singleline').Groups[1].Value
python -I -c "import pymupdf,sys; d=pymupdf.open(r'$pdf'); p=d[0]; print('pages',len(d),'mm',round(p.rect.width/72*25.4),round(p.rect.height/72*25.4)); p.get_pixmap(dpi=40).save(r'$here\preview.png')"
