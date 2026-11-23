cd /c/Users/nikhi/Desktop/IE643_final/poster_build
E="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
"$E" --headless --disable-gpu --virtual-time-budget=3000 --dump-dom "file:///$(pwd -W)/poster.html" 2>/dev/null | python -c "import sys,re;t=sys.stdin.read();print(re.search(r'<pre id=\"m\">(.*?)</pre>',t,re.S).group(1))"
