"""Per-run cache of the unmodified stdlib package mapping for owned workers."""
import fcntl,hashlib,importlib.metadata,json,os,sys,tempfile
from pathlib import Path

def packages_distributions():
    # Explicit task-controlled reuse is permitted only while installed package
    # files remain unchanged. Ordered distribution metadata still invalidates it.
    run=os.environ.get('KIMI_PACKAGE_METADATA_CACHE_EPOCH') or os.environ.get('KIMI_RUN_TOKEN')
    workspace=os.environ.get('KIMI_ALIGNMENT_WORKSPACE')
    if not run or not workspace:return importlib.metadata.packages_distributions()
    # Same ordered distribution inventory; changes of location/version or metadata invalidate.
    inventory=[]
    for d in importlib.metadata.distributions():
        base=getattr(d,'_path',None)
        if base is None:return importlib.metadata.packages_distributions()
        stamps=[]
        for name in ('METADATA','PKG-INFO','RECORD','top_level.txt'):
            p=Path(base)/name
            try:
                s=p.stat();stamps.append((name,s.st_size,s.st_mtime_ns))
            except FileNotFoundError:pass
        inventory.append((str(base),stamps))
    signature=hashlib.sha256(json.dumps([run,sys.executable,sys.version,inventory]).encode()).hexdigest()
    folder=Path(workspace)/'cache/local/package-metadata';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(signature+'.json')
    with (folder/(signature+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return json.loads(path.read_text())
        # No inferred/omitted packages: compute the original mapping once per run.
        value=importlib.metadata.packages_distributions()
        fd,tmp=tempfile.mkstemp(dir=folder,prefix=signature,suffix='.tmp')
        try:
            with os.fdopen(fd,'w') as f:json.dump(value,f)
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
        return value
