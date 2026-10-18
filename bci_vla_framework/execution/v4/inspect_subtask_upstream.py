"""Fetch the explicitly requested public implementation, without installing it."""
import importlib.util
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REMOTE = '/home/oyjt/Scene-aware-BCI-VLA-runs/v4_step11_upstream_20261018_v1'

def main():
    spec = importlib.util.spec_from_file_location('reader', WORKSPACE/'.server-inspect/remote_readonly.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    client = mod.connect()
    try:
        cmd = ('mkdir -p '+REMOTE+' && cd '+REMOTE+' && '
               'timeout 90 git clone --depth 1 https://github.com/LisavilaLee/openpi_with_subtask.git upstream '
               '&& cd upstream && git rev-parse HEAD && git status --porcelain '
               '&& git archive --format=zip --output=../upstream.zip HEAD')
        _, stdout, stderr = client.exec_command(cmd, timeout=110)
        output = stdout.read().decode(errors='replace'); error = stderr.read().decode(errors='replace')
        code = stdout.channel.recv_exit_status()
        (OUT/'11_upstream_fetch.json').write_text(json.dumps({'exit_code':code,'stdout':output,'stderr':error,
            'remote':REMOTE, 'url':'https://github.com/LisavilaLee/openpi_with_subtask'},indent=2)+'\n')
        print(json.dumps({'exit_code':code,'stdout':output,'stderr':error[-2000:]}))
        if code == 0:
            sftp = client.open_sftp(); sftp.get(REMOTE+'/upstream.zip',str(OUT/'11_upstream.zip')); sftp.close()
    finally:
        client.close()

if __name__ == '__main__': main()
