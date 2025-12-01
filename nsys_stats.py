import subprocess
import shlex
import argparse
import json
import sys
from typing import Union, Optional, Dict, Any

def run_command(cmd: Union[str, list], shell: bool = False, timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Execute a command and return stdout, stderr and exit code.
    cmd: str or list. If str and shell=False it will be split with shlex.
    shell: whether to run via the shell.
    timeout: seconds or None.
    """
    if isinstance(cmd, str) and not shell:
        cmd = shlex.split(cmd)
    proc = subprocess.run(
        cmd,
        shell=shell,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}

def nsys():
    for size in ("small", "medium", "large", "xl", "2.7B"):
        for length in (128, 256, 512, 1024):
            cmd = ["nsys", "stats", "-r", "nvtx_sum", f"/opt/cs336_systems/nsys/nsys_profile_{size}_context_length_{length}.sqlite"]
            stdout = run_command(cmd)["stdout"]
            # print(stdout)
            try:
                forward = stdout.split("\n")[8]
                backward = stdout.split("\n")[7]
                # print(forward)
                # print(backward)
                forward_sum = float(forward.split()[1].replace(",", "")) / 1e6
                forward_mean = float(forward.split()[3].replace(",", "")) / 1e6
                forward_std = float(forward.split()[7].replace(",", "")) / 1e6
                backward_sum = float(backward.split()[1].replace(",", "")) / 1e6
                backward_mean = float(backward.split()[3].replace(",", "")) / 1e6
                backward_std = float(backward.split()[7].replace(",", "")) / 1e6
                
                print(f"| {size} |  768    | 3072 |         12 |        12 |         {length}    |     {forward_sum:.5f} |       {forward_mean:.5f} |     {forward_std:.5f} |     {backward_sum:.5f} |      {backward_mean:.5f} |      {backward_std:.5f} |")
            except:
                pass
            

def profile():
    with open("profile.log", "r") as f:
        for line in f.readlines():
            splited = line.split()
            print(f"| {splited[1]}  |  {splited[2]}    | {splited[3]}  |         {splited[4]} |        {splited[5]} |         {splited[6]}    |   {float(splited[7]) * 1000:.5f}     |     {float(splited[8]) * 1000:.5f}     |     {float(splited[9]) * 1000:.5f}     |    {float(splited[10]) * 1000:.5f}     |      {float(splited[11]) * 1000:.5f}     |      {float(splited[12]) * 1000:.5f}    |")
            print("+--------+---------+-------+------------+-----------+----------------+-----------------+------------------+-----------------+------------------+-------------------+------------------+")

if __name__ == "__main__":
    profile()