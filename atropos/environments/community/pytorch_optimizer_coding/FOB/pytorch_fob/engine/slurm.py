"""
The MIT License (MIT)

Copyright (c) 2015 Brent Pedersen - Bioinformatics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Adapted from https://github.com/brentp/slurmpy

"""

from __future__ import print_function

import atexit
import datetime
import hashlib
import os
import subprocess
import sys
import tempfile
from typing import Optional, Sequence

TMPL = """\
#!/bin/bash

#SBATCH -e {log_dir}/{name}.%J.err
#SBATCH -o {log_dir}/{name}.%J.out
#SBATCH -J {name}

{header}

{bash_setup}

__script__"""


def tmp(suffix=".sh"):
    t = tempfile.mktemp(suffix=suffix)
    atexit.register(os.unlink, t)
    return t


class Slurm(object):
    def __init__(
        self,
        name,
        slurm_kwargs=None,
        tmpl=None,
        date_in_name=True,
        scripts_dir="slurm-scripts",
        log_dir="logs",
        bash_strict=True,
    ):
        if slurm_kwargs is None:
            slurm_kwargs = {}
        if tmpl is None:
            tmpl = TMPL
        self.log_dir = log_dir
        self.bash_strict = bash_strict

        header = []
        if "time" not in slurm_kwargs.keys():
            slurm_kwargs["time"] = "84:00:00"
        for k, v in slurm_kwargs.items():
            if len(k) > 1:
                k = "--" + k + "="
            else:
                k = "-" + k + " "
            header.append(f"#SBATCH {k}{v}")

        # add bash setup list to collect bash script config
        bash_setup = []
        if bash_strict:
            bash_setup.append("set -eo pipefail -o nounset")

        self.header = "\n".join(header)
        self.bash_setup = "\n".join(bash_setup)
        self.name = "".join(
            x for x in name.replace(" ", "-") if x.isalnum() or x == "-"
        )
        self.tmpl = tmpl
        self.slurm_kwargs = slurm_kwargs
        if scripts_dir is not None:
            self.scripts_dir = os.path.abspath(scripts_dir)
        else:
            self.scripts_dir = None
        self.date_in_name = bool(date_in_name)

    def __str__(self):
        return self.tmpl.format(
            name=self.name,
            header=self.header,
            log_dir=self.log_dir,
            bash_setup=self.bash_setup,
        )

    def _tmpfile(self):
        if self.scripts_dir is None:
            return tmp()
        else:
            for _dir in [self.scripts_dir, self.log_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
            return f"{self.scripts_dir}/{self.name}.sh"

    def run(
        self,
        command: str,
        name_addition: Optional[str] = None,
        cmd_kwargs: Optional[dict[str, str]] = None,
        _cmd: str = "sbatch",
        tries: int = 1,
        depends_on: Optional[Sequence[int]] = None,
        dependency_type: str = "afterok",
    ) -> Optional[int]:
        """
        command: a bash command that you want to run
        name_addition: if not specified, the sha1 of the command to run
                       appended to job name. if it is "date", the yyyy-mm-dd
                       date will be added to the job name.
        cmd_kwargs: dict of extra arguments to fill in command
                   (so command itself can be a template).
        _cmd: submit command (change to "bash" for testing).
        tries: try to run a job either this many times or until the first
               success.
        depends_on: job ids that this depends on before it is run
        dependency_type: after, afterok, afterany, afternotok
        """
        if name_addition is None:
            name_addition = hashlib.sha1(command.encode("utf-8")).hexdigest()

        if self.date_in_name:
            name_addition += "-" + str(datetime.date.today())
        name_addition = name_addition.strip(" -")

        if cmd_kwargs is None:
            cmd_kwargs = {}

        n = self.name
        self.name = self.name.strip(" -")
        self.name += "-" + name_addition.strip(" -")
        args = []
        for k, v in cmd_kwargs.items():
            args.append(f"export {k}={v}")
        args = "\n".join(args)

        tmpl = str(self).replace("__script__", args + "\n###\n" + command)
        if depends_on is None or (len(depends_on) == 1 and depends_on[0] is None):
            depends_on = []

        with open(self._tmpfile(), "w", encoding="utf8") as sh:
            sh.write(tmpl)

        job_id = None
        for itry in range(1, tries + 1):
            args = [_cmd]
            if depends_on is not None and len(depends_on) > 0:
                dep = f"--dependency={dependency_type}:" + ":".join(
                    [str(x) for x in depends_on]
                )
                args.append(dep)
            if itry > 1:
                mid = f"--dependency=afternotok:{job_id}"
                args.append(mid)
            args.append(sh.name)
            res = subprocess.check_output(args).strip()
            print(res.decode(), file=sys.stderr)
            self.name = n
            if not res.startswith(b"Submitted batch"):
                return None
            j_id = int(res.split()[-1])
            if itry == 1:
                job_id = j_id
        return job_id


if __name__ == "__main__":
    import doctest

    doctest.testmod()
