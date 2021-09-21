import os, sys
def add_pkgs_to_path(pkg_dir):
  pkgs = ['']
  pkgs += [pkg for pkg in os.listdir(pkg_dir) if pkg.endswith('.egg')]
  for pkg in pkgs:
    pkg_path = os.path.join(pkg_dir, pkg)
    if pkg_path not in sys.path:
      sys.path.insert(0, pkg_path)
      os.environ['PYTHONPATH'] = os.pathsep.join([pkg_path, os.environ['PYTHONPATH']])