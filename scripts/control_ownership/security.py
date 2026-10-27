"""Provision and check the experiment's default-deny SROS2 permissions.

This module never starts ROS nodes, opens CAN sockets or controls hardware.
The trust boundary is the provisioned program identities, not a hostile process
running as the same Unix user (which could steal its private keys) or root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

POLICY = Path(__file__).with_name('policies') / 'experiment.xml'
ENCLAVES = ('backend', 'gateway', 'bootstrap', 'moveit', 'h2', 'observer')
DEFAULT_DOMAIN = 117


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(keystore, domain=None):
    """Fail closed on missing/changed policy, permissions, governance or keys."""
    root = Path(keystore).resolve()
    manifest = json.loads((root / 'experiment_manifest.json').read_text())
    if manifest['policy_sha256'] != sha256(POLICY):
        raise ValueError('SROS2 policy changed; provision a NEW keystore')
    if domain is not None and manifest['domain_id'] != int(domain):
        raise ValueError('SROS2 keystore was signed for another ROS domain')
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError('keystore must be owned by current user with mode0700')
    for name, digest in manifest['files'].items():
        if sha256(root / name) != digest:
            raise ValueError('SROS2 artifact changed: ' + name)
    governance = ET.parse(root / 'enclaves/governance.xml').getroot()
    rule = governance.find('domain_access_rules/domain_rule')
    if (rule.findtext('allow_unauthenticated_participants') != 'false'
            or rule.findtext('enable_join_access_control') != 'true'):
        raise ValueError('DDS authentication or access control is disabled')
    for enclave in ENCLAVES:
        grant = ET.parse(root / f'enclaves/openarm/{enclave}/permissions.xml').find('permissions/grant')
        if grant.findtext('default') != 'DENY':
            raise ValueError('permissions must default DENY')
    return manifest


def environment(keystore, domain=None):
    """Return enforced security settings; never silently fall back to insecure DDS."""
    manifest = validate(keystore, domain)
    return {
        'ROS_SECURITY_KEYSTORE': str(Path(keystore).resolve()),
        'ROS_SECURITY_ENABLE': 'true',
        'ROS_SECURITY_STRATEGY': 'Enforce',
        'ROS_DOMAIN_ID': str(manifest['domain_id']),
        'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp',
        'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST',
    }


def provision(keystore, domain=DEFAULT_DOMAIN):
    # create_enclave temporarily creates wildcard artifacts; generate everything
    # offline in a new private directory, then seal only final constrained files.
    from sros2.api._artifact_generation import generate_artifacts
    root = Path(keystore).resolve()
    if root.exists():
        raise ValueError('keystore already exists; use a new empty path')
    if not 0 <= int(domain) <= 232:
        raise ValueError('ROS domain must be in0..232')
    old_umask = os.umask(0o077)
    old_domain = os.environ.get('ROS_DOMAIN_ID')
    old_rmw = os.environ.get('RMW_IMPLEMENTATION')
    try:
        os.environ['ROS_DOMAIN_ID'] = str(domain)
        os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
        generate_artifacts(root, policy_files=[POLICY])
        root.chmod(0o700)
        (root / 'experiment_policy.xml').write_bytes(POLICY.read_bytes())
        artifacts = {str(p.relative_to(root)): sha256(p)
                     for p in sorted(root.rglob('*')) if p.is_file()}
        manifest = {
            'version': 1, 'domain_id': int(domain),
            'policy_sha256': sha256(POLICY), 'files': artifacts,
            'trust_boundary': 'provisioned programs; not hostile same-UID key theft or root',
            'hardware_operations': False,
        }
        (root / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        validate(root, domain)
        return manifest
    finally:
        os.umask(old_umask)
        for name, value in [('ROS_DOMAIN_ID', old_domain), ('RMW_IMPLEMENTATION', old_rmw)]:
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['provision', 'verify', 'env'])
    parser.add_argument('--keystore', required=True)
    parser.add_argument('--domain', type=int)
    args = parser.parse_args()
    if args.operation == 'provision':
        result = provision(args.keystore, DEFAULT_DOMAIN if args.domain is None else args.domain)
        print(json.dumps({'ready': True, 'domain_id': result['domain_id'], 'hardware_operations': False}))
    elif args.operation == 'verify':
        result = validate(args.keystore, args.domain)
        print(json.dumps({'valid': True, 'domain_id': result['domain_id']}))
    else:
        import shlex
        for key, value in environment(args.keystore, args.domain).items():
            print(f'export {key}={shlex.quote(value)}')
        print('unset ROS_SECURITY_ENCLAVE_OVERRIDE ROS_LOCALHOST_ONLY')


if __name__ == '__main__':
    main()
